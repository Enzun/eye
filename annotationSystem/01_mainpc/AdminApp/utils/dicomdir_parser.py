# dicomdir_parser.py
"""
DICOMDIRパーサー
共有フォルダのDICOMDIRからシリーズ情報を取得
"""

import os
import pydicom
from pathlib import Path
from typing import List, Dict, Any, Optional


class DICOMDIRParser:
    """DICOMDIRファイルを解析してシリーズ情報を取得するクラス"""

    def __init__(self, dicomdir_path: str):
        """
        Args:
            dicomdir_path: DICOMDIRファイルのパス
        """
        self.dicomdir_path = Path(dicomdir_path)
        self.dicomdir_folder = self.dicomdir_path.parent
        self.dicomdir = None

    def load(self) -> bool:
        """DICOMDIRファイルを読み込む"""
        try:
            self.dicomdir = pydicom.dcmread(str(self.dicomdir_path))
            return True
        except Exception as e:
            print(f"[DICOMDIRParser] Error loading DICOMDIR: {e}")
            return False

    def parse_series(self, series_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        DICOMDIRからシリーズ情報を抽出

        Args:
            series_filter: シリーズ説明のフィルタ（例: "eT1W_SE_cor"）

        Returns:
            List[Dict]: シリーズ情報のリスト
                - patient_id: 患者ID
                - patient_name: 患者名
                - study_date: 検査日（YYYYMMDD）
                - study_time: 検査時刻（HHMMSS）
                - study_description: 検査説明
                - series_number: シリーズ番号
                - series_description: シリーズ説明
                - modality: モダリティ（MR, CTなど）
                - image_files: 画像ファイルパスのリスト
                - entry_id: 一意のID（{PatientID}_{StudyDate}_{StudyTime}_{SeriesNumber}）
        """
        if self.dicomdir is None:
            print("[DICOMDIRParser] DICOMDIR not loaded. Call load() first.")
            return []

        series_list = []
        current_patient = {}
        current_study = {}
        current_series = {}

        try:
            # DirectoryRecordSequenceを直接ループ（環境依存を回避）
            for record in self.dicomdir.DirectoryRecordSequence:
                record_type = record.DirectoryRecordType

                if record_type == "PATIENT":
                    current_patient = {
                        "patient_id": str(getattr(record, 'PatientID', '')).strip(),
                        "patient_name": str(getattr(record, 'PatientName', '')).strip()
                    }

                elif record_type == "STUDY":
                    current_study = {
                        "study_date": getattr(record, 'StudyDate', ''),
                        "study_time": getattr(record, 'StudyTime', ''),
                        "study_description": getattr(record, 'StudyDescription', '')
                    }

                elif record_type == "SERIES":
                    # 前のシリーズを保存
                    if current_series and current_series.get("image_files"):
                        # 患者IDが空またはゼロ埋めの場合はDICOMファイルから直接取得
                        patient_id = current_patient.get("patient_id", "")
                        patient_id = self._resolve_patient_id(patient_id, current_series.get("image_files", []))
                        resolved_patient = {**current_patient, "patient_id": patient_id}

                        entry_id = self._generate_entry_id(
                            patient_id,
                            current_study.get("study_date", ""),
                            current_study.get("study_time", ""),
                            current_series.get("series_number")
                        )
                        series_info = {
                            **resolved_patient,
                            **current_study,
                            **current_series,
                            "entry_id": entry_id
                        }
                        series_list.append(series_info)

                    # 新しいシリーズを開始
                    series_description = getattr(record, 'SeriesDescription', '')

                    # シリーズフィルタが指定されている場合はチェック
                    if series_filter and series_filter not in series_description:
                        current_series = {}  # フィルタに一致しないのでリセット
                        continue

                    current_series = {
                        "series_number": getattr(record, 'SeriesNumber', None),
                        "series_description": series_description,
                        "modality": getattr(record, 'Modality', ''),
                        "image_files": []
                    }

                elif record_type == "IMAGE":
                    # 現在のシリーズが有効な場合のみ画像を追加
                    if current_series:
                        # ReferencedFileIDから画像ファイルパスを取得
                        if hasattr(record, 'ReferencedFileID'):
                            file_id = record.ReferencedFileID
                            # file_idはリスト形式（例: ['DICOM', 'IM_0001']）
                            # MultiValueクラスの場合があるので、リストに変換
                            if hasattr(file_id, '__iter__') and not isinstance(file_id, str):
                                file_id = list(file_id)
                                rel_path = os.path.join(*file_id)
                            else:
                                rel_path = str(file_id)

                            abs_path = self.dicomdir_folder / rel_path
                            current_series["image_files"].append(str(abs_path))

            # ループ終了後、最後のシリーズを保存
            if current_series and current_series.get("image_files"):
                # 患者IDが空またはゼロ埋めの場合はDICOMファイルから直接取得
                patient_id = current_patient.get("patient_id", "")
                patient_id = self._resolve_patient_id(patient_id, current_series.get("image_files", []))
                resolved_patient = {**current_patient, "patient_id": patient_id}

                entry_id = self._generate_entry_id(
                    patient_id,
                    current_study.get("study_date", ""),
                    current_study.get("study_time", ""),
                    current_series.get("series_number")
                )
                series_info = {
                    **resolved_patient,
                    **current_study,
                    **current_series,
                    "entry_id": entry_id
                }
                series_list.append(series_info)

        except Exception as e:
            print(f"[DICOMDIRParser] Error parsing DICOMDIR: {e}")

        return series_list

    def _resolve_patient_id(self, patient_id: str, image_files: list) -> str:
        """
        患者IDを解決する。DICOMDIRのPATIENTレコードの値が空またはゼロ埋めの場合、
        実際のDICOMファイルからPatientIDを直接取得する。

        Args:
            patient_id: DICOMDIRのPATIENTレコードから取得した患者ID
            image_files: このシリーズに属する画像ファイルパスのリスト

        Returns:
            str: 解決された患者ID（取得できなかった場合は元の patient_id を返す）
        """
        # 患者IDが有効（空でなく、ゼロ埋めでもない）ならそのまま使用
        if patient_id and patient_id.strip('0') != '':
            return patient_id

        # ゼロ埋めまたは空の場合、実際のDICOMファイルから読み取る
        for file_path in image_files[:3]:  # 最初の3ファイルまで試す
            if not os.path.exists(file_path):
                continue
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                raw_id = str(getattr(ds, 'PatientID', '')).strip()
                if raw_id and raw_id.strip('0') != '':
                    print(f"[DICOMDIRParser] PatientID resolved from DICOM file: "
                          f"'{patient_id}' → '{raw_id}'")
                    return raw_id
            except Exception as e:
                print(f"[DICOMDIRParser] Could not read PatientID from {file_path}: {e}")
                continue

        # フォールバック: 元の値をそのまま返す（全ゼロもそのまま）
        print(f"[DICOMDIRParser] Could not resolve PatientID from DICOM files. Using DICOMDIR value: '{patient_id}'")
        return patient_id

    def _generate_entry_id(self, patient_id: str, study_date: str, study_time: str, series_number: Optional[int]) -> str:
        """
        一意のエントリIDを生成

        Format: {PatientID}_{StudyDate}_{StudyTime}_EX1_SE{SeriesNumber}
        Example: 147757_20240508_112048_EX1_SE3

        Args:
            patient_id: 患者ID
            study_date: 検査日（YYYYMMDD）
            study_time: 検査時刻（HHMMSS）
            series_number: シリーズ番号

        Returns:
            str: エントリID
        """
        # 時刻をHHMMSS形式に整形（DICOMは HHMMSS.ffffff の場合がある）
        if study_time and '.' in study_time:
            study_time = study_time.split('.')[0]

        # 6桁に満たない場合はゼロパディング
        study_time = study_time.zfill(6) if study_time else "000000"

        series_num_str = str(series_number) if series_number is not None else "0"

        # EX1_SE{SeriesNumber}形式に統一（DICOMDIRにはEX番号がないためEX1固定）
        return f"{patient_id}_{study_date}_{study_time}_EX1_SE{series_num_str}"

    def get_first_image_path(self, series_info: Dict[str, Any]) -> Optional[str]:
        """
        シリーズの最初の画像ファイルパスを取得（マルチフレームの場合はそれ1つ）

        Args:
            series_info: parse_series()で取得したシリーズ情報

        Returns:
            str: 画像ファイルパス（存在しない場合はNone）
        """
        image_files = series_info.get("image_files", [])
        if not image_files:
            return None

        # Enhanced DICOMの場合、通常は1ファイルのみ
        # 従来DICOMの場合は複数ファイル（最初のファイルを返す）
        first_image = image_files[0]

        if os.path.exists(first_image):
            return first_image
        else:
            print(f"[DICOMDIRParser] Image file not found: {first_image}")
            return None


def find_dicomdir(folder_path: str) -> Optional[str]:
    """
    指定されたフォルダ内でDICOMDIRファイルを検索

    Args:
        folder_path: 検索するフォルダパス

    Returns:
        str: DICOMDIRファイルパス（見つからない場合はNone）
    """
    folder = Path(folder_path)

    # 大文字・小文字を区別せずに検索
    for filename in ["DICOMDIR", "dicomdir", "Dicomdir"]:
        dicomdir_path = folder / filename
        if dicomdir_path.exists():
            return str(dicomdir_path)

    return None
