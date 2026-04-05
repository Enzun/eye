"""
auto_loader.py
共有フォルダ（DICOMDIR形式）からの自動データ取得

処理フロー:
  1. config.json の shared_folder_path から DICOMDIR を検索
  2. DICOMDIR を解析してシリーズ情報を取得
  3. セッションに未登録のシリーズのみ DICOM → NIfTI 変換
  4. SessionManager に登録
"""

from pathlib import Path

import SimpleITK as sitk
from PyQt5.QtCore import QThread, pyqtSignal

from parsers.dicomdir_parser import DICOMDIRParser, find_dicomdir
from gui.dicom_handler import convert_multiframe_dicom_to_nifti


class AutoLoaderThread(QThread):
    """共有フォルダの自動スキャンを行うバックグラウンドスレッド"""

    progress_updated = pyqtSignal(int, int, str)  # (current, total, message)
    scan_completed   = pyqtSignal(int, int)        # (new_count, existing_count)
    error_occurred   = pyqtSignal(str)
    load_completed   = pyqtSignal()

    def __init__(self, shared_folder, series_filter, images_dir, session, predictions_dir, edited_dir):
        super().__init__()
        self.shared_folder   = Path(shared_folder)
        self.series_filter   = series_filter
        self.images_dir      = Path(images_dir)
        self.session         = session
        self.predictions_dir = Path(predictions_dir)
        self.edited_dir      = Path(edited_dir)
        self.should_cancel   = False

    def run(self):
        try:
            # Step 1: 共有フォルダの確認
            if not self.shared_folder.exists():
                self.error_occurred.emit(
                    f"共有フォルダにアクセスできません:\n{self.shared_folder}"
                )
                return

            # Step 2: DICOMDIR を検索
            self.progress_updated.emit(0, 100, "DICOMDIR を検索中...")
            dicomdir_path = find_dicomdir(str(self.shared_folder))
            if not dicomdir_path:
                self.error_occurred.emit(
                    f"DICOMDIR が見つかりませんでした:\n{self.shared_folder}"
                )
                return

            # Step 3: DICOMDIR を解析
            self.progress_updated.emit(0, 100, "DICOMDIR を解析中...")
            parser = DICOMDIRParser(dicomdir_path)
            if not parser.load():
                self.error_occurred.emit("DICOMDIR の読み込みに失敗しました。")
                return

            series_list = parser.parse_series(series_filter=self.series_filter)
            if not series_list:
                self.progress_updated.emit(100, 100, "対象シリーズが見つかりませんでした。")
                self.scan_completed.emit(0, 0)
                return

            # Step 4: 新規 / 既存を分類
            new_series = []
            existing_count = 0
            existing_ids = {c["case_id"] for c in self.session.get_cases()}
            for s in series_list:
                if s["entry_id"] in existing_ids:
                    existing_count += 1
                else:
                    new_series.append(s)

            self.scan_completed.emit(len(new_series), existing_count)

            if not new_series:
                self.progress_updated.emit(100, 100, "新規データはありません。")
                self.load_completed.emit()
                return

            # Step 5: 新規データを NIfTI 変換
            total = len(new_series)
            for i, series_info in enumerate(new_series):
                if self.should_cancel:
                    return

                case_id = series_info["entry_id"]
                self.progress_updated.emit(i, total, f"変換中 ({i+1}/{total}): {case_id}")

                try:
                    first_image = parser.get_first_image_path(series_info)
                    if not first_image:
                        print(f"[AutoLoader] 画像ファイルなし: {case_id}")
                        continue

                    sitk_image = convert_multiframe_dicom_to_nifti(first_image)

                    img_path = self.images_dir / f"{case_id}_0000.nii.gz"
                    sitk.WriteImage(sitk_image, str(img_path))

                    pred_path      = self.predictions_dir / f"{case_id}_pred.nii.gz"
                    corrected_path = self.edited_dir / f"{case_id}_corrected.nii.gz"
                    self.session.add_case(
                        case_id, img_path,
                        pred_path if pred_path.exists() else None,
                        corrected_path,
                    )
                    print(f"[AutoLoader] 完了: {case_id}")

                except Exception as e:
                    print(f"[AutoLoader] エラー {case_id}: {e}")
                    continue

            self.progress_updated.emit(total, total, "取得完了")
            self.load_completed.emit()

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"予期しないエラー:\n{e}\n{traceback.format_exc()}")

    def cancel(self):
        self.should_cancel = True
