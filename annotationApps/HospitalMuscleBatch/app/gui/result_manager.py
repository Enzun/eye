# result_manager.py
"""
結果管理モジュール
予測結果のNIfTI保存、CSV追記、レビューリスト管理
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import SimpleITK as sitk


class ResultManager:
    """結果保存・管理クラス"""
    
    # 筋肉ラベルの定義順（CSVヘッダー用）
    MUSCLE_LABELS = ['r_ir', 'l_ir', 'r_mr', 'l_mr', 'r_sr', 'l_sr', 
                     'r_so', 'l_so', 'r_lr', 'l_lr', 'r_io', 'l_io']
    
    def __init__(self, base_dir, model_id=119):
        """
        Args:
            base_dir: アプリのベースディレクトリ
            model_id: nnUNetのDatasetID（例: 119）。出力先フォルダ名に使用する。
        """
        self.base_dir = Path(base_dir)
        self.model_id = model_id
        model_dir = self.base_dir / "output" / str(model_id)
        self.predictions_dir = model_dir / "predictions"
        self.csv_file = model_dir / "csv" / "muscle_measurements.csv"
        self.review_file = self.base_dir / "manual_review" / "pending_review.json"

        # ディレクトリ作成
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        self.review_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_prediction_nifti(self, folder_name, pred_array, spacing=(1.0, 1.0, 1.0)):
        """
        予測マスクをNIfTI形式で保存
        
        Args:
            folder_name: DICOMフォルダ名 (例: EX100SE3)
            pred_array: 予測マスク配列 (Z, Y, X)
            spacing: スペーシング情報 (X, Y, Z)
        
        Returns:
            保存したファイルのパス
        """
        # SimpleITK画像に変換
        pred_sitk = sitk.GetImageFromArray(pred_array.astype(np.uint8))
        pred_sitk.SetSpacing(spacing)
        
        # 保存
        filename = f"{folder_name}_pred.nii.gz"
        filepath = self.predictions_dir / filename
        sitk.WriteImage(pred_sitk, str(filepath))
        
        return filepath
    
    def load_prediction_nifti(self, folder_name):
        """
        保存済みの予測マスクを読み込む
        
        Args:
            folder_name: DICOMフォルダ名
        
        Returns:
            tuple: (pred_array, spacing) または None（見つからない場合）
        """
        filename = f"{folder_name}_pred.nii.gz"
        filepath = self.predictions_dir / filename
        
        if not filepath.exists():
            return None
        
        pred_sitk = sitk.ReadImage(str(filepath))
        pred_array = sitk.GetArrayFromImage(pred_sitk)
        spacing = pred_sitk.GetSpacing()
        
        return pred_array, spacing
    
    def has_saved_prediction(self, folder_name):
        """保存済みの予測があるか確認"""
        filename = f"{folder_name}_pred.nii.gz"
        filepath = self.predictions_dir / filename
        return filepath.exists()

    def is_prediction_valid(self, folder_name):
        """予測NIfTIが存在し、SimpleITKで読み込み可能かどうかを確認する。

        クラッシュ等で中途半端なファイルが残った場合を検出するために
        実際に読み込みを試みる。
        """
        filepath = self.predictions_dir / f"{folder_name}_pred.nii.gz"
        if not filepath.exists():
            return False
        try:
            sitk.ReadImage(str(filepath))
            return True
        except Exception:
            return False

    def get_csv_row(self, folder_name):
        """CSVから該当 folder_name の行を辞書で返す。存在しなければ None。"""
        if not self.csv_file.exists():
            return None
        try:
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("folder_name") == folder_name:
                        return dict(row)
        except Exception:
            pass
        return None
    
    def append_to_csv(self, folder_name,
                      volumes_all, volumes_dyn, volumes_fix, dyn_range,
                      max_areas=None):
        """
        測定値をCSVに追記（1行 = 1検査）

        列グループ（各グループとも MUSCLE_LABELS 順）:
            folder_name, timestamp,
            [vol_all]  r_ir, l_ir, ...          全スライス体積 (cm³)
            [vol_dyn]  r_ir_dyn, l_ir_dyn, ...  動的範囲体積 (cm³)
            [vol_fix]  r_ir_fix, l_ir_fix, ...  固定範囲(5-11)体積 (cm³)
            vol_dyn_range                        動的範囲の実際のスライス番号 例 "4-10"
            [area]     r_ir_area, l_ir_area, ... 最大断面積 (cm²)

        Raises:
            PermissionError: ファイルがExcel等で開かれている場合
        """
        if max_areas is None:
            max_areas = {}

        file_exists = self.csv_file.exists()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dyn_labels  = [f"{lb}_dyn"  for lb in self.MUSCLE_LABELS]
        fix_labels  = [f"{lb}_fix"  for lb in self.MUSCLE_LABELS]
        area_labels = [f"{lb}_area" for lb in self.MUSCLE_LABELS]

        def _vol_cells(volumes, labels_src):
            return [
                f"{volumes[lb]:.2f}" if lb in volumes else ""
                for lb in labels_src
            ]

        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                if not file_exists:
                    header = (
                        ["folder_name", "timestamp"]
                        + self.MUSCLE_LABELS   # vol_all
                        + dyn_labels           # vol_dyn
                        + fix_labels           # vol_fix
                        + ["vol_dyn_range"]
                        + area_labels          # max area
                        + ["review_status"]
                    )
                    writer.writerow(header)

                row = (
                    [folder_name, timestamp]
                    + _vol_cells(volumes_all, self.MUSCLE_LABELS)
                    + _vol_cells(volumes_dyn, self.MUSCLE_LABELS)
                    + _vol_cells(volumes_fix, self.MUSCLE_LABELS)
                    + [dyn_range]
                    + [f"{max_areas[lb]:.2f}" if lb in max_areas else "" for lb in self.MUSCLE_LABELS]
                    + ["pending"]
                )
                writer.writerow(row)

        except PermissionError:
            raise PermissionError(
                f"CSVファイルへの書き込みができません。\n\n"
                f"ファイル: {self.csv_file}\n\n"
                f"ExcelでCSVファイルを開いている場合は、\n"
                f"Excelを閉じてから再度保存してください。"
            )

        return self.csv_file

    def update_review_status(self, folder_name, status):
        """CSVの該当行の review_status 列を更新する。

        Args:
            folder_name: 対象の folder_name（CSVの1列目）
            status: "confirmed" | "needs_correction" | "pending"

        Raises:
            PermissionError: CSVがExcel等で開かれている場合
            FileNotFoundError: CSVが存在しない場合
        """
        if not self.csv_file.exists():
            return

        try:
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
                rows = list(csv.reader(f))
        except PermissionError:
            raise PermissionError(
                f"CSVファイルを読み込めません（Excel等で開いていませんか？）\n{self.csv_file}"
            )

        if not rows:
            return

        header = rows[0]
        # review_status 列のインデックスを取得（なければ末尾に追加）
        if "review_status" not in header:
            header.append("review_status")
            for row in rows[1:]:
                row.append("pending")

        col = header.index("review_status")

        updated = False
        for row in rows[1:]:
            if row and row[0] == folder_name:
                # 必要なら行を拡張してから更新
                while len(row) <= col:
                    row.append("")
                row[col] = status
                updated = True

        if not updated:
            return  # 該当行なし（異常系のため静かに終了）

        try:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)
        except PermissionError:
            raise PermissionError(
                f"CSVファイルへの書き込みができません（Excel等で開いていませんか？）\n{self.csv_file}"
            )

    def add_to_review_list(self, folder_name, reason="手動確認が必要"):
        """
        手動レビューリストにフォルダを追加
        
        Args:
            folder_name: DICOMフォルダ名
            reason: レビューが必要な理由
        """
        # 既存リストを読み込み
        review_list = []
        if self.review_file.exists():
            with open(self.review_file, 'r', encoding='utf-8') as f:
                review_list = json.load(f)
        
        # 新規エントリを追加
        entry = {
            "folder_name": folder_name,
            "added_at": datetime.now().isoformat(),
            "reason": reason,
            "status": "pending"
        }
        review_list.append(entry)
        
        # 保存
        with open(self.review_file, 'w', encoding='utf-8') as f:
            json.dump(review_list, f, ensure_ascii=False, indent=2)
        
        return self.review_file
    
    def get_pending_review_count(self):
        """レビュー待ちの件数を取得"""
        if not self.review_file.exists():
            return 0
        
        with open(self.review_file, 'r', encoding='utf-8') as f:
            review_list = json.load(f)
        
        return len([r for r in review_list if r.get("status") == "pending"])
