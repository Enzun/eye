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
    
    def __init__(self, base_dir):
        """
        Args:
            base_dir: HospitalMuscleApp ディレクトリのパス
        """
        self.base_dir = Path(base_dir)
        self.predictions_dir = self.base_dir / "output" / "predictions"
        self.csv_file = self.base_dir / "output" / "csv" / "muscle_measurements.csv"
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
    
    def append_to_csv(self, folder_name, volumes, max_areas=None):
        """
        測定値をCSVに追記（1行 = 1検査）
        
        Args:
            folder_name: DICOMフォルダ名 (例: EX100SE3)
            volumes: {label: volume} の辞書 (例: {"l_sr": 123.45, "r_sr": 130.20})
            max_areas: {label: max_area} の辞書（各筋肉の最大断面積 cm²）
        
        CSV形式:
            folder_name, timestamp, 
            r_ir, l_ir, ... (体積 cm³),
            r_ir_area, l_ir_area, ... (最大面積 cm²)
        
        Raises:
            PermissionError: ファイルがExcel等で開かれている場合
        """
        if max_areas is None:
            max_areas = {}
        
        file_exists = self.csv_file.exists()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 最大面積のヘッダー用ラベル
        area_labels = [f"{label}_area" for label in self.MUSCLE_LABELS]
        
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 新規ファイルの場合はヘッダーを書き込み
                if not file_exists:
                    header = ["folder_name", "timestamp"] + self.MUSCLE_LABELS + area_labels
                    writer.writerow(header)
                
                # 各筋肉の体積を順番に取得（ないものは空文字）
                row = [folder_name, timestamp]
                for label in self.MUSCLE_LABELS:
                    if label in volumes:
                        row.append(f"{volumes[label]:.2f}")
                    else:
                        row.append("")
                
                # 各筋肉の最大面積を順番に取得（ないものは空文字）
                for label in self.MUSCLE_LABELS:
                    if label in max_areas:
                        row.append(f"{max_areas[label]:.2f}")
                    else:
                        row.append("")
                
                writer.writerow(row)
        except PermissionError:
            raise PermissionError(
                f"CSVファイルへの書き込みができません。\n\n"
                f"ファイル: {self.csv_file}\n\n"
                f"ExcelでCSVファイルを開いている場合は、\n"
                f"Excelを閉じてから再度保存してください。"
            )
        
        return self.csv_file
    
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
