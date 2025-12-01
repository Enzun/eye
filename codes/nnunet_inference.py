# codes/nnunet_inference.py
"""
nnU-Net推論用ユーティリティ
GUIツールでYOLOの代わりにnnU-Netモデルを使用するための関数群
"""
import numpy as np
import pydicom
import SimpleITK as sitk
import cv2
from pathlib import Path
from typing import Tuple, Dict, List
import tempfile
import os


class NnUNetPredictor:
    """nnU-Net推論ラッパークラス"""
    
    def __init__(self, task_id: int = 501, configuration: str = "2d", fold: int = 0):
        """
        Args:
            task_id: nnU-NetのタスクID
            configuration: ネットワーク構成 (2d/3d_fullres/3d_lowres)
            fold: 使用するfold
        """
        self.task_id = task_id
        self.configuration = configuration
        self.fold = fold
        self.dataset_name = f"Dataset{task_id:03d}_EyeMuscleSegmentation"
        
        # ラベル定義
        self.label_names = {
            1: "so",  # 上斜筋
            2: "io",  # 下斜筋
            3: "sr",  # 上直筋
            4: "ir",  # 下直筋
            5: "lr",  # 外直筋
            6: "mr",  # 内直筋
        }
        
        # 色定義（GUIと同じ）
        self.label_colors = {
            "l_so": (255, 0, 0),    # 赤
            "r_so": (255, 0, 0),    
            "l_io": (0, 255, 0),    # 緑
            "r_io": (0, 255, 0),    
            "l_sr": (0, 0, 255),    # 青
            "r_sr": (0, 0, 255),    
            "l_ir": (255, 255, 0),  # 黄
            "r_ir": (255, 255, 0),  
            "l_lr": (255, 0, 255),  # マゼンタ
            "r_lr": (255, 0, 255),  
            "l_mr": (0, 255, 255),  # シアン
            "r_mr": (0, 255, 255),  
            "so": (255, 0, 0),
            "io": (0, 255, 0),
            "sr": (0, 0, 255),   
            "ir": (255, 255, 0), 
            "lr": (255, 0, 255), 
            "mr": (0, 255, 255)
        }

    def dicom_to_nifti(self, dicom_path: str, output_path: str) -> None:
        """DICOMをNIfTI形式に変換して保存"""
        ds = pydicom.dcmread(dicom_path)
        image_array = ds.pixel_array.astype(np.float32)
        
        # 正規化
        if 'WindowCenter' in ds and 'WindowWidth' in ds:
            window_center = ds.WindowCenter if isinstance(ds.WindowCenter, (int, float)) else ds.WindowCenter[0]
            window_width = ds.WindowWidth if isinstance(ds.WindowWidth, (int, float)) else ds.WindowWidth[0]
        else:
            window_center = np.mean(image_array)
            window_width = np.max(image_array) - np.min(image_array)
        
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2
        image_array = np.clip(image_array, min_val, max_val)
        image_array = (image_array - min_val) / (max_val - min_val)
        
        # 3D形式に変換
        image_array = np.expand_dims(image_array, axis=0)  # (1, H, W)
        sitk_image = sitk.GetImageFromArray(image_array)
        
        # スペーシング情報を設定
        if hasattr(ds, 'PixelSpacing'):
            spacing = [1.0, float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
            sitk_image.SetSpacing(spacing)
        
        # NIfTI形式で保存
        sitk.WriteImage(sitk_image, output_path)

    def run_nnunet_inference(self, input_dir: str, output_dir: str) -> bool:
        """nnU-Netコマンドで推論を実行"""
        import subprocess
        
        cmd = (
            f"nnUNetv2_predict "
            f"-i {input_dir} "
            f"-o {output_dir} "
            f"-d {self.dataset_name} "
            f"-c {self.configuration} "
            f"-f {self.fold}"
        )
        
        try:
            result = subprocess.run(cmd, shell=True, check=True, 
                                  capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def predict_from_dicom(self, dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """
        DICOMファイルからnnU-Netで予測を実行
        
        Returns:
            visualized_image: 可視化された画像
            label_areas: 各ラベルの面積情報
        """
        # 一時ディレクトリの作成
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # DICOMをNIfTIに変換
            nifti_path = input_dir / "case_0000_0000.nii.gz"
            self.dicom_to_nifti(dicom_path, str(nifti_path))
            
            # nnU-Net推論実行
            success = self.run_nnunet_inference(str(input_dir), str(output_dir))
            
            if not success:
                raise RuntimeError("nnU-Net推論に失敗しました")
            
            # 予測結果を読み込み
            pred_path = output_dir / "case_0000.nii.gz"
            if not pred_path.exists():
                raise FileNotFoundError(f"予測結果が見つかりません: {pred_path}")
            
            pred_sitk = sitk.ReadImage(str(pred_path))
            pred_array = sitk.GetArrayToNumpy(pred_sitk)[0]  # (H, W)
            
            # 元のDICOM画像を読み込み
            ds = pydicom.dcmread(dicom_path)
            img_width = ds.Columns
            img_height = ds.Rows
            pixel_spacing = ds.PixelSpacing
            ds_area = pixel_spacing[1] * pixel_spacing[0]  # mm^2
            
            # 可視化用の画像を準備
            pixel_array = ds.pixel_array
            if pixel_array.dtype != np.uint8:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                             (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            if len(pixel_array.shape) == 2:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            
            img = pixel_array.copy()
            
            # 各ラベルの処理
            label_areas = {}
            
            for label_id, label_name in self.label_names.items():
                mask = (pred_array == label_id).astype(np.uint8)
                
                if mask.sum() == 0:
                    continue
                
                # 輪郭検出
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # 重心計算
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                    else:
                        center_x = img_width // 2
                    
                    # 左右判定
                    side_label = f"l_{label_name}" if center_x < img_width // 2 else f"r_{label_name}"
                    
                    # 色取得
                    color = self.label_colors.get(side_label, (128, 128, 128))
                    
                    # 面積計算
                    area = cv2.contourArea(contour) * ds_area
                    if side_label not in label_areas:
                        label_areas[side_label] = 0
                    label_areas[side_label] += area
                    
                    # 描画
                    cv2.polylines(img, [contour], isClosed=True, color=color, thickness=2)
                    x, y = contour.min(axis=0)[0]
                    cv2.putText(img, side_label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return img, label_areas


def predict_and_visualize_from_dicom_nnunet(dicom_file_path: str, predictor: NnUNetPredictor):
    """
    GUI用のnnU-Net予測関数
    YOLOの代わりにこの関数を使用
    """
    return predictor.predict_from_dicom(dicom_file_path)


# ========================================
# GUIでの使用例
# ========================================
"""
# gui.pyの修正例：

# YOLOモデルの代わりにnnU-Net predictor を初期化
from nnunet_inference import NnUNetPredictor, predict_and_visualize_from_dicom_nnunet

class DICOMProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... 他の初期化 ...
        
        # YOLOモデルの代わりにnnU-Net predictorを使用
        self.use_nnunet = True  # フラグで切り替え可能にする
        
        if self.use_nnunet:
            self.predictor = NnUNetPredictor(task_id=501)
        else:
            self.model = self.load_yolo_model(model_path)
        
        # ... 他の初期化 ...
    
    def process_and_display_image(self, file_path):
        self.current_image = file_path
        
        # nnU-NetまたはYOLOで予測
        if self.use_nnunet:
            processed_image, label_areas = predict_and_visualize_from_dicom_nnunet(
                self.current_image, self.predictor)
            results = None  # nnU-Netの場合resultsは不要
        else:
            processed_image, results, label_areas = predict_and_visualize_from_dicom(
                self.current_image, self.model)
        
        # ... 以下同じ処理 ...
"""
