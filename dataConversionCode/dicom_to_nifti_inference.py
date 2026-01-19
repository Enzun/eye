# dataConversionCode/dicom_to_nifti_inference.py
"""
推論用: DICOM画像を直接NIfTI形式に変換するスクリプト
学習データ作成時と同じWindowing処理とスペーシング(1,1,1)を適用し、
TIFFを経由せずにnnU-Net用の入力ファイルを作成します。
"""

import os
import glob
import re
import argparse
import numpy as np
import pydicom
import SimpleITK as sitk

def load_and_preprocess_dicom(dicom_path):
    """DICOMを読み込み、学習時と同じ前処理（Windowing + 8bit化）を適用"""
    ds = pydicom.dcmread(dicom_path)
    image_array = ds.pixel_array
    
    # Windowing処理 (dcmToTiff.pyのロジックを再現)
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        # 配列の場合は最初の要素を取得
        wc = ds.WindowCenter if isinstance(ds.WindowCenter, float) or isinstance(ds.WindowCenter, int) else ds.WindowCenter[0]
        ww = ds.WindowWidth if isinstance(ds.WindowWidth, float) or isinstance(ds.WindowWidth, int) else ds.WindowWidth[0]
    else:
        wc = np.mean(image_array)
        ww = np.max(image_array) - np.min(image_array)
    
    min_val = wc - ww // 2
    max_val = wc + ww // 2
    
    # クリッピングと正規化 (0-255)
    image_8bit = np.clip(image_array, min_val, max_val)
    image_8bit = ((image_8bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return image_8bit

def convert_dicom_series_to_nifti(dicom_dir, output_path):
    """ディレクトリ内のDICOM列をまとめてNIfTIに変換"""
    
    # IMG番号順にソートして読み込む (dcmToTiff.pyと順序を合わせる)
    # ファイル名が IMG1, IMG2... となっていることを想定
    dicom_files = [f for f in os.listdir(dicom_dir) if f.startswith('IMG')]
    
    # 数字部分でソートする (IMG1, IMG2, ..., IMG10... となるように)
    def extract_number(filename):
        match = re.search(r'IMG(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    dicom_files.sort(key=extract_number)
    
    if not dicom_files:
        print(f"警告: {dicom_dir} にIMGから始まるDICOMファイルが見つかりません。")
        return False

    print(f"処理中: {dicom_dir} ({len(dicom_files)} slices)")
    
    image_slices = []
    try:
        for filename in dicom_files:
            file_path = os.path.join(dicom_dir, filename)
            img_array = load_and_preprocess_dicom(file_path)
            image_slices.append(img_array)
        
        # 3D化
        image_3d = np.stack(image_slices, axis=0)
        
        # SimpleITK変換
        sitk_image = sitk.GetImageFromArray(image_3d)
        
        # 学習時と同じスペーシング (1.0, 1.0, 1.0) を強制
        sitk_image.SetSpacing((1.0, 1.0, 1.0))
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(sitk_image, output_path)
        print(f"  ✓ 作成完了: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Direct DICOM to NIfTI for Inference")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing DICOM files (folder with IMG files)")
    parser.add_argument("--output", "-o", required=True, help="Output directory to save NIfTI file (e.g., inference_input)")
    parser.add_argument("--case_name", "-n", default="case", help="Case name for output file (default: case -> case_0000.nii.gz)")
    
    args = parser.parse_args()
    
    output_filename = f"{args.case_name}_0000.nii.gz"
    output_path = os.path.join(args.output, output_filename)
    
    success = convert_dicom_series_to_nifti(args.input, output_path)
    
    if success:
        print("\n変換完了！以下のコマンドで推論できます:")
        print(f"python codes/train_nnunet.py --mode predict --task_id 119 --input_folder {args.output}")

if __name__ == "__main__":
    main()
