# dataConversionCode/tiff_to_nifti_inference.py
"""
推論用: TIFF画像をNIfTI形式に変換するスクリプト
指定されたフォルダ内のTIFF画像を患者ごとにまとめて 3D NIfTI (__0000.nii.gz) に変換します。
"""

import os
import glob
import re
import argparse
import numpy as np
from PIL import Image
import SimpleITK as sitk
from collections import defaultdict

def extract_patient_info(filename):
    """
    ファイル名から患者情報を抽出
    対応フォーマット:
    1. {Hash}{EX}{SE}IMG{Num}
    2. {EX}{SE}IMG{Num} (Legacy)
    """
    # 1. Hash形式
    match_hash = re.search(r'([a-f0-9]{12})(EX\d+)(SE\d+)IMG(\d+)', filename)
    if match_hash:
        return match_hash.group(1), match_hash.group(2), match_hash.group(3), int(match_hash.group(4))
    
    # 2. Legacy形式
    match_legacy = re.search(r'(EX\d+)(SE\d+)IMG(\d+)', filename)
    if match_legacy:
        # Hashがない場合は "Legacy" をプレフィックスとするか、EX番号をそのまま使う
        return "Legacy", match_legacy.group(1), match_legacy.group(2), int(match_legacy.group(3))
        
    return None, None, None, None

def group_files_by_patient(tiff_dir):
    """TIFFファイルを患者ごとにグループ化"""
    patient_groups = defaultdict(list)
    
    # 再帰的に検索
    tiff_files = glob.glob(os.path.join(tiff_dir, "**", "*.tiff"), recursive=True)
    if not tiff_files:
        tiff_files = glob.glob(os.path.join(tiff_dir, "**", "*.tif"), recursive=True)
    
    print(f"発見されたTIFFファイル数: {len(tiff_files)}")

    for tiff_path in tiff_files:
        filename = os.path.basename(tiff_path)
        hash_id, ex_num, se_num, img_num = extract_patient_info(filename)
        
        if ex_num and se_num:
            # キー: (Hash, EX, SE)
            patient_key = (hash_id, ex_num, se_num)
            patient_groups[patient_key].append({
                'path': tiff_path,
                'img_num': img_num,
                'filename': filename
            })
            
    # ソート
    for key in patient_groups:
        patient_groups[key].sort(key=lambda x: x['img_num'])
        
    return patient_groups

def convert_to_nifti(patient_groups, output_dir):
    """グループ化されたデータをNIfTIに変換"""
    os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    
    for (hash_id, ex_num, se_num), files in patient_groups.items():
        if hash_id == "Legacy":
            case_name = f"{ex_num}_{se_num}"
        else:
            case_name = f"{hash_id}_{ex_num}_{se_num}"
            
        output_filename = f"{case_name}_0000.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"変換中: {case_name} ({len(files)} slices) -> {output_filename}")
        
        image_slices = []
        try:
            for file_info in files:
                img = Image.open(file_info['path'])
                img_array = np.array(img)
                
                # グレースケール変換
                if img_array.ndim == 3:
                    img_array = np.mean(img_array, axis=2).astype(np.uint8)
                
                image_slices.append(img_array)
            
            # 3D化
            image_3d = np.stack(image_slices, axis=0)
            
            # SimpleITK変換
            sitk_image = sitk.GetImageFromArray(image_3d)
            sitk_image.SetSpacing((1.0, 1.0, 1.0)) # 仮のスペーシング
            
            sitk.WriteImage(sitk_image, output_path)
            converted_count += 1
            
        except Exception as e:
            print(f"  ❌ エラー: {case_name} - {e}")
            
    return converted_count

def main():
    parser = argparse.ArgumentParser(description="TIFF to NIfTI for Inference")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing TIFF files")
    parser.add_argument("--output", "-o", required=True, help="Output directory to save NIfTI files")
    
    args = parser.parse_args()
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    groups = group_files_by_patient(args.input)
    print(f"検出された患者(シリーズ)数: {len(groups)}")
    
    if not groups:
        print("処理対象が見つかりませんでした。")
        return

    count = convert_to_nifti(groups, args.output)
    print(f"完了: {count} 件のNIfTIファイルを作成しました。")

if __name__ == "__main__":
    main()
