# dataConversionCode/cvat_to_nnunet.py
"""
CVATアノテーションをnnU-Net形式に変換するスクリプト
患者ごとにTIFFファイルをまとめて3D NIfTIに変換します
"""

import os
import json
import glob
import shutil
import re
from collections import defaultdict
import numpy as np
from PIL import Image
import SimpleITK as sitk
import cv2

# ==================== 設定項目 ====================

# タスクID (001から999の間)
TASK_ID = 102
TASK_NAME = "EyeMuscleSegmentation"
TASK_FOLDER_NAME = f"Dataset{TASK_ID:03d}_{TASK_NAME}"

# 入力データのパス
CVAT_JSON_DIR = "Data/cvat_output"  # CVATから出力したJSONファイルがある場所
TIFF_IMAGE_DIR = "Data/Tiffs"  # 対応するTIFF画像がある場所

# nnU-Net用の出力先
NNUNET_RAW_DATA_DIR = os.environ.get("nnUNet_raw")
if NNUNET_RAW_DATA_DIR is None:
    print("警告: 環境変数 'nnUNet_raw' が設定されていません。")
    print("デフォルトのパスを使用します: ./nnUNet_raw")
    NNUNET_RAW_DATA_DIR = "./nnUNet_raw"

NNUNET_RAW_DATA_DIR = os.path.normpath(NNUNET_RAW_DATA_DIR)

# ラベルマッピング (CVATのラベル名 → nnU-Netのラベル番号)
# 0は背景として予約されています
LABEL_MAP = {
    "ir": 1,
    "mr": 2,
    "sr": 3,
    "so": 4,
    "lr": 5,
    "io": 6,
}

# Seriesの種類（必要に応じて変更）
SERIES_NAME = "eT1W_SE_tra"

# ==================== 関数定義 ====================

def extract_patient_info(filename):
    """
    ファイル名から患者情報を抽出
    例: EX7SE3IMG05.tiff → (EX7, SE3, 5)
    """
    match = re.match(r'(EX\d+)(SE\d+)IMG(\d+)', filename)
    if match:
        ex_num = match.group(1)
        se_num = match.group(2)
        img_num = int(match.group(3))
        return ex_num, se_num, img_num
    return None, None, None

def group_files_by_patient(json_dir):
    """
    JSONファイルを患者ごとにグループ化
    返り値: {(EX番号, SE番号): [ファイルパスのリスト]}
    """
    patient_groups = defaultdict(list)
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    for json_path in json_files:
        filename = os.path.basename(json_path)
        ex_num, se_num, img_num = extract_patient_info(filename)
        
        if ex_num and se_num:
            patient_key = (ex_num, se_num)
            patient_groups[patient_key].append({
                'json_path': json_path,
                'img_num': img_num,
                'filename': filename
            })
    
    # 各患者のファイルをIMG番号でソート
    for patient_key in patient_groups:
        patient_groups[patient_key].sort(key=lambda x: x['img_num'])
    
    return patient_groups

def json_to_mask(json_path, tiff_path, label_map):
    """
    LabelMe JSONファイルとTIFF画像からマスク画像を生成
    """
    # JSONファイルを読み込み
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 画像サイズを取得
    height = data['imageHeight']
    width = data['imageWidth']
    
    # マスク画像を初期化（背景=0）
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 各シェイプをマスクに描画
    for shape in data['shapes']:
        label_name = shape['label']
        
        # l_やr_プレフィックスを削除
        if label_name.startswith('l_') or label_name.startswith('r_'):
            label_name = label_name[2:]
        
        if label_name not in label_map:
            continue
        
        label_value = label_map[label_name]
        points = np.array(shape['points'], dtype=np.int32)
        
        # ポリゴンを塗りつぶし
        cv2.fillPoly(mask, [points], label_value)
    
    return mask

def create_3d_nifti_from_patient_data(patient_files, tiff_dir, label_map):
    """
    患者の複数スライスから3D NIfTI画像とマスクを作成
    """
    image_slices = []
    mask_slices = []
    
    for file_info in patient_files:
        json_path = file_info['json_path']
        base_name = file_info['filename'].replace('.json', '')
        
        # 対応するTIFF画像を検索
        tiff_path = os.path.join(tiff_dir, base_name + '.tiff')
        if not os.path.exists(tiff_path):
            tiff_path = os.path.join(tiff_dir, base_name + '.tif')
        
        if not os.path.exists(tiff_path):
            # サブディレクトリも検索
            tiff_pattern = os.path.join(tiff_dir, "**", base_name + ".tiff")
            found = glob.glob(tiff_pattern, recursive=True)
            if found:
                tiff_path = found[0]
            else:
                print(f"  警告: TIFF画像が見つかりません: {base_name}")
                continue
        
        # TIFF画像を読み込み
        img = Image.open(tiff_path)
        img_array = np.array(img)
        
        # グレースケールに変換
        if img_array.ndim == 3:
            img_array = np.mean(img_array, axis=2).astype(np.uint8)
        
        # マスクを生成
        mask_array = json_to_mask(json_path, tiff_path, label_map)
        
        image_slices.append(img_array)
        mask_slices.append(mask_array)
    
    if not image_slices:
        return None, None
    
    # 3D配列にスタック
    image_3d = np.stack(image_slices, axis=0)  # (Z, H, W)
    mask_3d = np.stack(mask_slices, axis=0)    # (Z, H, W)
    
    # SimpleITKイメージに変換
    sitk_image = sitk.GetImageFromArray(image_3d)
    sitk_mask = sitk.GetImageFromArray(mask_3d)
    
    # スペーシング情報を設定（必要に応じて調整）
    # (X, Y, Z) の順で設定
    spacing = (1.0, 1.0, 1.0)  # mm単位
    sitk_image.SetSpacing(spacing)
    sitk_mask.SetSpacing(spacing)
    
    return sitk_image, sitk_mask

def create_nnunet_dataset(patient_groups, tiff_dir, output_dir, label_map):
    """
    nnU-Net用のデータセットを作成
    """
    # 出力ディレクトリを作成
    imagesTr_dir = os.path.join(output_dir, "imagesTr")
    labelsTr_dir = os.path.join(output_dir, "labelsTr")
    
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)
    
    print(f"\nnnU-Netデータセットを作成中: {output_dir}")
    
    case_list = []
    
    # 患者ごとに処理
    for idx, (patient_key, patient_files) in enumerate(patient_groups.items()):
        ex_num, se_num = patient_key
        case_name = f"{ex_num}_{se_num}"
        
        print(f"\n[{idx+1}/{len(patient_groups)}] 処理中: {case_name} ({len(patient_files)}スライス)")
        
        # 3D NIfTIを作成
        sitk_image, sitk_mask = create_3d_nifti_from_patient_data(
            patient_files, tiff_dir, label_map
        )
        
        if sitk_image is None:
            print(f"  スキップ: データが不足しています")
            continue
        
        # ファイル名を生成（nnU-Net形式）
        # 画像: {case_name}_0000.nii.gz （_0000はモダリティを示す）
        # ラベル: {case_name}.nii.gz
        image_filename = os.path.join(imagesTr_dir, f"{case_name}_0000.nii.gz")
        label_filename = os.path.join(labelsTr_dir, f"{case_name}.nii.gz")
        
        # NIfTI形式で保存
        sitk.WriteImage(sitk_image, image_filename)
        sitk.WriteImage(sitk_mask, label_filename)
        
        case_list.append(case_name)
        print(f"  保存完了: {case_name}")
    
    return case_list

def create_dataset_json(output_dir, label_map, num_training):
    """
    nnU-Net用のdataset.jsonを作成
    """
    # ラベル辞書を作成（文字列キーが必要）
    labels_dict = {str(val): key for key, val in label_map.items()}
    labels_dict["0"] = "background"
    
    dataset_json = {
        "channel_names": {
            "0": "MRI"  # _0000.nii.gz に対応
        },
        "labels": labels_dict,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        # 2D/3Dデータセットの設定
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\ndataset.jsonを作成しました: {json_path}")

# ==================== メイン処理 ====================

def main():
    print("=" * 60)
    print("CVAT → nnU-Net 変換スクリプト")
    print("=" * 60)
    
    # 出力ディレクトリのパス
    output_dir = os.path.join(NNUNET_RAW_DATA_DIR, TASK_FOLDER_NAME)
    
    # 既存のディレクトリをクリーンアップ
    if os.path.exists(output_dir):
        response = input(f"\n{output_dir} は既に存在します。削除して続行しますか? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_dir)
        else:
            print("処理を中止しました。")
            return
    
    # JSONファイルを患者ごとにグループ化
    print(f"\nJSONファイルをグループ化中...")
    patient_groups = group_files_by_patient(CVAT_JSON_DIR)
    
    if not patient_groups:
        print(f"エラー: {CVAT_JSON_DIR} にJSONファイルが見つかりません。")
        return
    
    print(f"患者数: {len(patient_groups)}")
    for patient_key, files in patient_groups.items():
        print(f"  {patient_key[0]}_{patient_key[1]}: {len(files)}スライス")
    
    # nnU-Netデータセットを作成
    case_list = create_nnunet_dataset(
        patient_groups, TIFF_IMAGE_DIR, output_dir, LABEL_MAP
    )
    
    # dataset.jsonを作成
    create_dataset_json(output_dir, LABEL_MAP, len(case_list))
    
    print("\n" + "=" * 60)
    print("変換完了!")
    print("=" * 60)
    print(f"変換されたケース数: {len(case_list)}")
    print(f"出力先: {output_dir}")
    print("\n次のステップ:")
    print("1. nnU-Netの前処理を実行:")
    print(f"   nnUNetv2_plan_and_preprocess -d {TASK_ID} --verify_dataset_integrity")
    print("2. 学習を開始:")
    print(f"   nnUNetv2_train {TASK_ID} 2d 0")

if __name__ == "__main__":
    main()
