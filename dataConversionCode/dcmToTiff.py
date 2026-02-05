# dataConversionCode/dcmToTiff.py
# 指定したSeriesのdicomファイルをEXFILESフォルダからTiffファイルにしてTiffsフォルダに保存。
# 同時に分割したければもろもろのステータス指定
import os
import pydicom
from PIL import Image
import numpy as np
import re
import random
import shutil
import glob
import hashlib

def get_hash(id_str, date_str, time_str):
    """ID, Date, Timeからハッシュを生成"""
    raw = f"{id_str}{date_str}{time_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

# eT1W_SE_tra, T2 TIME sue1 rfa180, eT1W_SE_cor, eT1W_SE_sag, T2W_SPIR_cor
series = "eT1W_SE_cor"

date = "2025_11_03_2"
# プロジェクトルートを取得 (dataConversionCodeの親ディレクトリ)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ex_dir = os.path.join(PROJECT_ROOT, 'data_sort', date, "")

sepl = False
distributer_num = 10 #何人に分けるか
f_num = 15 #一人何フォルダか
# output= 'C:/imageProcessing/data_Tiff/'+series+"/"+date+"/" #出力フォルダ
output = os.path.join(PROJECT_ROOT, 'data_Tiff', series, "")

p = re.compile(".+({})".format(series))
exp = re.compile(r"EX\d+")
sep = re.compile(r"SE\d")
cnt=0

with open(os.path.join(ex_dir, "data.txt"), "r", encoding="utf-8") as f:
    lines = f.readlines()
    target_lines = [line for line in lines if p.match(line) ]
    print(len(target_lines))

    # Tiffファイルを保存するための出力ディレクトリ
    os.makedirs(output, exist_ok=True)

    for i in range(len(target_lines)):
    # パス部分を抽出
        path_match = re.search(r"DATA\\(.+)", target_lines[i])
        if path_match:
            path = path_match.group(1)  # 100241\20240508\112048\ex140\SE1
            parts = path.split("\\")
            
            id_num = parts[0]    # 100241
            date = parts[1]      # 20240508
            time = parts[2]      # 112048
            ex = parts[3].upper()  # EX140
            se = parts[4].upper()  # SE1
            
            # ハッシュ生成
            hash_str = get_hash(id_num, date, time)
            
            # フォルダ名: hash + EX
            folder_name = hash_str + ex
            
            # ファイル名プリフィックス: hash + EX + SE
            file_prefix = hash_str + ex + se
            # print(folder_name)

        # SEフォルダ内の画像を取得するためにパスを構築
        se_dir_path = os.path.join(ex_dir, "DATA",folder_name, se)
        
        if not os.path.exists(se_dir_path):
            print(f"Directory not found: {se_dir_path}")
            continue

        # ディレクトリ内のすべてのファイルを取得し、DICOMファイルのみをリストに追加
        dicom_files = [f for f in os.listdir(se_dir_path) if f.startswith('IMG')]

        # DICOMファイルをループ処理してtiff化
        for i in range(len(dicom_files)):
            zero=""
            if i < 9:
                zero="0"
            file_name = "IMG" + str(i+1)
            file_path = os.path.join(se_dir_path, file_name)
            ds = pydicom.dcmread(file_path)
            
            # 画像データを取得
            image_array = ds.pixel_array
            
            # ウィンドウ幅とウィンドウ中心を取得
            if 'WindowCenter' in ds and 'WindowWidth' in ds:
                window_center = ds.WindowCenter if isinstance(ds.WindowCenter, float) else ds.WindowCenter[0]
                window_width = ds.WindowWidth if isinstance(ds.WindowWidth, float) else ds.WindowWidth[0]
            else:
                window_center = np.mean(image_array)
                window_width = np.max(image_array) - np.min(image_array)
            
            # 画像データにウィンドウ幅とウィンドウ中心を適用
            min_val = window_center - window_width // 2
            max_val = window_center + window_width // 2
            image_8bit = np.clip(image_array, min_val, max_val)
            image_8bit = ((image_8bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            
            # 画像データをPIL Imageオブジェクトに変換
            image = Image.fromarray(image_8bit)
            
            # 出力ファイル名を生成
            output_file = os.path.join(output, f'{file_prefix}IMG{zero}{i+1}.tiff')
            
            # 画像をTIFF形式で保存（16ビット）
            image.save(output_file, 'TIFF')
            cnt+=1
    print(f'Converted {cnt} DICOM files to TIFF format.')


    #分割
    if distributer_num*f_num>len(target_lines):
        print("数が合わないよ")
    elif sepl:
        cnt=0
        l = []
        for i in range(len(target_lines)):
            l.append(i)
        l_shuffled = random.sample(l, len(l))

        for i in range(distributer_num):
            output_dir = output + '{}/'.format(i)
            os.makedirs(output_dir, exist_ok=True)
            for j in range(f_num):
                ex = glob.glob(output+'EX{}SE*.tiff'.format(l_shuffled[i*f_num+j]))
                for k in range(len(ex)):
                    file=ex[k]
                    # print(file)
                    shutil.copy(file,output_dir)
                    cnt+=1
        print("Copied "+{cnt}+" files to "+output+" .")
    else:
        print("Skip Sepalation")