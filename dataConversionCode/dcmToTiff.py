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

# eT1W_SE_tra, T2 TIME sue1 rfa180, eT1W_SE_cor, eT1W_SE_sag, T2W_SPIR_cor
series = "eT1W_SE_cor"

date = "2025_11_03_2"
ex_dir = 'C:/imageProcessing/data_sort/' + date + "/"

sepl = False
distributer_num = 10 #何人に分けるか
f_num = 15 #一人何フォルダか
output= 'data_Tiff/'+series+"/1/" #出力フォルダ

p = re.compile(".+({})".format(series))
exp = re.compile(r"EX\d+")
sep = re.compile(r"SE\d")
cnt=0

with open(ex_dir + "/data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    target_lines = [line for line in lines if p.match(line) ]
    print(len(target_lines))

    # Tiffファイルを保存するための出力ディレクトリ
    output_dir = 'data_Tiff/{}/'.format(series)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(target_lines)):
        ex = re.search(exp,target_lines[i]).group()
        se = re.search(sep,target_lines[i]).group()
        # DICOMファイルが保存されているディレクトリ
        dicom_dir = ex_dir + ex_id + ex_date + ex+"/"+se+"/"

        # ディレクトリ内のすべてのファイルを取得し、DICOMファイルのみをリストに追加
        dicom_files = [f for f in os.listdir(dicom_dir) if f.startswith('IMG')]

        # DICOMファイルをループ処理してtiff化
        for i in range(len(dicom_files)):
            zero=""
            if i < 9:
                zero="0"
            file_name = "IMG" + str(i+1)
            file_path = os.path.join(dicom_dir, file_name)
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
            output_file = os.path.join(output_dir, f'{ex+se}IMG{zero}{i+1}.tiff')
            
            # 画像をTIFF形式で保存（16ビット）
            image.save(output_file, 'TIFF')
            cnt+=1
    print(f'Converted {cnt} DICOM files to TIFF format.')


    #分割
    if distributer_num*f_num>len(target_lines):
        print("数が合わないよ")
    elif sepl:
        cnt=0
        dir="Data/Tiffs/"+series+"/"
        l = []
        for i in range(len(target_lines)):
            l.append(i)
        l_shuffled = random.sample(l, len(l))

        for i in range(distributer_num):
            output_dir = output + '{}/'.format(i)
            os.makedirs(output_dir, exist_ok=True)
            for j in range(f_num):
                ex = glob.glob(dir+'EX{}SE*.tiff'.format(l_shuffled[i*f_num+j]))
                for k in range(len(ex)):
                    file=ex[k]
                    # print(file)
                    shutil.copy(file,output_dir)
                    cnt+=1
        print("Copied "+{cnt}+" files to "+output+" .")
    else:
        print("Skip Sepalation")