# dataConversionCode/DATA2data.py
# ファイル構造シンプル化
import os
import shutil
import hashlib

def get_hash(id_str, date_str, time_str):
    """ID, Date, Timeからハッシュを生成"""
    raw = f"{id_str}{date_str}{time_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]  # 先頭12文字を使用

# プロジェクトルートを取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

date = "2025_11_03_2"
need_txt = True

output_dir = os.path.join(PROJECT_ROOT, 'data_sort', date, "")
os.makedirs(output_dir, exist_ok=True)

ex_output = os.path.join(output_dir, "DATA", "")
os.makedirs(ex_output, exist_ok=True)

dir1 = os.path.join(PROJECT_ROOT, 'DATA', date, 'DATA')

if os.path.exists(dir1):
    folders = [f for f in os.listdir(dir1)] #100559, 109021...

    for i in range(len(folders)): #len(folders)
        dir2  = os.path.join(dir1, folders[i]) #DATA/100559
        folders2 = [f for f in os.listdir(dir2)] 

        for j in range(len(folders2)):
            dir3  = os.path.join(dir2, folders2[j]) # DATA/100559/20241003
            folders3 = [f for f in os.listdir(dir3)]

            for k in range(len(folders3)):
                dir4  = os.path.join(dir3, folders3[k]) # DATA/100559/20241003/130714
                folders4 = [f for f in os.listdir(dir4)]

                for l in range(len(folders4)):
                    EXdir  = os.path.join(dir4, folders4[l]) # DATA/100559/20241003/130714/EX26
                    
                    id_num = folders[i]
                    date_str = folders2[j]
                    time_str = folders3[k]
                    ex = folders4[l]
                    
                    # ハッシュ生成
                    hash_str = get_hash(id_num, date_str, time_str)
                    
                    new_name = hash_str + ex # hash+EX
                    shutil.move(EXdir, os.path.join(ex_output, new_name))

if need_txt:
    txtfile = os.path.join(PROJECT_ROOT, 'DATA', date, "Data.txt")
    if os.path.exists(txtfile):
        shutil.move(txtfile, output_dir)

    dicomdir = os.path.join(PROJECT_ROOT, 'DATA', date, "DICOMDIR")
    if os.path.exists(dicomdir):
        shutil.move(dicomdir, output_dir)
