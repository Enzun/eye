# dataConversionCode/rename_old_files.py
# 既存の「time+EX」形式のフォルダを「ID+DATE+EX」形式にリネーム
バックアップとれ
import os
import re
import hashlib

def get_hash(id_str, date_str, time_str):
    """ID, Date, Timeからハッシュを生成"""
    raw = f"{id_str}{date_str}{time_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

date = "2025_11_03_2"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# リネーム対象のディレクトリ
ex_dir = os.path.join(PROJECT_ROOT, 'data_sort', date, "DATA", "")

# data.txtのパス
data_txt = os.path.join(PROJECT_ROOT, 'data_sort', date, "data.txt")

# パスを抽出するパターン
p = re.compile(r"DATA\\(.+)")

renamed_count = 0
not_found_count = 0

print(f"Starting rename process for: {ex_dir}")
print("-" * 60)

with open(data_txt, "r", encoding="utf-8") as f:
    lines = f.readlines()
    
    for line in lines:
        path_match = re.search(p, line)
        if path_match:
            path = path_match.group(1)  # 100241\20240508\112048\ex140\SE1
            parts = path.split("\\")
            
            if len(parts) >= 4:
                id_num = parts[0]       # 100241
                date_str = parts[1]     # 20240508
                time = parts[2]         # 112048
                ex = parts[3].upper()   # EX140
                
                # 古い名前（time+EX）と新しい名前（Hash+EX）
                old_name = time + ex                    # 112048EX140
                hash_str = get_hash(id_num, date_str, time)
                new_name = hash_str + ex                # {hash}EX140
                
                old_path = os.path.join(ex_dir, old_name)
                new_path = os.path.join(ex_dir, new_name)
                
                # リネーム実行
                if os.path.exists(old_path):
                    if not os.path.exists(new_path):  # 新しい名前が既に存在しないか確認
                        os.rename(old_path, new_path)
                        print(f"✅ Renamed: {old_name} -> {new_name}")
                        renamed_count += 1
                    else:
                        print(f"⚠️  Skip (already exists): {new_name}")
                else:
                    # 既にリネーム済みかもしれない
                    if os.path.exists(new_path):
                        print(f"ℹ️  Already renamed: {new_name}")
                    else:
                        print(f"❌ Not found: {old_path}")
                        not_found_count += 1

print("-" * 60)
print(f"Rename completed!")
print(f"  Renamed: {renamed_count} folders")
print(f"  Not found: {not_found_count} folders")