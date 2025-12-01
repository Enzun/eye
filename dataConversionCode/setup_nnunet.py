# dataConversionCode/setup_nnunet.py
"""
nnU-Net環境の初期セットアップスクリプト
必要なディレクトリ構造を自動作成します
"""

import os
import sys

def create_directory_structure(base_path):
    """nnU-Netに必要なディレクトリ構造を作成"""
    
    directories = [
        os.path.join(base_path, "nnUNet_raw"),
        os.path.join(base_path, "nnUNet_preprocessed"),
        os.path.join(base_path, "nnUNet_results"),
    ]
    
    print("=" * 60)
    print("nnU-Net ディレクトリ構造を作成中...")
    print("=" * 60)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 作成: {directory}")
    
    return directories

def create_env_script(base_path, directories):
    """環境変数設定スクリプトを作成"""
    
    # Bashスクリプト（Linux/Mac用）
    bash_script = f"""#!/bin/bash
# nnU-Net 環境変数設定スクリプト

export nnUNet_raw="{directories[0]}"
export nnUNet_preprocessed="{directories[1]}"
export nnUNet_results="{directories[2]}"

echo "nnU-Net環境変数を設定しました:"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo ""
echo "使用方法: source set_nnunet_env.sh"
"""
    
    bash_path = os.path.join(base_path, "set_nnunet_env.sh")
    with open(bash_path, 'w') as f:
        f.write(bash_script)
    
    # Windowsバッチスクリプト
    batch_script = f"""@echo off
REM nnU-Net 環境変数設定スクリプト

set nnUNet_raw={directories[0]}
set nnUNet_preprocessed={directories[1]}
set nnUNet_results={directories[2]}

echo nnU-Net環境変数を設定しました:
echo   nnUNet_raw: %nnUNet_raw%
echo   nnUNet_preprocessed: %nnUNet_preprocessed%
echo   nnUNet_results: %nnUNet_results%
echo.
echo 使用方法: set_nnunet_env.bat
"""
    
    batch_path = os.path.join(base_path, "set_nnunet_env.bat")
    with open(batch_path, 'w') as f:
        f.write(batch_script)
    
    print(f"\n環境変数設定スクリプトを作成しました:")
    print(f"  Linux/Mac: {bash_path}")
    print(f"  Windows:   {batch_path}")
    
    return bash_path, batch_path

def create_readme(base_path):
    """README.txtを作成"""
    
    readme_content = """nnU-Net セットアップディレクトリ

このディレクトリにはnnU-Netの学習に必要なすべてのデータが格納されます。

■ ディレクトリ構造:
  nnUNet_raw/          : 元データ（NIfTI形式）を配置
  nnUNet_preprocessed/ : 前処理済みデータ（自動生成）
  nnUNet_results/      : 学習結果・モデル（自動生成）

■ 環境変数の設定:
  
  【Windowsの場合】
  set_nnunet_env.bat を実行してください
  
  【Linux/Macの場合】
  source set_nnunet_env.sh を実行してください

■ 次のステップ:
  1. cvat_to_nnunet.pyでデータを変換
  2. 前処理: nnUNetv2_plan_and_preprocess -d [タスクID]
  3. 学習: nnUNetv2_train [タスクID] 2d 0

詳細はCVAT_to_nnUNet_Guide.mdを参照してください。
"""
    
    readme_path = os.path.join(base_path, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\nREADMEを作成しました: {readme_path}")

def main():
    print("\n" + "=" * 60)
    print("nnU-Net 環境セットアップ")
    print("=" * 60)
    
    # ベースパスの入力
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        print("\nnnU-Netのデータを保存するディレクトリを指定してください。")
        print("例: C:/workspace/nnUNet_data または /home/user/nnUNet_data")
        base_path = input("\nパス: ").strip()
    
    if not base_path:
        print("エラー: パスが指定されていません。")
        return
    
    # パスを正規化
    base_path = os.path.abspath(base_path)
    
    print(f"\n以下のパスにnnU-Net環境を構築します:")
    print(f"  {base_path}")
    
    response = input("\n続行しますか? (y/n): ")
    if response.lower() != 'y':
        print("セットアップをキャンセルしました。")
        return
    
    # ディレクトリ構造を作成
    directories = create_directory_structure(base_path)
    
    # 環境変数設定スクリプトを作成
    create_env_script(base_path, directories)
    
    # READMEを作成
    create_readme(base_path)
    
    print("\n" + "=" * 60)
    print("セットアップ完了!")
    print("=" * 60)
    print("\n次のステップ:")
    
    if sys.platform == "win32":
        print(f"1. {os.path.join(base_path, 'set_nnunet_env.bat')} を実行")
    else:
        print(f"1. source {os.path.join(base_path, 'set_nnunet_env.sh')} を実行")
    
    print("2. cvat_to_nnunet.pyを実行してデータを変換")
    print("3. nnU-Netで学習を開始")
    print(f"\n詳細は {os.path.join(base_path, 'README.txt')} を参照してください。")

if __name__ == "__main__":
    main()
