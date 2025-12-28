# gui/CheckEnvironment.py
#!/usr/bin/env python3
"""
眼筋セグメンテーションGUI - 環境チェックスクリプト
GUI起動前に必要な環境が整っているか確認します
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Pythonのバージョンをチェック"""
    print("=" * 60)
    print("1. Pythonバージョンチェック")
    print("=" * 60)
    
    version = sys.version_info
    print(f"現在のバージョン: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✓ OK: Python 3.7以上")
        return True
    else:
        print("✗ NG: Python 3.7以上が必要です")
        return False


def check_packages():
    """必要なパッケージをチェック"""
    print("\n" + "=" * 60)
    print("2. 必要なパッケージのチェック")
    print("=" * 60)
    
    required_packages = {
        'PyQt5': 'PyQt5',
        'pydicom': 'pydicom',
        'SimpleITK': 'SimpleITK',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'nnunetv2': 'nnunetv2'
    }
    
    all_ok = True
    
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)
            print(f"✓ {package_name:15s} - インストール済み")
        except ImportError:
            print(f"✗ {package_name:15s} - 未インストール")
            print(f"  インストール: pip install {pip_name} --break-system-packages")
            all_ok = False
    
    return all_ok


def check_environment_variables():
    """環境変数をチェック"""
    print("\n" + "=" * 60)
    print("3. nnU-Net環境変数のチェック")
    print("=" * 60)
    
    required_vars = {
        'nnUNet_raw': 'raw dataの保存場所',
        'nnUNet_preprocessed': '前処理済みデータの保存場所',
        'nnUNet_results': '学習結果の保存場所'
    }
    
    all_ok = True
    
    for var_name, description in required_vars.items():
        value = os.environ.get(var_name)
        if value:
            print(f"✓ {var_name:25s} = {value}")
            
            # ディレクトリが存在するかチェック
            if not Path(value).exists():
                print(f"  ⚠ 警告: ディレクトリが存在しません")
        else:
            print(f"✗ {var_name:25s} - 未設定")
            print(f"  説明: {description}")
            all_ok = False
    
    if not all_ok:
        print("\n環境変数の設定方法:")
        print("  Linux/Mac:")
        print('    export nnUNet_raw="/path/to/nnUNet_raw"')
        print('    export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"')
        print('    export nnUNet_results="/path/to/nnUNet_results"')
        print("\n  Windows:")
        print('    set nnUNet_raw=C:\\path\\to\\nnUNet_raw')
        print('    set nnUNet_preprocessed=C:\\path\\to\\nnUNet_preprocessed')
        print('    set nnUNet_results=C:\\path\\to\\nnUNet_results')
    
    return all_ok


def check_trained_models():
    """学習済みモデルの存在をチェック"""
    print("\n" + "=" * 60)
    print("4. 学習済みモデルのチェック")
    print("=" * 60)
    
    results_dir = os.environ.get('nnUNet_results')
    
    if not results_dir:
        print("✗ nnUNet_results環境変数が設定されていません")
        return False
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"✗ ディレクトリが存在しません: {results_path}")
        return False
    
    # Dataset102_EyeMuscleSegmentationを探す
    datasets = list(results_path.glob("Dataset*"))
    
    if not datasets:
        print("⚠ 学習済みモデルが見つかりません")
        print("  学習を実行してください:")
        print("    python train_nnunet.py --mode full --task_id 102")
        return False
    
    print(f"見つかったデータセット:")
    for dataset in datasets:
        print(f"  - {dataset.name}")
        
        # チェックポイントを確認
        checkpoint_dirs = list(dataset.glob("**/fold_*"))
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pth"))
            if checkpoint_files:
                print(f"    ✓ チェックポイント: {len(checkpoint_files)}個")
    
    return True


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("眼筋セグメンテーションGUI - 環境チェック")
    print("=" * 60 + "\n")
    
    checks = [
        ("Pythonバージョン", check_python_version()),
        ("パッケージ", check_packages()),
        ("環境変数", check_environment_variables()),
        ("学習済みモデル", check_trained_models())
    ]
    
    print("\n" + "=" * 60)
    print("チェック結果のサマリー")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ OK" if result else "✗ NG"
        print(f"{status:6s} - {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✓ すべてのチェックをパスしました！")
        print("\nGUIを起動できます:")
        print("  python eye_muscle_gui_v1.py")
    else:
        print("✗ いくつかの問題があります。上記のメッセージを確認してください。")
        print("\n学習済みモデルがない場合は、先に学習を実行してください:")
        print("  python train_nnunet.py --mode full --task_id 102")
    
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)