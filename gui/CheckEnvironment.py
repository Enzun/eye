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
        print("[OK] Python 3.7以上")
        return True
    else:
        print("[NG] Python 3.7以上が必要です")
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
            print(f"[OK] {package_name:15s} - インストール済み")
        except ImportError:
            print(f"[NG] {package_name:15s} - 未インストール")
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
            print(f"[OK] {var_name:25s} = {value}")
            
            # ディレクトリが存在するかチェック
            if not Path(value).exists():
                print(f"  [WARN] 警告: ディレクトリが存在しません")
        else:
            print(f"[NG] {var_name:25s} - 未設定")
            print(f"  説明: {description}")
            all_ok = False
    

    if not all_ok:
        print("\n環境変数の設定方法:")
        print("  プロジェクトルートの .env ファイルを確認・編集してください。")
        print("  setup_nnunet.sh を実行することでも生成できます。")

    
    return all_ok


def check_trained_models():
    """学習済みモデルの存在をチェック"""
    print("\n" + "=" * 60)
    print("4. 学習済みモデルのチェック")
    print("=" * 60)
    
    results_dir = os.environ.get('nnUNet_results')
    
    if not results_dir:
        print("[NG] nnUNet_results環境変数が設定されていません")
        return False
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"[NG] ディレクトリが存在しません: {results_path}")
        return False
    
    # Dataset102_EyeMuscleSegmentationを探す
    datasets = list(results_path.glob("Dataset*"))
    
    if not datasets:
        print("[WARN] 学習済みモデルが見つかりません")
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
                print(f"    [OK] チェックポイント: {len(checkpoint_files)}個")
    
    return True



def load_env_file():
    """Project rootの.envファイルを読み込む"""
    try:
        # このスクリプトは gui/ フォルダにある想定
        # プロジェクトルートは一つ上の階層
        env_path = Path(__file__).resolve().parent.parent / '.env'
        
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # クォート除去
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value
            return True
    except Exception as e:
        print(f"Warning: Failed to load .env file: {e}")
    return False


def main():
    """メイン処理"""
    # 環境変数をロード
    load_env_file()


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
        status = "[OK]" if result else "[NG]"
        print(f"{status:6s} - {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("[OK] すべてのチェックをパスしました！")
        print("\nGUIを起動できます:")
        print("  python eye_muscle_gui_v1.py")
    else:
        print("[NG] いくつかの問題があります。上記のメッセージを確認してください。")
        print("\n学習済みモデルがない場合は、先に学習を実行してください:")
        print("  python train_nnunet.py --mode full --task_id 102")
    
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)