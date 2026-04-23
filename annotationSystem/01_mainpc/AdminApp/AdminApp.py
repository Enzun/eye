import sys
import os
from pathlib import Path

# スクリプト自身のディレクトリを sys.path に追加（portable Python 経由で起動した場合に ui/ が見つからない問題を回避）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# DLLパスの設定（ポータブル化対応 / nnUNet推論などでPyTorch・PyQt5のDLLが競合・欠損するのを回避）
if sys.platform == 'win32':
    _base = os.path.dirname(os.path.abspath(__file__))
    # annotationApps の python311 などを優先的に探す（既存のフォルダ構成にあわせる）
    # 目安: 01_mainpc/AdminApp から見て上位階層
    for _python_dir in [
        os.path.normpath(os.path.join(_base, '..', '..', 'python311')),
        os.path.normpath(os.path.join(sys.prefix)),
    ]:
        _dll_dirs = [
            os.path.join(_python_dir, 'Lib', 'site-packages', 'torch', 'lib'),
            os.path.join(_python_dir, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'bin'),
            _python_dir,
        ]
        for _d in _dll_dirs:
            if os.path.isdir(_d):
                os.add_dll_directory(_d)
                if _d not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def setup_env():
    """nnU-Net環境変数などの設定"""
    app_path = Path(__file__).resolve().parent
    # 01_mainpc/AdminApp -> 01_mainpc/predictions などは config.json で指定する形とし、
    # ここでは nnUNet_results などの環境変数をセット
    nnunet_results = app_path.parent.parent / 'nnUNet_results'
    if nnunet_results.exists():
        os.environ['nnUNet_results'] = str(nnunet_results)
        os.environ['nnUNet_raw'] = str(nnunet_results.parent / 'nnUNet_raw')
        os.environ['nnUNet_preprocessed'] = str(nnunet_results.parent / 'nnUNet_preprocessed')
    else:
        print(f"[Warning] nnUNet_results not found at {nnunet_results}")

def main():
    setup_env()
    app = QApplication(sys.argv)
    
    # スタイル調整
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
