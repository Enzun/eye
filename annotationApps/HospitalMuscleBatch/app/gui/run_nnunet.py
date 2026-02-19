"""
nnUNet推論用のサブプロセススクリプト
PyQt5とPyTorchのDLL競合を回避するため、別プロセスで推論を実行する

重要: import torch の前にkernel32.LoadLibraryWでDLLを事前ロードする
VC++ DLLは事前ロードしない（システムのVC++ Redistributableを使用する）
"""
import sys
import os
import ctypes

def preload_torch_dlls():
    """kernel32.LoadLibraryW を使ってtorch DLLを事前ロードする"""
    base = os.path.dirname(sys.executable)
    torch_lib = os.path.join(base, 'Lib', 'site-packages', 'torch', 'lib')
    
    if not os.path.exists(torch_lib):
        return
    
    # DLL検索パスに追加
    os.add_dll_directory(torch_lib)
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
    
    # kernel32を取得
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    
    # 依存関係順にDLLをロード
    # ※VC++ DLL（vcruntime140, msvcp140等）はシステムのものを使うため含めない
    dll_order = [
        'libiomp5md.dll',
        'libiompstubs5md.dll',
        'uv.dll',
        'c10.dll',
        'torch_global_deps.dll',
        'torch_cpu.dll',
        'torch_cuda.dll',  # GPU使用時のためにプリロード（存在しない場合はスキップ）
        'torch.dll',
        'torch_python.dll',
    ]
    
    for dll_name in dll_order:
        dll_path = os.path.join(torch_lib, dll_name)
        if os.path.exists(dll_path):
            res = kernel32.LoadLibraryW(dll_path)
            if not res:
                err = ctypes.get_last_error()
                print(f"WARNING: Failed to preload {dll_name}: WinError {err}")

def main():
    """predict_entry_point_modelfolder() を呼び出す"""
    # torchインポート前にDLLを事前ロード
    preload_torch_dlls()
    
    # nnUNet推論を実行
    from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder
    predict_entry_point_modelfolder()

if __name__ == '__main__':
    main()
