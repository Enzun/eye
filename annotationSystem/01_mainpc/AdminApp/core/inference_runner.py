import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal

class InferenceThread(QThread):
    progress = pyqtSignal(int, int, str)
    log_msg = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, raw_dir, output_dir, model_id=119, device="cuda",
                 dataset_name="EyeMuscleSegmentation"):
        super().__init__()
        self.raw_dir      = Path(raw_dir)
        self.output_dir   = Path(output_dir)
        self.model_id     = model_id
        self.device       = device
        self.dataset_name = dataset_name
        
    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_msg.emit(f"推論対象フォルダ: {self.raw_dir}")
            
            # 生データのリスト取得
            raw_files = list(self.raw_dir.glob("*_0000.nii.gz"))
            if not raw_files:
                self.log_msg.emit("推論対象の NIfTI (*_0000.nii.gz) が見つかりません。")
                self.finished.emit(True)
                return

            # 未推論のファイルを絞り込み
            to_predict = []
            for rf in raw_files:
                base_name = rf.name.replace("_0000.nii.gz", "")
                pred_file = self.output_dir / f"{base_name}.nii.gz"
                if not pred_file.exists():
                    to_predict.append(rf)
                    
            if not to_predict:
                self.log_msg.emit("すべての画像の予測結果がすでに存在します。")
                self.progress.emit(len(raw_files), len(raw_files), "推論完了")
                self.finished.emit(True)
                return
                
            self.log_msg.emit(f"未推論の画像 {len(to_predict)} 件に対して一括推論を開始します。")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_input = Path(tmpdir) / "input"
                tmp_output = Path(tmpdir) / "output"
                tmp_input.mkdir()
                tmp_output.mkdir()
                
                # 対象ファイルを temp にコピー
                for f in to_predict:
                    shutil.copy2(f, tmp_input / f.name)
                    
                # 実行
                self.log_msg.emit("nnUNet推論を実行中...")
                success = self.run_nnunet_subprocess(tmp_input, tmp_output)
                
                if success:
                    # 結果を本来の prediction フォルダへ移動
                    for pred_file in tmp_output.glob("*.nii.gz"):
                        shutil.copy2(pred_file, self.output_dir / pred_file.name)
                    self.log_msg.emit("推論結果を配置しました。")
                    self.finished.emit(True)
                else:
                    self.finished.emit(False)

        except Exception as e:
            self.log_msg.emit(f"エラー発生: {e}")
            self.finished.emit(False)

    def run_nnunet_subprocess(self, input_dir, output_dir):
        nnunet_results = os.environ.get('nnUNet_results')
        if not nnunet_results:
            self.log_msg.emit("エラー: nnUNet_results環境変数が設定されていません")
            return False
            
        model_folder = os.path.join(
            nnunet_results,
            f"Dataset{self.model_id:03d}_{self.dataset_name}",
            "nnUNetTrainer__nnUNetPlans__2d"
        )
        
        script_path = str(Path(__file__).parent.parent / 'utils' / 'run_nnunet.py')
        
        cmd = [
            sys.executable,
            script_path,
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-m", model_folder,
            "-f", "0",
            "-chk", "checkpoint_best.pth",
            "-device", self.device,
            "--disable_tta",
        ]
        
        try:
            # ログを随時キャプチャしながら実行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            for line in iter(process.stdout.readline, ''):
                self.log_msg.emit(line.strip())
                
            process.stdout.close()
            process.wait()
            
            if process.returncode == 0:
                self.log_msg.emit("nnUNetコマンド正常終了")
                return True
            else:
                self.log_msg.emit(f"nnUNetコマンド異常終了 (コード={process.returncode})")
                return False
                
        except Exception as e:
            self.log_msg.emit(f"推論サブプロセスでエラー: {e}")
            return False
