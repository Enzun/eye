# codes/train_nnunet.py
"""
nnU-Net v2 トレーニング実行スクリプト
眼筋セグメンテーションモデルの学習
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_environment():
    """環境変数の確認"""
    required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    
    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)
        else:
            print(f"✓ {var} = {os.environ.get(var)}")
    
    if missing:
        print("\n❌ 以下の環境変数が設定されていません:")
        for var in missing:
            print(f"  - {var}")
        print("\nsetup_nnunet.sh を実行するか、以下のコマンドで設定してください:")
        print('export nnUNet_raw="/home/claude/nnunet_data/nnUNet_raw"')
        print('export nnUNet_preprocessed="/home/claude/nnunet_data/nnUNet_preprocessed"')
        print('export nnUNet_results="/home/claude/nnunet_data/nnUNet_results"')
        return False
    
    return True


def run_planning(task_id: int):
    """データのプランニングと前処理を実行"""
    print("\n=== Step 1: データのプランニングと前処理 ===")
    
    cmd = f"nnUNetv2_plan_and_preprocess -d {task_id} --verify_dataset_integrity --clean"
    print(f"実行コマンド: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("✓ プランニング完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラーが発生しました:")
        print(e.stderr)
        return False


def run_training(task_id: int, fold: int = 0, configuration: str = "2d", 
                 trainer: str = "nnUNetTrainer", epochs: int = 100):
    """nnU-Netトレーニングを実行"""
    print(f"\n=== Step 2: モデルトレーニング (Fold {fold}) ===")
    dataset_name = f"Dataset{task_id:03d}_EyeMuscleSegmentation"
    
    # トレーニングコマンド構築
    cmd = (
        f"nnUNetv2_train {dataset_name} {configuration} {fold} "
        f"-tr {trainer} "
        f"--npz "  # 検証時にnpzファイルを保存
    )
    
    # if epochs != 1000:  # デフォルトは1000エポック
    #     cmd += f"--epochs {epochs} "
    
    print(f"実行コマンド: {cmd}")
    print("(これには時間がかかります...)")
    
    try:
        # トレーニングは時間がかかるのでリアルタイムで出力を表示
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # リアルタイムで出力を表示
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("✓ トレーニング完了")
            return True
        else:
            print(f"❌ トレーニングが失敗しました (終了コード: {process.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ トレーニングが中断されました")
        process.terminate()
        return False
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return False


def find_best_checkpoint(task_id: int, configuration: str = "2d", fold: int = 0):
    """最良のチェックポイントを探す"""
    print("\n=== 学習済みモデルの場所を確認 ===")
    results_dir = Path(os.environ.get('nnUNet_results', ''))
    dataset_name = f"Dataset{task_id:03d}_EyeMuscleSegmentation"
    
    checkpoint_dir = results_dir / dataset_name / "nnUNetTrainer__nnUNetPlans__" / configuration / f"fold_{fold}"
    
    if not checkpoint_dir.exists():
        print(f"チェックポイントディレクトリが見つかりません: {checkpoint_dir}")
        return None
    
    # checkpoint_best.pthを探す
    best_checkpoint = checkpoint_dir / "checkpoint_best.pth"
    if best_checkpoint.exists():
        return str(best_checkpoint)
    print("checkpoint_best.pth が見つかりません。checkpoint_final.pth を探します...")
    # checkpoint_final.pthを探す
    final_checkpoint = checkpoint_dir / "checkpoint_final.pth"
    if final_checkpoint.exists():
        return str(final_checkpoint)
    print("checkpoint_final.pth も見つかりません。")
    return None


def run_inference(task_id: int, input_folder: str, output_folder: str, 
                  configuration: str = "2d", fold: int = 0):
    """学習済みモデルで推論を実行"""
    print("\n=== 推論実行 ===")
    
    dataset_name = f"Dataset{task_id:03d}_EyeMuscleSegmentation"
    
    cmd = (
        f"nnUNetv2_predict "
        f"-i {input_folder} "
        f"-o {output_folder} "
        f"-d {dataset_name} "
        f"-c {configuration} "
        f"-f {fold} "
        f"-chk checkpoint_best.pth"
    )
    
    print(f"実行コマンド: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("✓ 推論完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラーが発生しました:")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='nnU-Net v2 トレーニング実行')
    parser.add_argument('--task_id', type=int, default=102,
                       help='タスクID (デフォルト: 102)')
    parser.add_argument('--mode', choices=['full', 'plan', 'train', 'predict'], 
                       default='full',
                       help='実行モード: full=全工程, plan=プランニングのみ, train=学習のみ, predict=推論のみ')
    parser.add_argument('--configuration', type=str, default='2d',
                       help='ネットワーク構成 (2d, 3d_fullres, 3d_lowres)')
    parser.add_argument('--fold', type=int, default=0,
                       help='クロスバリデーションのfold番号')
    parser.add_argument('--epochs', type=int, default=100,
                       help='トレーニングエポック数')
    parser.add_argument('--input_folder', type=str,
                       help='推論時の入力フォルダ')
    parser.add_argument('--output_folder', type=str, 
                       default='C:/Users/mitae/workspace/imageP/nnUNet_predictions',
                       help='推論時の出力フォルダ')
    
    args = parser.parse_args()
    
    # 環境変数チェック
    if not check_environment():
        sys.exit(1)
    
    # モード別実行
    if args.mode in ['full', 'plan']:
        if not run_planning(args.task_id):
            sys.exit(1)
    
    if args.mode in ['full', 'train']:
        if not run_training(args.task_id, args.fold, args.configuration,epochs=args.epochs):
            sys.exit(1)
    
    if args.mode == 'predict':
        if not args.input_folder:
            print("❌ 推論モードでは --input_folder が必要です")
            sys.exit(1)
        if not run_inference(args.task_id, args.input_folder, args.output_folder, 
                            args.configuration, args.fold):
            sys.exit(1)
    
    print("\n=== 処理完了 ===")
    
    # 学習済みモデルの場所を表示
    if args.mode in ['full', 'train']:
        checkpoint = find_best_checkpoint(args.task_id, args.configuration, args.fold)
        if checkpoint:
            print(f"\n学習済みモデル: {checkpoint}")
            print("\nGUIで使用する場合は、このモデルをロードしてください。")


if __name__ == "__main__":
    main()