# codes/setup_nnunet.sh
#!/bin/bash
# nnU-Net v2 セットアップスクリプト

echo "=== nnU-Net v2 環境セットアップ ==="

# 1. nnU-Net v2のインストール
echo "1. nnU-Net v2をインストール..."
pip install nnunetv2 --break-system-packages

# 2. 必要な依存関係のインストール
echo "2. 追加の依存関係をインストール..."
pip install SimpleITK nibabel matplotlib --break-system-packages

# 3. 環境変数の設定（必要に応じてパスを変更してください）
echo "3. 環境変数を設定..."
export nnUNet_raw="C:/Users/mitae/workspace/imageP/nnUNet_raw"
export nnUNet_preprocessed="C:/Users/mitae/workspace/imageP/nnUNet_preprocessed"
export nnUNet_results="C:/Users/mitae/workspace/imageP/nnUNet_results"

# ディレクトリの作成
echo "4. 必要なディレクトリを作成..."
mkdir -p $nnUNet_raw
mkdir -p $nnUNet_preprocessed
mkdir -p $nnUNet_results

echo ""
echo "=== セットアップ完了 ==="
echo "環境変数:"
echo "  nnUNet_raw=$nnUNet_raw"
echo "  nnUNet_preprocessed=$nnUNet_preprocessed"
echo "  nnUNet_results=$nnUNet_results"
echo ""