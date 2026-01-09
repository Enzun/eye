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


# 3. 環境変数の設定
echo "3. 環境変数を設定..."

# .envファイルのパス (カレントディレクトリがルートであることを想定)
if [ -f ".env" ]; then
    ENV_FILE=".env"
elif [ -f "../.env" ]; then
    ENV_FILE="../.env"
else
    # デフォルトの.envを作成
    echo "  .envファイルを作成します..."
    cat > .env << EOL
nnUNet_raw="C:/Users/mitae/workspace/imageP/nnUNet_raw"
nnUNet_preprocessed="C:/Users/mitae/workspace/imageP/nnUNet_preprocessed"
nnUNet_results="C:/Users/mitae/workspace/imageP/nnUNet_results"
EOL
    ENV_FILE=".env"
fi

echo "  $ENV_FILE から環境変数を読み込み中..."
# .envを読み込んでエクスポート (Windows/Git Bash互換)
export $(grep -v '^#' $ENV_FILE | xargs)

# ディレクトリの作成
echo "4. 必要なディレクトリを作成..."
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo ""
echo "=== セットアップ完了 ==="
echo "環境変数 (from $ENV_FILE):"
echo "  nnUNet_raw=$nnUNet_raw"
echo "  nnUNet_preprocessed=$nnUNet_preprocessed"
echo "  nnUNet_results=$nnUNet_results"
echo ""