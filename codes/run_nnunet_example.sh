# codes/run_nnunet_example.sh
#!/bin/bash
# nnU-Net学習実行例
# このスクリプトは実際のデータパスに合わせて修正してください

echo "=== nnU-Net 眼筋セグメンテーション学習 ==="
echo ""
echo "このスクリプトは以下の手順を実行します:"
echo "1. 環境変数の設定"
echo "2. データの準備（JSON → NIfTI変換）"
echo "3. nnU-Netのプランニングと前処理"
echo "4. モデルの学習"
echo ""

# ========================================

# 1. 環境変数の設定
# ========================================
echo "Step 1: 環境変数を設定..."

# .envファイルのパス
if [ -f ".env" ]; then
    ENV_FILE=".env"
elif [ -f "../.env" ]; then
    ENV_FILE="../.env"
else
    echo "エラー: .envファイルが見つかりません。setup_nnunet.shを実行してください。"
    exit 1
fi

# .envを読み込んでエクスポート
export $(grep -v '^#' $ENV_FILE | xargs)

# ディレクトリ作成
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo "  ✓ 環境変数設定完了 (from $ENV_FILE)"


# ========================================
# 2. 必要なパッケージのインストール
# ========================================
echo ""
echo "Step 2: 必要なパッケージをインストール..."

pip install nnunetv2 SimpleITK nibabel opencv-python --break-system-packages > /dev/null 2>&1

echo "  ✓ パッケージインストール完了"

# ========================================
# 3. データの準備
# ========================================
echo ""
echo "Step 3: データを準備..."
echo ""
echo "【重要】以下のパスを実際のデータに合わせて修正してください:"
echo ""
echo "  --json_dir: JSONアノテーションファイルのディレクトリ"
echo "           例: datasets/2024_07_11_09_34_02/filtered_json4"
echo ""
echo "  --image_dir: 画像ファイル（TIFF/DICOM）のディレクトリ"  
echo "            例: DATA/Tiffs/eT1W_SE_tra/"
echo ""
echo "修正例:"
echo "python prepare_nnunet_data.py \\"
echo "  --json_dir datasets/2024_07_11_09_34_02/filtered_json4 \\"
echo "  --image_dir DATA/Tiffs/eT1W_SE_tra/ \\"
echo "  --train_ratio 0.8"
echo ""
echo "データ準備が完了したら、以下のコマンドを実行してください:"
echo ""

# ========================================
# 4. nnU-Netのトレーニング
# ========================================
echo "Step 4: nnU-Netトレーニング"
echo ""
echo "以下のコマンドで学習を開始:"
echo ""
echo "# プランニングと前処理（初回のみ）"
echo "python train_nnunet.py --mode plan --task_id 501"
echo ""
echo "# トレーニング開始（100エポック、2D設定）"
echo "python train_nnunet.py --mode train --task_id 501 --epochs 100"
echo ""
echo "# または、全工程を一度に実行"
echo "python train_nnunet.py --mode full --task_id 501 --epochs 100"
echo ""

# ========================================
# 5. 学習済みモデルでの推論
# ========================================
echo "Step 5: 推論（学習完了後）"
echo ""
echo "学習が完了したら、以下のコマンドで推論:"
echo ""
echo "python train_nnunet.py --mode predict --task_id 501 \\"
echo "  --input_folder /path/to/test/images \\"
echo "  --output_folder /home/claude/predictions"
echo ""

# ========================================
# 注意事項
# ========================================
echo "========================================" 
echo "注意事項:"
echo "========================================" 
echo "1. 実際のデータパスに合わせて修正してください"
echo "2. GPUがある環境では学習が大幅に高速化されます"
echo "3. 2D画像の場合、configuration='2d'を使用"
echo "4. 初回実行時はプランニングに時間がかかります"
echo ""
echo "質問がある場合はお知らせください！"
