"""
アノテーションツールのテスト用サンプル画像生成スクリプト
グレースケールのTIFF画像を生成します
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2


def create_sample_tiff(output_path, size=(512, 512)):
    """
    テスト用のサンプルTIFF画像を生成
    
    Args:
        output_path: 保存先パス
        size: 画像サイズ (width, height)
    """
    # グレースケール画像を作成
    img = np.zeros(size, dtype=np.uint8)
    
    # 背景にグラデーションを追加
    for i in range(size[0]):
        img[i, :] = int(100 + 50 * np.sin(i / size[0] * np.pi))
    
    # いくつかの円を描画（眼筋のシミュレーション）
    center_x, center_y = size[1] // 2, size[0] // 2
    
    # 中央の円（眼球）
    cv2.circle(img, (center_x, center_y), 80, 200, -1)
    
    # 周囲の楕円（筋肉のような形）
    positions = [
        (center_x - 150, center_y - 50),  # 左上
        (center_x + 150, center_y - 50),  # 右上
        (center_x - 150, center_y + 50),  # 左下
        (center_x + 150, center_y + 50),  # 右下
        (center_x, center_y - 120),       # 上
        (center_x, center_y + 120),       # 下
    ]
    
    for pos in positions:
        # 楕円を描画
        axes = (30, 60)
        angle = np.random.randint(0, 180)
        cv2.ellipse(img, pos, axes, angle, 0, 360, 180, -1)
        
        # 少しノイズを追加
        noise_level = 20
        x, y = pos
        noise_region = img[max(0, y-70):min(size[0], y+70), 
                          max(0, x-40):min(size[1], x+40)]
        noise = np.random.randint(-noise_level, noise_level, noise_region.shape, dtype=np.int16)
        noise_region = np.clip(noise_region.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img[max(0, y-70):min(size[0], y+70), 
            max(0, x-40):min(size[1], x+40)] = noise_region
    
    # 全体にノイズを追加
    noise = np.random.normal(0, 10, size).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # TIFFとして保存
    Image.fromarray(img).save(output_path, 'TIFF')
    
    return img


def create_sample_dataset(output_dir, num_images=5):
    """
    複数のサンプル画像を生成
    
    Args:
        output_dir: 出力ディレクトリ
        num_images: 生成する画像数
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"サンプル画像を生成中: {output_dir}")
    
    for i in range(1, num_images + 1):
        # ファイル名を実際のデータに近い形式にする
        filename = f"EX{i}SE1IMG{i:02d}.tiff"
        filepath = output_path / filename
        
        create_sample_tiff(filepath)
        print(f"  作成: {filename}")
    
    print(f"\n完了！ {num_images}枚の画像を生成しました。")
    print(f"場所: {output_path.absolute()}")
    print(f"\nアノテーションツールで開くには:")
    print(f"  python annotation_tool.py")
    print(f"  → 「フォルダを選択」で {output_path.name} を選択")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='アノテーションツール用サンプル画像生成')
    parser.add_argument('--output', '-o', default='sample_images',
                       help='出力ディレクトリ (デフォルト: sample_images)')
    parser.add_argument('--num', '-n', type=int, default=5,
                       help='生成する画像数 (デフォルト: 5)')
    parser.add_argument('--size', '-s', type=int, nargs=2, default=[512, 512],
                       help='画像サイズ [width height] (デフォルト: 512 512)')
    
    args = parser.parse_args()
    
    create_sample_dataset(args.output, args.num)


if __name__ == "__main__":
    main()
