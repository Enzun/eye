"""
眼筋アノテーションツール - 基本版
TIFF画像にポリゴンアノテーションを行い、JSON形式で保存
"""

import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QGroupBox,
    QRadioButton, QButtonGroup, QMessageBox, QScrollArea, QListWidgetItem
)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPolygon
from PIL import Image
import numpy as np
import cv2


class AnnotationCanvas(QLabel):
    """アノテーション用のキャンバス"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        
        # 画像データ
        self.original_image = None
        self.display_image = None
        self.image_path = None
        
        # アノテーションデータ
        self.current_polygon = []  # 現在描画中のポリゴン
        self.annotations = []  # 完成したアノテーション [{label, points}]
        self.selected_annotation_idx = None
        
        # 描画設定
        self.point_radius = 4
        self.line_width = 2
        
        # ラベル色定義（BGR→RGB変換）
        self.label_colors = {
            "l_so": (255, 0, 0),    # 赤
            "r_so": (255, 0, 0),    
            "l_io": (0, 255, 0),    # 緑
            "r_io": (0, 255, 0),    
            "l_sr": (0, 0, 255),    # 青
            "r_sr": (0, 0, 255),    
            "l_ir": (255, 255, 0),  # 黄
            "r_ir": (255, 255, 0),  
            "l_lr": (255, 0, 255),  # マゼンタ
            "r_lr": (255, 0, 255),  
            "l_mr": (0, 255, 255),  # シアン
            "r_mr": (0, 255, 255),
        }
        
        self.setMouseTracking(True)
        self.setMinimumSize(600, 600)
        self.setStyleSheet("border: 2px solid #666; background-color: #2b2b2b;")
    
    def load_image(self, image_path):
        """TIFF画像を読み込み"""
        self.image_path = image_path
        
        # PILで画像を読み込み
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # グレースケール変換
        if img_array.ndim == 3:
            img_array = np.mean(img_array, axis=2).astype(np.uint8)
        
        # 正規化して表示用に変換
        img_normalized = ((img_array - img_array.min()) / 
                         (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        # RGBに変換
        self.original_image = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        
        # アノテーションをクリア
        self.annotations = []
        self.current_polygon = []
        
        self.update_display()
    
    def update_display(self):
        """表示を更新"""
        if self.original_image is None:
            return
        
        # 画像をコピー
        img = self.original_image.copy()
        
        # 完成したアノテーションを描画
        for idx, annotation in enumerate(self.annotations):
            label = annotation['label']
            points = annotation['points']
            color = self.label_colors.get(label, (128, 128, 128))
            
            # 選択されているアノテーションは太く表示
            line_width = 4 if idx == self.selected_annotation_idx else self.line_width
            
            # ポリゴンを描画
            if len(points) > 1:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_width)
                
                # 半透明で塗りつぶし
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            # ラベルを表示
            if len(points) > 0:
                x, y = points[0]
                cv2.putText(img, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 現在描画中のポリゴンを描画
        if self.current_polygon:
            current_label = self.parent_window.get_current_label()
            color = self.label_colors.get(current_label, (255, 255, 255))
            
            # 点を描画
            for point in self.current_polygon:
                cv2.circle(img, tuple(point), self.point_radius, color, -1)
            
            # 線を描画
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=False, color=color, thickness=self.line_width)
            
            # 最初と最後の点を結ぶ線（閉じるプレビュー）
            if len(self.current_polygon) > 2:
                cv2.line(img, tuple(self.current_polygon[-1]), 
                        tuple(self.current_polygon[0]), color, 1, cv2.LINE_AA)
        
        # QPixmapに変換して表示
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # ウィンドウサイズに合わせてスケーリング
        scaled_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
    
    def mousePressEvent(self, event):
        """マウスクリックイベント"""
        if self.original_image is None:
            return
        
        if event.button() == Qt.LeftButton:
            # クリック位置を画像座標に変換
            pos = self.map_to_image_coords(event.pos())
            if pos:
                self.current_polygon.append(pos)
                self.update_display()
        
        elif event.button() == Qt.RightButton:
            # 右クリックでポリゴンを完成
            if len(self.current_polygon) >= 3:
                self.finish_polygon()
            else:
                QMessageBox.warning(self, "警告", "最低3点必要です")
    
    def map_to_image_coords(self, widget_pos):
        """ウィジェット座標を画像座標に変換"""
        if self.pixmap() is None:
            return None
        
        # ピクセルマップのサイズと位置を取得
        pixmap_rect = self.pixmap().rect()
        widget_rect = self.rect()
        
        # スケール比を計算
        scale_x = self.original_image.shape[1] / pixmap_rect.width()
        scale_y = self.original_image.shape[0] / pixmap_rect.height()
        
        # ピクセルマップの表示位置を計算（中央配置）
        x_offset = (widget_rect.width() - pixmap_rect.width()) / 2
        y_offset = (widget_rect.height() - pixmap_rect.height()) / 2
        
        # ウィジェット座標からピクセルマップ座標に変換
        pixmap_x = widget_pos.x() - x_offset
        pixmap_y = widget_pos.y() - y_offset
        
        # 範囲チェック
        if (pixmap_x < 0 or pixmap_x >= pixmap_rect.width() or
            pixmap_y < 0 or pixmap_y >= pixmap_rect.height()):
            return None
        
        # 画像座標に変換
        image_x = int(pixmap_x * scale_x)
        image_y = int(pixmap_y * scale_y)
        
        # 画像範囲内に制限
        image_x = max(0, min(image_x, self.original_image.shape[1] - 1))
        image_y = max(0, min(image_y, self.original_image.shape[0] - 1))
        
        return [image_x, image_y]
    
    def finish_polygon(self):
        """現在のポリゴンを完成させる"""
        if len(self.current_polygon) < 3:
            return
        
        current_label = self.parent_window.get_current_label()
        
        annotation = {
            'label': current_label,
            'points': self.current_polygon.copy(),
            'shape_type': 'polygon'
        }
        
        self.annotations.append(annotation)
        self.current_polygon = []
        
        # アノテーションリストを更新
        self.parent_window.update_annotation_list()
        
        self.update_display()
    
    def delete_selected_annotation(self):
        """選択されたアノテーションを削除"""
        if self.selected_annotation_idx is not None:
            del self.annotations[self.selected_annotation_idx]
            self.selected_annotation_idx = None
            self.parent_window.update_annotation_list()
            self.update_display()
    
    def clear_current_polygon(self):
        """現在描画中のポリゴンをクリア"""
        self.current_polygon = []
        self.update_display()
    
    def get_annotations_data(self):
        """アノテーションデータを取得（JSON保存用）"""
        if self.original_image is None or self.image_path is None:
            return None
        
        data = {
            "version": "1.1.0",
            "imageData": None,
            "imagePath": Path(self.image_path).name,
            "imageHeight": self.original_image.shape[0],
            "imageWidth": self.original_image.shape[1],
            "shapes": self.annotations.copy()
        }
        
        return data
    
    def load_annotations_data(self, data):
        """アノテーションデータを読み込み"""
        if 'shapes' in data:
            self.annotations = data['shapes']
            self.selected_annotation_idx = None
            self.update_display()
            self.parent_window.update_annotation_list()


class AnnotationTool(QMainWindow):
    """アノテーションツールのメインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        
        # 筋肉ラベル定義
        self.muscle_labels = ["ir", "mr", "sr", "so", "lr", "io"]
        self.current_muscle = "sr"  # デフォルト
        self.current_side = "l"     # デフォルト
        
        # 作業ディレクトリ
        self.work_dir = None
        self.current_file_index = 0
        self.file_list = []
        
        self.init_ui()
    
    def init_ui(self):
        """UIを初期化"""
        self.setWindowTitle("眼筋アノテーションツール v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # メインウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 先にキャンバスを作成（コントロールパネルから参照されるため）
        self.canvas = AnnotationCanvas(self)
        
        # 左パネル（コントロール）
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右パネル（キャンバス）
        main_layout.addWidget(self.canvas, 3)
    
    def create_control_panel(self):
        """コントロールパネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # タイトル
        title = QLabel("アノテーションツール")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        # ファイル選択グループ
        file_group = QGroupBox("ファイル操作")
        file_layout = QVBoxLayout()
        
        self.select_folder_btn = QPushButton("フォルダを選択")
        self.select_folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.select_folder_btn)
        
        self.select_file_btn = QPushButton("単一ファイルを選択")
        self.select_file_btn.clicked.connect(self.select_single_file)
        file_layout.addWidget(self.select_file_btn)
        
        self.file_label = QLabel("ファイル: 未選択")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        # ナビゲーションボタン
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 前")
        self.prev_btn.clicked.connect(self.prev_file)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("次 ▶")
        self.next_btn.clicked.connect(self.next_file)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        file_layout.addLayout(nav_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # ラベル選択グループ
        label_group = QGroupBox("ラベル選択")
        label_layout = QVBoxLayout()
        
        # 左右選択
        side_layout = QHBoxLayout()
        side_layout.addWidget(QLabel("左右:"))
        
        self.side_group = QButtonGroup()
        self.left_radio = QRadioButton("左 (l_)")
        self.left_radio.setChecked(True)
        self.left_radio.toggled.connect(self.on_side_changed)
        self.side_group.addButton(self.left_radio)
        side_layout.addWidget(self.left_radio)
        
        self.right_radio = QRadioButton("右 (r_)")
        self.right_radio.toggled.connect(self.on_side_changed)
        self.side_group.addButton(self.right_radio)
        side_layout.addWidget(self.right_radio)
        
        label_layout.addLayout(side_layout)
        
        # 筋肉選択
        label_layout.addWidget(QLabel("筋肉:"))
        self.muscle_group = QButtonGroup()
        
        muscle_names = {
            "ir": "下直筋 (ir)",
            "mr": "内直筋 (mr)",
            "sr": "上直筋 (sr)",
            "so": "上斜筋 (so)",
            "lr": "外直筋 (lr)",
            "io": "下斜筋 (io)"
        }
        
        for muscle_id in self.muscle_labels:
            radio = QRadioButton(muscle_names[muscle_id])
            radio.setProperty("muscle_id", muscle_id)
            radio.toggled.connect(self.on_muscle_changed)
            self.muscle_group.addButton(radio)
            label_layout.addWidget(radio)
            
            if muscle_id == "sr":
                radio.setChecked(True)
        
        # 現在のラベル表示
        self.current_label_display = QLabel()
        self.update_label_display()
        self.current_label_display.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 10px; "
            "background-color: #444; border-radius: 5px;"
        )
        label_layout.addWidget(self.current_label_display)
        
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)
        
        # 描画コントロール
        draw_group = QGroupBox("描画操作")
        draw_layout = QVBoxLayout()
        
        help_text = QLabel(
            "左クリック: 点を追加\n"
            "右クリック: ポリゴン完成\n"
            "(最低3点必要)"
        )
        help_text.setStyleSheet("color: #aaa; font-size: 11px;")
        draw_layout.addWidget(help_text)
        
        self.finish_btn = QPushButton("ポリゴンを完成")
        self.finish_btn.clicked.connect(self.canvas.finish_polygon)
        draw_layout.addWidget(self.finish_btn)
        
        self.clear_current_btn = QPushButton("現在の描画をクリア")
        self.clear_current_btn.clicked.connect(self.canvas.clear_current_polygon)
        draw_layout.addWidget(self.clear_current_btn)
        
        draw_group.setLayout(draw_layout)
        layout.addWidget(draw_group)
        
        # アノテーションリスト
        anno_group = QGroupBox("アノテーション一覧")
        anno_layout = QVBoxLayout()
        
        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.on_annotation_selected)
        anno_layout.addWidget(self.annotation_list)
        
        self.delete_btn = QPushButton("選択を削除")
        self.delete_btn.clicked.connect(self.delete_annotation)
        anno_layout.addWidget(self.delete_btn)
        
        anno_group.setLayout(anno_layout)
        layout.addWidget(anno_group)
        
        # 保存/読込
        save_group = QGroupBox("保存/読込")
        save_layout = QVBoxLayout()
        
        self.save_btn = QPushButton("JSONを保存")
        self.save_btn.clicked.connect(self.save_json)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        
        self.load_json_btn = QPushButton("JSONを読込")
        self.load_json_btn.clicked.connect(self.load_json)
        self.load_json_btn.setEnabled(False)
        save_layout.addWidget(self.load_json_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        layout.addStretch()
        
        # 初期化完了後にラベル表示を更新
        self.update_label_display()
        
        return panel
    
    def select_folder(self):
        """フォルダを選択してTIFFファイル一覧を取得"""
        folder = QFileDialog.getExistingDirectory(self, "TIFFフォルダを選択")
        
        if folder:
            self.work_dir = Path(folder)
            
            # TIFFファイルを検索
            self.file_list = sorted(list(self.work_dir.glob("*.tiff")) + 
                                   list(self.work_dir.glob("*.tif")))
            
            if not self.file_list:
                QMessageBox.warning(self, "警告", "TIFFファイルが見つかりません")
                return
            
            self.current_file_index = 0
            self.load_current_file()
            
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.load_json_btn.setEnabled(True)
    
    def select_single_file(self):
        """単一ファイルを選択"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "TIFFファイルを選択", "", 
            "TIFF Files (*.tiff *.tif);;All Files (*.*)"
        )
        
        if file_path:
            self.work_dir = Path(file_path).parent
            self.file_list = [Path(file_path)]
            self.current_file_index = 0
            self.load_current_file()
            
            self.save_btn.setEnabled(True)
            self.load_json_btn.setEnabled(True)
    
    def load_current_file(self):
        """現在のファイルを読み込み"""
        if not self.file_list:
            return
        
        current_file = self.file_list[self.current_file_index]
        self.canvas.load_image(str(current_file))
        
        # ファイル情報を表示
        self.file_label.setText(
            f"ファイル: {current_file.name}\n"
            f"({self.current_file_index + 1} / {len(self.file_list)})"
        )
        
        # 対応するJSONファイルがあれば自動読込
        json_path = current_file.with_suffix('.json')
        if json_path.exists():
            self.load_json_from_path(json_path)
    
    def prev_file(self):
        """前のファイルに移動"""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_current_file()
    
    def next_file(self):
        """次のファイルに移動"""
        if self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            self.load_current_file()
    
    def on_side_changed(self):
        """左右選択が変更された"""
        self.current_side = "l" if self.left_radio.isChecked() else "r"
        if hasattr(self, 'current_label_display'):
            self.update_label_display()
    
    def on_muscle_changed(self, checked):
        """筋肉選択が変更された"""
        if checked:
            sender = self.sender()
            self.current_muscle = sender.property("muscle_id")
            if hasattr(self, 'current_label_display'):
                self.update_label_display()
    
    def update_label_display(self):
        """現在のラベル表示を更新"""
        current_label = self.get_current_label()
        self.current_label_display.setText(f"現在のラベル: {current_label}")
    
    def get_current_label(self):
        """現在選択されているラベルを取得"""
        return f"{self.current_side}_{self.current_muscle}"
    
    def update_annotation_list(self):
        """アノテーション一覧を更新"""
        self.annotation_list.clear()
        
        for idx, annotation in enumerate(self.canvas.annotations):
            label = annotation['label']
            num_points = len(annotation['points'])
            
            item = QListWidgetItem(f"{idx + 1}. {label} ({num_points}点)")
            self.annotation_list.addItem(item)
    
    def on_annotation_selected(self, item):
        """アノテーションが選択された"""
        self.canvas.selected_annotation_idx = self.annotation_list.row(item)
        self.canvas.update_display()
    
    def delete_annotation(self):
        """選択されたアノテーションを削除"""
        self.canvas.delete_selected_annotation()
    
    def save_json(self):
        """JSONファイルを保存"""
        data = self.canvas.get_annotations_data()
        
        if data is None:
            QMessageBox.warning(self, "警告", "保存する画像がありません")
            return
        
        # デフォルトの保存先を提案
        if self.file_list:
            current_file = self.file_list[self.current_file_index]
            default_path = str(current_file.with_suffix('.json'))
        else:
            default_path = "annotation.json"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "JSONを保存", default_path, "JSON Files (*.json)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "成功", f"保存しました:\n{file_path}")
    
    def load_json(self):
        """JSONファイルを読み込み"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "JSONを読込", "", "JSON Files (*.json)"
        )
        
        if file_path:
            self.load_json_from_path(Path(file_path))
    
    def load_json_from_path(self, json_path):
        """指定されたパスからJSONを読み込み"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.canvas.load_annotations_data(data)
            
            # ステータスバーに表示
            self.statusBar().showMessage(f"読込: {json_path.name}", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"JSON読込に失敗:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    
    # ダークテーマ風のスタイル設定
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:pressed {
            background-color: #0a3d91;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #888;
        }
        QRadioButton {
            spacing: 5px;
        }
        QRadioButton::indicator {
            width: 15px;
            height: 15px;
        }
        QListWidget {
            background-color: #1e1e1e;
            border: 1px solid #555;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background-color: #0d47a1;
        }
        QLabel {
            color: #ffffff;
        }
    """)
    
    window = AnnotationTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()