"""
眼筋アノテーションツール - 改良版
TIFF画像にポリゴンアノテーションを行い、JSON形式で保存
- マウスホイールでズーム
- ドラッグでパン
- ポリゴン完成時・画像切替時に自動保存
- ラベルごとのクリア機能
"""

import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QGroupBox,
    QRadioButton, QButtonGroup, QMessageBox, QScrollArea, QListWidgetItem,
    QCheckBox, QSlider
)
from PyQt5.QtCore import Qt, QPoint, QRect, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPolygon, QTransform, QPolygonF, QBrush
from PIL import Image
import numpy as np
import cv2


class AnnotationCanvas(QWidget):
    """アノテーション用のキャンバス（ズーム・パン対応）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        
        # 画像データ
        self.original_image = None
        self.display_image = None
        self.image_path = None
        self.display_pixmap = None  # 描画用pixmap
        
        # アノテーションデータ
        self.current_polygon = []  # 現在描画中のポリゴン
        self.annotations = []  # 完成したアノテーション [{label, points}]
        self.selected_annotation_idx = None
        
        # ズーム・パン設定
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 0.15
        self.pan_offset_x = 0.0  # パンオフセット（ピクセル）
        self.pan_offset_y = 0.0
        self.last_pan_pos = None
        self.is_panning = False
        
        # 編集モード
        self.edit_mode = False
        self.dragging_point_idx = None  # ドラッグ中の頂点インデックス
        
        # 描画設定
        self.point_radius = 4
        self.line_width = 2
        
        # ラベル色定義（BGR→RGB変換）
        self.label_colors = {
            "so": (255, 0, 0),    # 赤
            "io": (0, 255, 0),    # 緑
            "sr": (0, 0, 255),    # 青
            "ir": (255, 255, 0),  # 黄
            "lr": (255, 0, 255),  # マゼンタ
            "mr": (0, 255, 255),  # シアン
        }
        
        self.setMouseTracking(True)
        self.setMinimumSize(600, 600)
        self.setStyleSheet("border: 2px solid #666; background-color: #2b2b2b;")
        self.setFocusPolicy(Qt.StrongFocus)
    
    def load_image(self, image_path, reset_zoom=True):
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
        
        # ズームをリセット（設定による）
        if reset_zoom:
            self.zoom_level = 1.0
            self.pan_offset_x = 0.0
            self.pan_offset_y = 0.0
        
        self.update_display()
    
    def update_display(self):
        """表示を更新（画像のみ。アノテーションはpaintEventで描画）"""
        if self.original_image is None:
            return
        
        # クリーンな画像をQPixmapに変換（アノテーションは描画しない）
        img = self.original_image
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(q_img)
        
        # ズームレベル表示を更新
        if self.parent_window:
            self.parent_window.update_zoom_display()
        
        # 再描画をトリガー
        self.update()
    
    def image_to_screen(self, img_x, img_y):
        """画像座標をスクリーン座標に変換"""
        sx = img_x * self.zoom_level + self.pan_offset_x
        sy = img_y * self.zoom_level + self.pan_offset_y
        return sx, sy
    
    def paintEvent(self, event):
        """カスタム描画（ズーム・パン対応、アノテーションはスクリーン座標で描画）"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 背景を塗りつぶし
        painter.fillRect(self.rect(), QColor(43, 43, 43))
        
        if self.display_pixmap is None:
            return
        
        # ズーム後のサイズを計算
        scaled_width = int(self.display_pixmap.width() * self.zoom_level)
        scaled_height = int(self.display_pixmap.height() * self.zoom_level)
        
        # 描画位置（パンオフセット適用）
        x = int(self.pan_offset_x)
        y = int(self.pan_offset_y)
        
        # スケーリングして画像を描画
        scaled_pixmap = self.display_pixmap.scaled(
            scaled_width, scaled_height, 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # --- アノテーションをスクリーン座標で描画（固定サイズ） ---
        
        has_selection = self.selected_annotation_idx is not None
        
        # 完成したアノテーションを描画
        for idx, annotation in enumerate(self.annotations):
            label = annotation['label']
            points = annotation['points']
            color_tuple = self.label_colors.get(label, (128, 128, 128))
            is_selected = (idx == self.selected_annotation_idx)
            
            if has_selection and not is_selected:
                # 非選択: 半透明に薄く表示
                color = QColor(*color_tuple, 80)
                fill_alpha = 20
                line_w = 1
            else:
                # 選択中 or 選択なし: 通常表示
                color = QColor(*color_tuple)
                fill_alpha = 60
                line_w = 3 if is_selected else self.line_width
            
            if len(points) > 1:
                fill_color = QColor(*color_tuple, fill_alpha)
                screen_points = []
                for pt in points:
                    sx, sy = self.image_to_screen(pt[0], pt[1])
                    screen_points.append(QPointF(sx, sy))
                
                polygon = QPolygonF(screen_points)
                painter.setBrush(QBrush(fill_color))
                painter.setPen(QPen(color, line_w))
                painter.drawPolygon(polygon)
                painter.setBrush(Qt.NoBrush)
                
                # 選択中: 頂点を白丸で強調表示
                if is_selected:
                    for sp in screen_points:
                        # 白い外枠 + ラベル色の内側
                        painter.setPen(QPen(QColor(255, 255, 255), 2))
                        painter.setBrush(QBrush(QColor(*color_tuple)))
                        painter.drawEllipse(sp, 5, 5)
                    painter.setBrush(Qt.NoBrush)
            
            # ラベルテキスト表示
            if len(points) > 0:
                sx, sy = self.image_to_screen(points[0][0], points[0][1])
                painter.setPen(QPen(color))
                painter.setFont(painter.font())
                painter.drawText(int(sx), int(sy) - 8, label)
        
        # 現在描画中のポリゴンを描画
        if self.current_polygon:
            current_label = self.parent_window.get_current_label()
            color_tuple = self.label_colors.get(current_label, (255, 255, 255))
            color = QColor(*color_tuple)
            
            # 点を描画（固定サイズ）
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            for point in self.current_polygon:
                sx, sy = self.image_to_screen(point[0], point[1])
                painter.drawEllipse(QPointF(sx, sy), self.point_radius, self.point_radius)
            painter.setBrush(Qt.NoBrush)
            
            # 線を描画（固定幅）
            if len(self.current_polygon) > 1:
                painter.setPen(QPen(color, self.line_width))
                for i in range(len(self.current_polygon) - 1):
                    sx1, sy1 = self.image_to_screen(*self.current_polygon[i])
                    sx2, sy2 = self.image_to_screen(*self.current_polygon[i+1])
                    painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))
            
            # 閉じるプレビュー線
            if len(self.current_polygon) > 2:
                pen = QPen(color, 1, Qt.DashLine)
                painter.setPen(pen)
                sx1, sy1 = self.image_to_screen(*self.current_polygon[-1])
                sx2, sy2 = self.image_to_screen(*self.current_polygon[0])
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))
    
    def wheelEvent(self, event):
        """マウスホイールでズーム（カーソル位置を中心に）"""
        if self.original_image is None:
            return
        
        # マウス位置を取得
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()
        
        # ズーム前のマウス位置に対応する画像座標を計算
        old_img_x = (mouse_x - self.pan_offset_x) / self.zoom_level
        old_img_y = (mouse_y - self.pan_offset_y) / self.zoom_level
        
        # ズームレベルを更新
        old_zoom = self.zoom_level
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_level = min(self.max_zoom, self.zoom_level * 1.15)
        else:
            self.zoom_level = max(self.min_zoom, self.zoom_level / 1.15)
        
        # マウス位置が同じ画像座標を指すようにパンオフセットを調整
        self.pan_offset_x = mouse_x - old_img_x * self.zoom_level
        self.pan_offset_y = mouse_y - old_img_y * self.zoom_level
        
        self.update()
    
    def mousePressEvent(self, event):
        """マウスクリックイベント"""
        if self.original_image is None:
            return
        
        if event.button() == Qt.MiddleButton:
            # 中ボタンでパン開始
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        
        elif event.button() == Qt.LeftButton:
            if self.edit_mode:
                # 編集モード: 頂点のドラッグ開始または編集モード終了
                nearest = self.find_nearest_vertex(event.pos())
                if nearest is not None:
                    self.dragging_point_idx = nearest
                    self.setCursor(Qt.ClosedHandCursor)
                else:
                    # 空いた場所をクリック → 編集モード終了
                    self.exit_edit_mode()
            else:
                # 通常モード: 点を追加
                pos = self.map_to_image_coords(event.pos())
                if pos:
                    self.current_polygon.append(pos)
                    self.update_display()
        
        elif event.button() == Qt.RightButton:
            if self.edit_mode:
                # 編集モード中の右クリック→編集モード終了
                self.exit_edit_mode()
            else:
                # 右クリックでポリゴンを完成
                if len(self.current_polygon) >= 3:
                    self.finish_polygon()
                elif len(self.current_polygon) > 0:
                    QMessageBox.warning(self, "警告", "最低3点必要です")
    
    def find_nearest_vertex(self, widget_pos):
        """選択中アノテーションの頂点の中で、クリック位置に最も近いものを検出"""
        if self.selected_annotation_idx is None:
            return None
        
        annotation = self.annotations[self.selected_annotation_idx]
        points = annotation['points']
        
        hit_threshold = 10  # スクリーンピクセルでの判定距離
        min_dist = float('inf')
        nearest_idx = None
        
        for i, pt in enumerate(points):
            sx, sy = self.image_to_screen(pt[0], pt[1])
            dx = widget_pos.x() - sx
            dy = widget_pos.y() - sy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < hit_threshold and dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def exit_edit_mode(self):
        """編集モードを終了"""
        self.edit_mode = False
        self.selected_annotation_idx = None
        self.dragging_point_idx = None
        if self.parent_window:
            self.parent_window.annotation_list.clearSelection()
        self.update_display()
    
    def mouseMoveEvent(self, event):
        """マウス移動イベント"""
        if self.is_panning and self.last_pan_pos:
            delta = event.pos() - self.last_pan_pos
            self.pan_offset_x += delta.x()
            self.pan_offset_y += delta.y()
            self.last_pan_pos = event.pos()
            self.update_display()
        elif self.edit_mode and self.dragging_point_idx is not None:
            # 編集モード: 頂点をドラッグ中
            pos = self.map_to_image_coords(event.pos())
            if pos and self.selected_annotation_idx is not None:
                self.annotations[self.selected_annotation_idx]['points'][self.dragging_point_idx] = pos
                self.update()
        elif not self.is_panning:
            # カーソル変更
            if self.edit_mode and self.find_nearest_vertex(event.pos()) is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                pos = self.map_to_image_coords(event.pos())
                if pos is not None:
                    self.setCursor(Qt.CrossCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        """マウスリリースイベント"""
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton and self.dragging_point_idx is not None:
            # ドラッグ終了 → 自動保存
            self.dragging_point_idx = None
            self.setCursor(Qt.ArrowCursor)
            if self.parent_window:
                self.parent_window.auto_save_json()
            self.update_display()
    
    def map_to_image_coords(self, widget_pos):
        """ウィジェット座標を画像座標に変換（パン・ズーム考慮）"""
        if self.original_image is None:
            return None
        
        # パンオフセットを考慮してズーム画像上の位置を計算
        zoomed_x = widget_pos.x() - self.pan_offset_x
        zoomed_y = widget_pos.y() - self.pan_offset_y
        
        # 画像座標に変換
        image_x = zoomed_x / self.zoom_level
        image_y = zoomed_y / self.zoom_level
        
        # 範囲チェック
        img_h, img_w = self.original_image.shape[:2]
        if image_x < 0 or image_x >= img_w or image_y < 0 or image_y >= img_h:
            return None
        
        # 四捨五入で座標を変換（ずれを最小化）
        return [round(image_x), round(image_y)]
    
    def widget_to_image_coords(self, widget_pos):
        """ウィジェット座標を画像座標に変換（float版）"""
        if self.original_image is None:
            return None
        
        # パンオフセットを考慮してズーム画像上の位置を計算
        zoomed_x = widget_pos.x() - self.pan_offset_x
        zoomed_y = widget_pos.y() - self.pan_offset_y
        
        # 画像座標に変換
        image_x = zoomed_x / self.zoom_level
        image_y = zoomed_y / self.zoom_level
        
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
        
        # 自動保存
        self.parent_window.auto_save_json()
        
        self.update_display()
    
    def delete_selected_annotation(self):
        """選択されたアノテーションを削除"""
        if self.selected_annotation_idx is not None:
            del self.annotations[self.selected_annotation_idx]
            self.selected_annotation_idx = None
            self.parent_window.update_annotation_list()
            self.parent_window.auto_save_json()
            self.update_display()
    
    def clear_annotations_by_label(self, label):
        """指定されたラベルのアノテーションを削除"""
        self.annotations = [a for a in self.annotations if a['label'] != label]
        self.selected_annotation_idx = None
        self.parent_window.update_annotation_list()
        self.parent_window.auto_save_json()
        self.update_display()
    
    def clear_current_polygon(self):
        """現在描画中のポリゴンをクリア"""
        self.current_polygon = []
        self.update_display()
    
    def undo_last_point(self):
        """最後に追加した点を1つ削除"""
        if self.current_polygon:
            self.current_polygon.pop()
            self.update_display()
    
    def reset_zoom(self):
        """ズームをリセット"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update_display()
    
    def get_annotations_data(self):
        """アノテーションデータを取得（JSON保存用）"""
        if self.original_image is None or self.image_path is None:
            return None
        
        # shapesにflags, group_idを追加
        shapes_with_metadata = []
        for annotation in self.annotations:
            shape = annotation.copy()
            shape['group_id'] = None
            shape['flags'] = {}
            shapes_with_metadata.append(shape)
        
        data = {
            "version": "4.5.9",
            "flags": {},
            "shapes": shapes_with_metadata,
            "imagePath": Path(self.image_path).name,
            "imageData": None,
            "imageHeight": self.original_image.shape[0],
            "imageWidth": self.original_image.shape[1]
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
        
        # 作業ディレクトリ
        self.work_dir = None
        self.series_name = None  # シリーズ名（親フォルダ名）
        self.current_file_index = 0
        self.file_list = []
        
        # プロジェクトルート（annotation_tool.pyの2つ上）
        self.project_root = Path(__file__).parent.parent
        self.annotated_data_dir = self.project_root / "annotated_data"
        self.data_input_dir = self.project_root / "data_input" / "tiff"
        
        # ズーム維持設定
        self.maintain_zoom_on_switch = True
        
        self.init_ui()
        
        # 起動時にシリーズを自動スキャン
        self.scan_series()
    
    def init_ui(self):
        """UIを初期化"""
        self.setWindowTitle("眼筋アノテーションツール v2.0")
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
        title = QLabel("アノテーションツール v2.0")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        # シリーズ選択グループ
        series_group = QGroupBox("シリーズ選択")
        series_layout = QVBoxLayout()
        
        self.series_list = QListWidget()
        self.series_list.itemClicked.connect(self.on_series_selected)
        self.series_list.setMaximumHeight(150)
        series_layout.addWidget(self.series_list)
        
        self.refresh_btn = QPushButton("🔄 更新")
        self.refresh_btn.clicked.connect(self.scan_series)
        series_layout.addWidget(self.refresh_btn)
        
        series_group.setLayout(series_layout)
        layout.addWidget(series_group)
        
        # ファイル情報・ナビゲーション
        file_group = QGroupBox("ファイル")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("シリーズを選択してください")
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
        
        # ズームリセット
        zoom_layout = QHBoxLayout()
        self.reset_zoom_btn = QPushButton("🔄 ズーム/位置リセット")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(self.reset_zoom_btn)
        layout.addLayout(zoom_layout)
        
        self.maintain_zoom_checkbox = QCheckBox("画像切替時にズームを維持")
        self.maintain_zoom_checkbox.setChecked(True)
        self.maintain_zoom_checkbox.toggled.connect(self.on_maintain_zoom_changed)
        layout.addWidget(self.maintain_zoom_checkbox)
        
        # ラベル選択グループ
        label_group = QGroupBox("ラベル選択")
        label_layout = QVBoxLayout()
        
        # 筋肉選択
        label_layout.addWidget(QLabel("筋肉:"))
        self.muscle_group = QButtonGroup()
        self.muscle_radios = {}
        
        muscle_names = {
            "ir": "下直筋 (ir)",
            "mr": "内直筋 (mr)",
            "sr": "上直筋 (sr)",
            "so": "上斜筋 (so)",
            "lr": "外直筋 (lr)",
            "io": "下斜筋 (io)"
        }
        
        for i, muscle_id in enumerate(self.muscle_labels):
            radio = QRadioButton(f"{i+1}. {muscle_names[muscle_id]}")
            radio.setProperty("muscle_id", muscle_id)
            radio.toggled.connect(self.on_muscle_changed)
            self.muscle_group.addButton(radio)
            self.muscle_radios[muscle_id] = radio
            label_layout.addWidget(radio)
            
            if muscle_id == "sr":
                radio.setChecked(True)
        
        # 現在のラベル表示
        self.current_label_display = QLabel()
        self.current_label_display.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 10px; "
            "background-color: #0d47a1; border-radius: 5px; text-align: center;"
        )
        label_layout.addWidget(self.current_label_display)
        
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)
        
        # 描画操作ボタン
        draw_layout = QHBoxLayout()
        
        self.finish_btn = QPushButton("ポリゴン完成")
        self.finish_btn.clicked.connect(self.canvas.finish_polygon)
        draw_layout.addWidget(self.finish_btn)
        
        self.clear_current_btn = QPushButton("描画クリア")
        self.clear_current_btn.clicked.connect(self.canvas.clear_current_polygon)
        draw_layout.addWidget(self.clear_current_btn)
        
        layout.addLayout(draw_layout)
        
        # ラベル別クリアボタン
        self.clear_label_btn = QPushButton("選択ラベルをクリア")
        self.clear_label_btn.clicked.connect(self.clear_current_label_annotations)
        self.clear_label_btn.setStyleSheet("background-color: #c62828;")
        layout.addWidget(self.clear_label_btn)
        
        # アノテーションリスト（大きく表示）
        anno_group = QGroupBox("アノテーション一覧")
        anno_layout = QVBoxLayout()
        
        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.on_annotation_selected)
        anno_layout.addWidget(self.annotation_list, 1)  # stretch=1 で空きスペースを使う
        
        self.delete_btn = QPushButton("選択を削除")
        self.delete_btn.clicked.connect(self.delete_annotation)
        anno_layout.addWidget(self.delete_btn)
        
        anno_group.setLayout(anno_layout)
        layout.addWidget(anno_group, 1)  # stretch=1 で残りスペースを埋める
        
        # 自動保存の情報表示
        save_info = QLabel("※ポリゴン完成・画像切替時に自動保存")
        save_info.setStyleSheet("color: #4caf50; font-size: 11px; padding: 3px;")
        layout.addWidget(save_info)
        
        # 操作一覧ボタン
        help_btn = QPushButton("❓ 操作一覧")
        help_btn.setStyleSheet(
            "background-color: #424242; font-size: 11px; padding: 5px;"
        )
        help_btn.clicked.connect(self.show_help_dialog)
        layout.addWidget(help_btn)
        
        # 初期化完了後にラベル表示を更新
        self.update_label_display()
        
        return panel
    
    def keyPressEvent(self, event):
        """キーボードショートカット"""
        key = event.key()
        
        # 1-6キーでラベル切替
        if Qt.Key_1 <= key <= Qt.Key_6:
            idx = key - Qt.Key_1
            if idx < len(self.muscle_labels):
                muscle_id = self.muscle_labels[idx]
                self.muscle_radios[muscle_id].setChecked(True)
        
        # 矢印キーで画像切替
        elif key == Qt.Key_Left:
            self.prev_file()
        elif key == Qt.Key_Right:
            self.next_file()
        
        # Escで現在の描画をクリア
        elif key == Qt.Key_Escape:
            self.canvas.clear_current_polygon()
        
        # Deleteで選択アノテーション削除
        elif key == Qt.Key_Delete:
            self.delete_annotation()
        
        # Ctrl+Zで1点戻る
        elif key == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.canvas.undo_last_point()
    
    def scan_series(self):
        """data_input/tiff/ をスキャンしてシリーズ一覧を表示"""
        self.series_list.clear()
        
        if not self.data_input_dir.exists():
            self.data_input_dir.mkdir(parents=True, exist_ok=True)
            self.statusBar().showMessage("data_input/tiff/ フォルダを作成しました", 3000)
            return
        
        # サブフォルダを検索（TIFFファイルを含むもののみ）
        series_folders = []
        for folder in sorted(self.data_input_dir.iterdir()):
            if folder.is_dir():
                tiff_count = len(list(folder.glob("*.tiff")) + list(folder.glob("*.tif")))
                if tiff_count > 0:
                    series_folders.append((folder.name, tiff_count))
        
        if not series_folders:
            self.series_list.addItem("（TIFFフォルダが見つかりません）")
            self.statusBar().showMessage(
                f"data_input/tiff/ にTIFFフォルダを配置してください", 5000)
            return
        
        # アノテーション済みの進捗も表示
        for series_name, tiff_count in series_folders:
            # アノテーション済みファイル数をカウント
            json_dir = self.annotated_data_dir / series_name
            annotated_count = 0
            if json_dir.exists():
                for json_file in json_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if data.get('shapes', []):
                            annotated_count += 1
                    except:
                        pass
            
            if annotated_count > 0:
                item_text = f"📁 {series_name}  ({tiff_count}枚, {annotated_count}済)"
            else:
                item_text = f"📁 {series_name}  ({tiff_count}枚)"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, series_name)  # フォルダ名を保持
            self.series_list.addItem(item)
        
        self.statusBar().showMessage(
            f"{len(series_folders)}個のシリーズを検出", 3000)
    
    def on_series_selected(self, item):
        """シリーズがクリックされたときの処理"""
        series_name = item.data(Qt.UserRole)
        if series_name is None:
            return  # 「見つかりません」メッセージの場合
        
        # 現在のアノテーションを保存
        if self.file_list:
            self.auto_save_json()
        
        self.series_name = series_name
        self.work_dir = self.data_input_dir / series_name
        
        # TIFFファイルを検索
        self.file_list = sorted(list(self.work_dir.glob("*.tiff")) + 
                               list(self.work_dir.glob("*.tif")))
        
        if not self.file_list:
            QMessageBox.warning(self, "警告", "TIFFファイルが見つかりません")
            return
        
        # 全TIFFファイルに対応するJSONを事前作成
        self.create_json_files_for_all()
        
        self.current_file_index = 0
        self.load_current_file()
        
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        
        self.statusBar().showMessage(
            f"シリーズ「{series_name}」を読み込みました ({len(self.file_list)}枚)", 3000)
    
    def get_json_save_dir(self):
        """JSONの保存先ディレクトリを取得（annotated_data/{series_name}/）"""
        if self.series_name is None:
            return None
        json_dir = self.annotated_data_dir / self.series_name
        json_dir.mkdir(parents=True, exist_ok=True)
        return json_dir
    
    def get_json_path(self, tiff_path):
        """TIFFファイルに対応するJSONのパスを取得"""
        json_dir = self.get_json_save_dir()
        if json_dir is None:
            return None
        return json_dir / (Path(tiff_path).stem + '.json')
    
    def create_json_files_for_all(self):
        """全TIFFファイルに対応するJSONファイルを作成（存在しない場合）"""
        json_dir = self.get_json_save_dir()
        if json_dir is None:
            return
        
        created_count = 0
        for tiff_path in self.file_list:
            json_path = self.get_json_path(tiff_path)
            if not json_path.exists():
                # 空のアノテーションデータを作成
                empty_data = {
                    "version": "4.5.9",
                    "flags": {},
                    "shapes": [],
                    "imagePath": tiff_path.name,
                    "imageData": None,
                    "imageHeight": 0,
                    "imageWidth": 0
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_data, f, indent=2, ensure_ascii=False)
                created_count += 1
        
        if created_count > 0:
            self.statusBar().showMessage(f"{created_count}個のJSONを {json_dir} に作成", 3000)
    
    def load_current_file(self):
        """現在のファイルを読み込み"""
        if not self.file_list:
            return
        
        current_file = self.file_list[self.current_file_index]
        
        # ズーム維持設定を反映
        reset_zoom = not self.maintain_zoom_on_switch
        self.canvas.load_image(str(current_file), reset_zoom=reset_zoom)
        
        # ファイル情報を表示
        self.file_label.setText(
            f"ファイル: {current_file.name}\n"
            f"シリーズ: {self.series_name}\n"
            f"({self.current_file_index + 1} / {len(self.file_list)})"
        )
        
        # 対応するJSONファイルがあれば自動読込
        json_path = self.get_json_path(current_file)
        if json_path and json_path.exists():
            self.load_json_from_path(json_path)
    
    def prev_file(self):
        """前のファイルに移動"""
        if self.current_file_index > 0:
            # 現在の画像を保存
            self.auto_save_json()
            
            self.current_file_index -= 1
            self.load_current_file()
    
    def next_file(self):
        """次のファイルに移動"""
        if self.current_file_index < len(self.file_list) - 1:
            # 現在の画像を保存
            self.auto_save_json()
            
            self.current_file_index += 1
            self.load_current_file()
    
    def auto_save_json(self):
        """現在のアノテーションを自動保存（annotated_data/{series_name}/ に保存）"""
        if not self.file_list:
            return
        
        data = self.canvas.get_annotations_data()
        if data is None:
            return
        
        current_file = self.file_list[self.current_file_index]
        json_path = self.get_json_path(current_file)
        
        if json_path is None:
            return
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.statusBar().showMessage(f"保存: {self.series_name}/{json_path.name}", 2000)
    
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
        color = self.canvas.label_colors.get(current_label, (128, 128, 128))
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        
        # 明度を計算して文字色を決定（明るい背景→黒文字、暗い背景→白文字）
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "#000000" if luminance > 128 else "#ffffff"
        
        self.current_label_display.setText(f"現在のラベル: {current_label}")
        self.current_label_display.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 10px; "
            f"background-color: {color_hex}; color: {text_color}; "
            f"border: 2px solid #fff; border-radius: 5px; text-align: center;"
        )
    
    def update_zoom_display(self):
        """ズーム表示を更新（表示を削除したのでno-op）"""
        pass
    
    def show_help_dialog(self):
        """操作一覧をポップアップ表示"""
        help_text = (
            "【マウス操作】\n"
            "　左クリック　　　点を追加\n"
            "　右クリック　　　ポリゴンを完成\n"
            "　ホイール　　　　ズーム\n"
            "　中ボタンドラッグ　パン（移動）\n"
            "\n"
            "【キーボード】\n"
            "　1-6キー　　　ラベル切替\n"
            "　← →　　　　画像切替\n"
            "　Ctrl+Z　　　1点戻る\n"
            "　Esc　　　　　描画をクリア\n"
            "　Delete　　　選択削除"
        )
        QMessageBox.information(self, "操作一覧", help_text)
    
    def reset_zoom(self):
        """ズームをリセット"""
        self.canvas.reset_zoom()
    
    def on_maintain_zoom_changed(self, checked):
        """ズーム維持設定が変更された"""
        self.maintain_zoom_on_switch = checked
    
    def get_current_label(self):
        """現在選択されているラベルを取得"""
        return self.current_muscle
    
    def update_annotation_list(self):
        """アノテーション一覧を更新"""
        self.annotation_list.clear()
        
        for idx, annotation in enumerate(self.canvas.annotations):
            label = annotation['label']
            num_points = len(annotation['points'])
            
            item = QListWidgetItem(f"{idx + 1}. {label} ({num_points}点)")
            self.annotation_list.addItem(item)
    
    def on_annotation_selected(self, item):
        """アノテーションが選択された → 編集モードに入る"""
        # 描画中のポリゴンがあれば確定
        if self.canvas.current_polygon and len(self.canvas.current_polygon) >= 3:
            self.canvas.finish_polygon()
        else:
            self.canvas.current_polygon = []
        
        self.canvas.selected_annotation_idx = self.annotation_list.row(item)
        self.canvas.edit_mode = True
        self.canvas.update_display()
    
    def delete_annotation(self):
        """選択されたアノテーションを削除"""
        self.canvas.delete_selected_annotation()
    
    def clear_current_label_annotations(self):
        """現在選択中のラベルのアノテーションをクリア"""
        current_label = self.get_current_label()
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self, "確認",
            f"ラベル「{current_label}」のアノテーションをすべて削除しますか？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.clear_annotations_by_label(current_label)
    
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
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
        }
    """)
    
    window = AnnotationTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()