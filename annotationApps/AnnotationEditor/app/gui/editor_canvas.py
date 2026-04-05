"""
editor_canvas.py
アノテーション編集専用キャンバス

機能:
  - 読み込み時点から常に編集可能（モード切り替え不要）
  - 11 ラベル対応（labels.json から動的に設定）
  - Undo / Redo（スライス単位、最大 30 ステップ）
  - マスク表示 ON/OFF（Tab キー）
  - Window / Level によるコントラスト調整
  - ズーム・パン（マウスホイール / 中ボタン）
  - 頂点ドラッグ（選択ツール）
  - ポリゴン描画（ラベルボタン or キーボードショートカット）
"""

import copy
from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QWidget

MAX_UNDO = 30


class EditorCanvas(QWidget):
    """ポリゴン編集キャンバス"""

    # ポリゴンが変更されたとき（dirty フラグ管理用）
    slice_changed = pyqtSignal()

    def __init__(self, label_config, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.label_config = label_config

        # ─── 画像データ ───────────────────────────────────
        self.image_array = None    # (Y, X) グレースケール
        self.display_pixmap = None
        self.wc = 128.0            # Window Center
        self.ww = 256.0            # Window Width

        # ─── ポリゴンデータ ───────────────────────────────
        self.polygons = []
        self.selected_polygon_idx = None
        self.dragging_point_idx = None
        self._drag_start_polygons = None  # ドラッグ開始前スナップショット

        # ─── ラベル・描画状態 ─────────────────────────────
        self.current_label = 0        # 0 = 選択ツール
        self.current_polygon = []     # 描画中の点列

        # ─── 表示設定 ──────────────────────────────────────
        self.mask_visible = True
        self.polygon_opacity_mode = 0  # 0=通常, 1=薄め, 2=輪郭のみ

        # ─── ズーム・パン ────────────────────────────────
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.last_pan_pos = None
        self.is_panning = False

        # ─── スライス・Undo/Redo ─────────────────────────
        self.current_slice_idx = 0
        self._undo_stacks = {}  # {slice_idx: deque[list[dict]]}
        self._redo_stacks = {}

        self.setMouseTracking(True)
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setFocusPolicy(Qt.StrongFocus)

    # ════════════════════════════════════════════════════
    # 公開 API
    # ════════════════════════════════════════════════════

    def load_slice(self, image_array, polygons, slice_idx,
                   reset_zoom=True, wc=None, ww=None):
        """スライス画像とポリゴンを読み込む。

        Args:
            image_array: (Y, X) uint8 グレースケール
            polygons   : ポリゴンリスト
            slice_idx  : スライスインデックス
            reset_zoom : True でウィンドウにフィット
            wc, ww     : Window Center / Width（None で現在値を維持）
        """
        self.image_array = image_array
        self.polygons = copy.deepcopy(polygons)
        self.current_slice_idx = slice_idx
        self.selected_polygon_idx = None
        self.current_polygon = []

        if wc is not None:
            self.wc = wc
        if ww is not None:
            self.ww = ww

        self._render_image()

        if reset_zoom:
            self._fit_image_to_window()

        self.update()

    def load_empty(self):
        """表示をクリアして初期状態に戻す"""
        self.image_array = None
        self.display_pixmap = None
        self.polygons = []
        self.selected_polygon_idx = None
        self.current_polygon = []
        self.update()

    def get_polygons(self):
        """現在のポリゴンのコピーを返す"""
        return copy.deepcopy(self.polygons)

    def set_window_level(self, wc, ww):
        """Window / Level を更新して再描画する"""
        self.wc = float(wc)
        self.ww = float(max(1, ww))
        if self.image_array is not None:
            self._render_image()
            self.update()

    def can_undo(self):
        return bool(self._undo_stacks.get(self.current_slice_idx))

    def can_redo(self):
        return bool(self._redo_stacks.get(self.current_slice_idx))

    def undo(self):
        """一つ前の状態に戻す"""
        sidx = self.current_slice_idx
        stack = self._undo_stacks.get(sidx)
        if not stack:
            return
        if sidx not in self._redo_stacks:
            self._redo_stacks[sidx] = deque(maxlen=MAX_UNDO)
        self._redo_stacks[sidx].append(copy.deepcopy(self.polygons))
        self.polygons = stack.pop()
        self.selected_polygon_idx = None
        self.update()
        self.slice_changed.emit()

    def redo(self):
        """やり直す"""
        sidx = self.current_slice_idx
        stack = self._redo_stacks.get(sidx)
        if not stack:
            return
        if sidx not in self._undo_stacks:
            self._undo_stacks[sidx] = deque(maxlen=MAX_UNDO)
        self._undo_stacks[sidx].append(copy.deepcopy(self.polygons))
        self.polygons = stack.pop()
        self.selected_polygon_idx = None
        self.update()
        self.slice_changed.emit()

    # ════════════════════════════════════════════════════
    # 内部実装
    # ════════════════════════════════════════════════════

    def _push_undo(self):
        """現在のポリゴン状態を Undo スタックに積む"""
        sidx = self.current_slice_idx
        if sidx not in self._undo_stacks:
            self._undo_stacks[sidx] = deque(maxlen=MAX_UNDO)
        self._undo_stacks[sidx].append(copy.deepcopy(self.polygons))
        self._redo_stacks.pop(sidx, None)
        self.slice_changed.emit()

    def _render_image(self):
        """image_array + W/L 設定から display_pixmap を生成する"""
        if self.image_array is None:
            return
        lo = self.wc - self.ww / 2.0
        hi = self.wc + self.ww / 2.0
        arr = np.clip(self.image_array.astype(np.float32), lo, hi)
        if hi > lo:
            arr = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
        else:
            arr = np.zeros_like(self.image_array, dtype=np.uint8)
        img_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        h, w, _ = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(q_img)

    def _fit_image_to_window(self):
        """画像をウィンドウサイズにフィットさせる（95% 使用）"""
        if self.display_pixmap is None:
            return
        cw = max(self.width(), 600)
        ch = max(self.height(), 600)
        iw = self.display_pixmap.width()
        ih = self.display_pixmap.height()
        if iw > 0 and ih > 0:
            self.zoom_level = min(cw * 0.95 / iw, ch * 0.95 / ih)
            self.pan_offset_x = (cw - iw * self.zoom_level) / 2
            self.pan_offset_y = (ch - ih * self.zoom_level) / 2

    def showEvent(self, event):
        super().showEvent(event)
        if self.display_pixmap is not None:
            QTimer.singleShot(80, lambda: (self._fit_image_to_window(), self.update()))

    # ── 座標変換 ──────────────────────────────────────────

    def image_to_screen(self, ix, iy):
        return ix * self.zoom_level + self.pan_offset_x, iy * self.zoom_level + self.pan_offset_y

    def screen_to_image(self, sx, sy):
        if self.display_pixmap is None:
            return None, None
        ix = (sx - self.pan_offset_x) / self.zoom_level
        iy = (sy - self.pan_offset_y) / self.zoom_level
        w = self.display_pixmap.width()
        h = self.display_pixmap.height()
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return None, None
        return ix, iy

    # ── 描画 ──────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        if self.display_pixmap is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "ケースを選択してください")
            return

        # 画像
        sw = int(self.display_pixmap.width() * self.zoom_level)
        sh = int(self.display_pixmap.height() * self.zoom_level)
        scaled = self.display_pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(int(self.pan_offset_x), int(self.pan_offset_y), scaled)

        if not self.mask_visible:
            return

        # ポリゴン
        has_sel = self.selected_polygon_idx is not None
        for idx, poly in enumerate(self.polygons):
            if poly.get("is_hole"):
                continue
            label = poly["label"]
            pts = poly["points"]
            color = self.label_config.get_color(label)
            is_sel = (idx == self.selected_polygon_idx)
            is_cur = (label == self.current_label) and (self.current_label > 0)

            if is_sel:
                if self.polygon_opacity_mode == 0:
                    la, fa, lw = 255, 170, 3
                else:
                    la, fa, lw = 180, 80, 2
            elif is_cur and not has_sel:
                la, fa, lw = 255, 140, 2
            else:
                la, fa, lw = 200, 80, 1

            if len(pts) < 3:
                continue

            spts = [QPointF(*self.image_to_screen(x, y)) for x, y in pts]
            poly_f = QPolygonF(spts)
            r, g, b = color
            painter.setBrush(QBrush(QColor(r, g, b, fa)))
            painter.setPen(QPen(QColor(r, g, b, la), lw))
            painter.drawPolygon(poly_f)

            # 選択中は頂点表示
            if is_sel and self.polygon_opacity_mode < 2:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(r, g, b), 2))
                for p in spts:
                    painter.drawEllipse(p, 4, 4)

            # ラベル名をポリゴンの重心付近に表示
            if is_cur or is_sel:
                cx = sum(p.x() for p in spts) / len(spts)
                cy = sum(p.y() for p in spts) / len(spts)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(QPointF(cx + 4, cy - 4), self.label_config.get_name(label))

        # 描画中ポリゴン
        if self.current_polygon and self.current_label > 0:
            r, g, b = self.label_config.get_color(self.current_label)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(r, g, b)))
            for x, y in self.current_polygon:
                sx, sy = self.image_to_screen(x, y)
                painter.drawEllipse(QPointF(sx, sy), 4, 4)
            if len(self.current_polygon) > 1:
                painter.setPen(QPen(QColor(r, g, b), 2))
                for i in range(len(self.current_polygon) - 1):
                    x1, y1 = self.current_polygon[i]
                    x2, y2 = self.current_polygon[i + 1]
                    sx1, sy1 = self.image_to_screen(x1, y1)
                    sx2, sy2 = self.image_to_screen(x2, y2)
                    painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))
            # 最初の点と現在のカーソル位置を結ぶ補助線（2点以上のとき）
            # ※ mouseMoveEvent で update() するとリアルタイムプレビューになる

    # ── ホイール（ズーム） ────────────────────────────────

    def wheelEvent(self, event):
        if self.display_pixmap is None:
            return
        mx, my = event.pos().x(), event.pos().y()
        old_ix = (mx - self.pan_offset_x) / self.zoom_level
        old_iy = (my - self.pan_offset_y) / self.zoom_level
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.zoom_level = max(0.05, min(20.0, self.zoom_level * factor))
        self.pan_offset_x = mx - old_ix * self.zoom_level
        self.pan_offset_y = my - old_iy * self.zoom_level
        self.update()

    # ── 頂点・ポリゴン検索 ────────────────────────────────

    def _find_nearest_vertex(self, screen_pos):
        if self.selected_polygon_idx is None:
            return None
        pts = self.polygons[self.selected_polygon_idx]["points"]
        HIT = 12
        best_idx, best_d = None, float("inf")
        for i, (px, py) in enumerate(pts):
            sx, sy = self.image_to_screen(px, py)
            d = ((screen_pos.x() - sx) ** 2 + (screen_pos.y() - sy) ** 2) ** 0.5
            if d < HIT and d < best_d:
                best_idx, best_d = i, d
        return best_idx

    def _find_polygon_at(self, ix, iy):
        if ix is None:
            return None
        for idx, poly in enumerate(self.polygons):
            if poly.get("is_hole"):
                continue
            pts = np.array(poly["points"], dtype=np.int32)
            if len(pts) >= 3 and cv2.pointPolygonTest(pts, (float(ix), float(iy)), False) >= 0:
                return idx
        return None

    # ── マウスイベント ────────────────────────────────────

    def mousePressEvent(self, event):
        # 中ボタン: パン開始
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            if self.current_label == 0:
                # 選択ツール: 頂点ドラッグ or ポリゴン選択
                if self.selected_polygon_idx is not None:
                    v = self._find_nearest_vertex(event.pos())
                    if v is not None:
                        self._drag_start_polygons = copy.deepcopy(self.polygons)
                        self.dragging_point_idx = v
                        self.setCursor(Qt.ClosedHandCursor)
                        return
                ix, iy = self.screen_to_image(event.x(), event.y())
                self.selected_polygon_idx = self._find_polygon_at(ix, iy)
                self.update()
            else:
                # 描画モード: 点を追加
                ix, iy = self.screen_to_image(event.x(), event.y())
                if ix is not None:
                    self.current_polygon.append([ix, iy])
                    self.update()

        elif event.button() == Qt.RightButton:
            if self.current_label == 0 and self.selected_polygon_idx is not None:
                # 選択ツール: 透明度切り替え
                self.polygon_opacity_mode = (self.polygon_opacity_mode + 1) % 3
                self.update()
            elif self.current_label > 0 and len(self.current_polygon) >= 3:
                # 描画モード: ポリゴン完成
                self._finish_polygon()

    def mouseMoveEvent(self, event):
        if self.is_panning and self.last_pan_pos is not None:
            d = event.pos() - self.last_pan_pos
            self.pan_offset_x += d.x()
            self.pan_offset_y += d.y()
            self.last_pan_pos = event.pos()
            self.update()
            return

        if self.dragging_point_idx is not None and self.selected_polygon_idx is not None:
            ix, iy = self.screen_to_image(event.x(), event.y())
            if ix is not None:
                self.polygons[self.selected_polygon_idx]["points"][self.dragging_point_idx] = [ix, iy]
                self.update()
            return

        # カーソル形状
        if self.current_label == 0 and self.selected_polygon_idx is not None:
            v = self._find_nearest_vertex(event.pos())
            self.setCursor(Qt.PointingHandCursor if v is not None else Qt.ArrowCursor)
        else:
            self.setCursor(Qt.CrossCursor if self.current_label > 0 else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)

        if self.dragging_point_idx is not None:
            # ドラッグ完了 → Undo スタックに開始前状態を積む
            if self._drag_start_polygons is not None:
                sidx = self.current_slice_idx
                if sidx not in self._undo_stacks:
                    self._undo_stacks[sidx] = deque(maxlen=MAX_UNDO)
                self._undo_stacks[sidx].append(self._drag_start_polygons)
                self._redo_stacks.pop(sidx, None)
                self._drag_start_polygons = None
                self.slice_changed.emit()
            self.dragging_point_idx = None
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """ダブルクリック: 描画中のポリゴンを完成させる"""
        if self.current_label > 0 and len(self.current_polygon) >= 3:
            self._finish_polygon()

    # ── キーボード ────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        key_char = event.text().lower()

        # Ctrl+Z: Undo（描画中は最後の点を削除）
        if key == Qt.Key_Z and mods == Qt.ControlModifier:
            if self.current_polygon:
                self.current_polygon.pop()
                self.update()
            else:
                self.undo()
            return

        # Ctrl+Y: Redo
        if key == Qt.Key_Y and mods == Qt.ControlModifier:
            self.redo()
            return

        # Tab: マスク表示切り替え
        if key == Qt.Key_Tab:
            event.accept()
            self.mask_visible = not self.mask_visible
            self.update()
            if self.parent_window and hasattr(self.parent_window, "btn_mask_toggle"):
                self.parent_window.btn_mask_toggle.setChecked(self.mask_visible)
                self.parent_window.btn_mask_toggle.setText(
                    "マスク: ON" if self.mask_visible else "マスク: OFF"
                )
            return

        # Delete: 選択ポリゴン削除
        if key == Qt.Key_Delete:
            if self.selected_polygon_idx is not None:
                self._push_undo()
                del self.polygons[self.selected_polygon_idx]
                self.selected_polygon_idx = None
                self.update()
            return

        # Escape: 描画キャンセル
        if key == Qt.Key_Escape:
            self.current_polygon = []
            self.update()
            return

        # 矢印キー（修飾なし）: スライス移動を親に委譲
        if key in (Qt.Key_Left, Qt.Key_Right) and not mods:
            if self.parent_window:
                self.parent_window.keyPressEvent(event)
            return

        # 0: 選択ツール
        if key_char == "0":
            self._set_label(0)
            return

        # ラベルキー（1-9, q, w など）
        if key_char and not mods:
            label_id = self.label_config.key_to_label_id(key_char)
            if label_id is not None:
                self._set_label(label_id)

    def _set_label(self, label_id):
        self.current_label = label_id
        self.current_polygon = []  # 描画中をリセット
        self.update()
        if self.parent_window and hasattr(self.parent_window, "update_label_palette_selection"):
            self.parent_window.update_label_palette_selection(label_id)

    # ── ポリゴン操作 ──────────────────────────────────────

    def _finish_polygon(self):
        """描画中ポリゴンを確定する"""
        if len(self.current_polygon) < 3:
            return
        label_id = self.current_label
        label_name = self.label_config.get_name(label_id)
        self._push_undo()
        self.polygons.append({
            "label": label_id,
            "label_name": label_name,
            "points": self.current_polygon.copy(),
            "is_hole": False,
        })
        self.current_polygon = []
        self.update()

    def clear_label_polygons(self, label_id):
        """指定ラベルのポリゴンを全削除する"""
        new_polys = [p for p in self.polygons if p["label"] != label_id]
        if len(new_polys) < len(self.polygons):
            self._push_undo()
            self.polygons = new_polys
            self.selected_polygon_idx = None
            self.update()
