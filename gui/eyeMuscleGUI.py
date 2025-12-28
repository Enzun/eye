"""
眼筋セグメンテーションGUI - バージョン2
機能: NIfTIファイル選択、予測、画像・面積・体積表示
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QSpinBox, QComboBox, QMessageBox, QProgressBar, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import SimpleITK as sitk
import cv2
import tempfile
import subprocess


class PredictionThread(QThread):
    """予測処理を別スレッドで実行"""
    finished = pyqtSignal(object, dict, dict)  # 画像配列、面積データ、体積データ
    error = pyqtSignal(str)
    
    def __init__(self, nifti_path, predictor):
        super().__init__()
        self.nifti_path = nifti_path
        self.predictor = predictor
    
    def run(self):
        try:
            img_slices, slice_areas, volumes = self.predictor.predict_from_nifti(self.nifti_path)
            self.finished.emit(img_slices, slice_areas, volumes)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class NnUNetPredictor:
    """nnU-Net推論クラス（NIfTI対応版）"""
    
    def __init__(self, task_id=102, configuration="2d", fold=0, checkpoint="checkpoint_best.pth"):
        self.task_id = task_id
        self.configuration = configuration
        self.fold = fold
        self.checkpoint = checkpoint  # 使用するチェックポイント
        self.dataset_name = f"Dataset{task_id:03d}_EyeMuscleSegmentation"
        
        # ラベル定義
        self.label_names = {
            1: "ir",  # 下直筋
            2: "mr",  # 内直筋
            3: "sr",  # 上直筋
            4: "so",  # 上斜筋
            5: "lr",  # 外直筋
            6: "io",  # 下斜筋
        }
        
        # 色定義（BGR形式）
        self.label_colors = {
            "l_so": (0, 0, 255),      # 赤
            "r_so": (0, 0, 255),    
            "l_io": (0, 255, 0),      # 緑
            "r_io": (0, 255, 0),    
            "l_sr": (255, 0, 0),      # 青
            "r_sr": (255, 0, 0),    
            "l_ir": (0, 255, 255),    # 黄
            "r_ir": (0, 255, 255),  
            "l_lr": (255, 0, 255),    # マゼンタ
            "r_lr": (255, 0, 255),  
            "l_mr": (255, 255, 0),    # シアン
            "r_mr": (255, 255, 0),
        }
    
    def run_nnunet_inference(self, input_dir, output_dir):
        """nnU-Net推論を実行"""
        cmd = [
            "nnUNetv2_predict",
            "-i", input_dir,
            "-o", output_dir,
            "-d", str(self.task_id),
            "-c", self.configuration,
            "-f", str(self.fold),
            "-chk", self.checkpoint  # 設定されたチェックポイントを使用
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"推論成功: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = f"nnUNetコマンドエラー:\n"
            error_msg += f"コマンド: {' '.join(cmd)}\n"
            error_msg += f"終了コード: {e.returncode}\n"
            error_msg += f"標準エラー出力:\n{e.stderr}\n"
            error_msg += f"標準出力:\n{e.stdout}"
            print(error_msg)
            raise RuntimeError(error_msg)  # より詳細なエラーメッセージを投げる
    
    def visualize_slice(self, image_slice, pred_slice, spacing):
        """単一スライスの可視化と面積計算"""
        # 画像を8ビットに正規化
        img_normalized = ((image_slice - image_slice.min()) / 
                         (image_slice.max() - image_slice.min()) * 255).astype(np.uint8)
        
        # グレースケールからBGRに変換
        img_bgr = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
        
        img_height, img_width = image_slice.shape
        pixel_area_mm2 = spacing[1] * spacing[2]  # Y * X spacing
        
        label_areas = {}
        
        # 各ラベルの処理
        for label_id, label_name in self.label_names.items():
            mask = (pred_slice == label_id).astype(np.uint8)
            
            if mask.sum() == 0:
                continue
            
            # 輪郭検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 重心計算で左右判定
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                else:
                    center_x = img_width // 2
                
                side = "l" if center_x < img_width // 2 else "r"
                side_label = f"{side}_{label_name}"
                
                # 色取得
                color = self.label_colors.get(side_label, (128, 128, 128))
                
                # 面積計算（mm²）
                area_pixels = cv2.contourArea(contour)
                area_mm2 = area_pixels * pixel_area_mm2
                
                if side_label not in label_areas:
                    label_areas[side_label] = 0
                label_areas[side_label] += area_mm2
                
                # 描画
                cv2.polylines(img_bgr, [contour], isClosed=True, color=color, thickness=2)
                x, y = contour.min(axis=0)[0]
                cv2.putText(img_bgr, side_label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_bgr, label_areas
    
    def predict_from_nifti(self, nifti_path):
        """NIfTIファイルから予測を実行"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # NIfTIファイルをコピー
            input_file = input_dir / "case_0000.nii.gz"
            import shutil
            shutil.copy(nifti_path, input_file)
            
            print(f"入力ファイル: {input_file}")
            print(f"出力ディレクトリ: {output_dir}")
            
            # 推論実行
            try:
                self.run_nnunet_inference(str(input_dir), str(output_dir))
            except RuntimeError as e:
                # より詳細なエラー情報を提供
                raise RuntimeError(f"nnU-Net推論に失敗しました。\n\n詳細:\n{str(e)}")
            
            # 予測結果を読み込み
            pred_path = output_dir / "case.nii.gz"
            if not pred_path.exists():
                # 出力ディレクトリの内容を確認
                output_files = list(output_dir.glob("*"))
                raise FileNotFoundError(
                    f"予測結果が見つかりません: {pred_path}\n"
                    f"出力ディレクトリの内容: {[f.name for f in output_files]}"
                )
            
            # 元の画像を読み込み
            image_sitk = sitk.ReadImage(str(nifti_path))
            image_array = sitk.GetArrayFromImage(image_sitk)  # (Z, Y, X)
            
            # 予測結果を読み込み
            pred_sitk = sitk.ReadImage(str(pred_path))
            pred_array = sitk.GetArrayFromImage(pred_sitk)  # (Z, Y, X)
            
            # スペーシング情報取得
            spacing = image_sitk.GetSpacing()  # (X, Y, Z)
            
            num_slices = image_array.shape[0]
            
            # 各スライスを可視化
            visualized_slices = []
            slice_areas = {}  # {slice_idx: {label: area}}
            
            for i in range(num_slices):
                img_slice = image_array[i]
                pred_slice = pred_array[i]
                
                vis_img, areas = self.visualize_slice(img_slice, pred_slice, spacing)
                visualized_slices.append(vis_img)
                slice_areas[i] = areas
            
            # 体積計算
            volumes = {}
            slice_thickness = spacing[2]  # Z spacing
            
            # 各ラベルの体積を計算
            for label_id, label_name in self.label_names.items():
                for side in ['l', 'r']:
                    side_label = f"{side}_{label_name}"
                    total_volume = 0
                    
                    for slice_idx, areas in slice_areas.items():
                        if side_label in areas:
                            # 面積 × スライス厚 = 体積
                            total_volume += areas[side_label] * slice_thickness
                    
                    if total_volume > 0:
                        volumes[side_label] = total_volume
            
            return visualized_slices, slice_areas, volumes


class EyeMuscleGUI(QMainWindow):
    """眼筋セグメンテーションGUIメインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.current_nifti_path = None
        self.predictor = None
        self.prediction_thread = None
        self.visualized_slices = None
        self.slice_areas = None
        self.volumes = None
        self.current_slice_idx = 0
        
        self.init_ui()
    
    def init_ui(self):
        """UIを初期化"""
        self.setWindowTitle("眼筋セグメンテーション GUI v2 (NIfTI対応)")
        self.setGeometry(100, 100, 1400, 900)
        
        # メインウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左パネル（コントロール）
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右パネル（画像表示）
        right_panel = self.create_image_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_control_panel(self):
        """コントロールパネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # モデル設定グループ
        model_group = QGroupBox("モデル設定")
        model_layout = QVBoxLayout()
        
        # タスクID
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("タスクID:"))
        self.task_id_spin = QSpinBox()
        self.task_id_spin.setRange(1, 999)
        self.task_id_spin.setValue(102)
        task_layout.addWidget(self.task_id_spin)
        model_layout.addLayout(task_layout)
        
        # Configuration
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Configuration:"))
        self.config_combo = QComboBox()
        self.config_combo.addItems(["2d", "3d_fullres", "3d_lowres"])
        config_layout.addWidget(self.config_combo)
        model_layout.addLayout(config_layout)
        
        # Fold
        fold_layout = QHBoxLayout()
        fold_layout.addWidget(QLabel("Fold:"))
        self.fold_spin = QSpinBox()
        self.fold_spin.setRange(0, 4)
        self.fold_spin.setValue(0)
        fold_layout.addWidget(self.fold_spin)
        model_layout.addLayout(fold_layout)
        
        # Checkpoint
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(QLabel("Checkpoint:"))
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItems(["checkpoint_best.pth", "checkpoint_final.pth"])
        checkpoint_layout.addWidget(self.checkpoint_combo)
        model_layout.addLayout(checkpoint_layout)
        
        # モデル初期化ボタン
        self.init_model_btn = QPushButton("モデルを初期化")
        self.init_model_btn.clicked.connect(self.initialize_model)
        model_layout.addWidget(self.init_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # ファイル選択グループ
        file_group = QGroupBox("NIfTIファイル")
        file_layout = QVBoxLayout()
        
        self.select_file_btn = QPushButton("NIfTIファイルを選択")
        self.select_file_btn.clicked.connect(self.select_nifti_file)
        self.select_file_btn.setEnabled(False)
        file_layout.addWidget(self.select_file_btn)
        
        self.file_label = QLabel("ファイル: 未選択")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 予測ボタン
        self.predict_btn = QPushButton("予測を実行")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 体積テーブル
        volume_group = QGroupBox("筋肉の体積 (mm³)")
        volume_layout = QVBoxLayout()
        
        self.volume_table = QTableWidget()
        self.volume_table.setColumnCount(2)
        self.volume_table.setHorizontalHeaderLabels(["筋肉", "体積 (mm³)"])
        self.volume_table.horizontalHeader().setStretchLastSection(True)
        volume_layout.addWidget(self.volume_table)
        
        volume_group.setLayout(volume_layout)
        layout.addWidget(volume_group)
        
        # 面積テーブル（現在のスライス）
        area_group = QGroupBox("現在のスライスの面積 (mm²)")
        area_layout = QVBoxLayout()
        
        self.area_table = QTableWidget()
        self.area_table.setColumnCount(2)
        self.area_table.setHorizontalHeaderLabels(["筋肉", "面積 (mm²)"])
        self.area_table.horizontalHeader().setStretchLastSection(True)
        area_layout.addWidget(self.area_table)
        
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)
        
        layout.addStretch()
        
        return panel
    
    def create_image_panel(self):
        """画像表示パネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # タイトル
        title = QLabel("予測結果")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # スライス情報
        self.slice_info_label = QLabel("スライス: - / -")
        self.slice_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.slice_info_label)
        
        # 画像表示ラベル
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        self.image_label.setMinimumSize(600, 600)
        layout.addWidget(self.image_label)
        
        # スライススライダー
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("スライス選択:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slider_layout.addWidget(self.slice_slider)
        layout.addLayout(slider_layout)
        
        return panel
    
    def initialize_model(self):
        """モデルを初期化"""
        task_id = self.task_id_spin.value()
        configuration = self.config_combo.currentText()
        fold = self.fold_spin.value()
        checkpoint = self.checkpoint_combo.currentText()
        
        try:
            self.predictor = NnUNetPredictor(task_id, configuration, fold, checkpoint)
            QMessageBox.information(self, "成功", 
                f"モデルを初期化しました\n"
                f"タスク: {task_id}\n"
                f"Config: {configuration}\n"
                f"Fold: {fold}\n"
                f"Checkpoint: {checkpoint}")
            self.select_file_btn.setEnabled(True)
            self.init_model_btn.setText("モデルを再初期化")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"モデルの初期化に失敗しました:\n{str(e)}")
    
    def select_nifti_file(self):
        """NIfTIファイルを選択"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "NIfTIファイルを選択", "", 
            "NIfTI Files (*.nii.gz *.nii);;All Files (*.*)"
        )
        
        if file_path:
            self.current_nifti_path = file_path
            self.file_label.setText(f"ファイル: {Path(file_path).name}")
            self.predict_btn.setEnabled(True)
    
    def run_prediction(self):
        """予測を実行"""
        if not self.current_nifti_path or not self.predictor:
            return
        
        # UIを無効化
        self.predict_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不定形式
        
        # 予測スレッドを開始
        self.prediction_thread = PredictionThread(self.current_nifti_path, self.predictor)
        self.prediction_thread.finished.connect(self.on_prediction_finished)
        self.prediction_thread.error.connect(self.on_prediction_error)
        self.prediction_thread.start()
    
    def on_prediction_finished(self, visualized_slices, slice_areas, volumes):
        """予測完了時の処理"""
        self.visualized_slices = visualized_slices
        self.slice_areas = slice_areas
        self.volumes = volumes
        
        # スライダーを設定
        num_slices = len(visualized_slices)
        self.slice_slider.setRange(0, num_slices - 1)
        self.slice_slider.setValue(num_slices // 2)  # 中央のスライスを表示
        self.slice_slider.setEnabled(True)
        
        # 中央のスライスを表示
        self.current_slice_idx = num_slices // 2
        self.update_display()
        
        # 体積テーブルを更新
        self.update_volume_table(volumes)
        
        # UIを有効化
        self.predict_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "完了", 
            f"予測が完了しました\n総スライス数: {num_slices}")
    
    def on_prediction_error(self, error_msg):
        """予測エラー時の処理"""
        QMessageBox.critical(self, "エラー", f"予測に失敗しました:\n{error_msg}")
        
        self.predict_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def on_slice_changed(self, value):
        """スライダーの値が変更されたとき"""
        self.current_slice_idx = value
        self.update_display()
    
    def update_display(self):
        """画像と面積テーブルを更新"""
        if self.visualized_slices is None:
            return
        
        # スライス情報を更新
        total_slices = len(self.visualized_slices)
        self.slice_info_label.setText(
            f"スライス: {self.current_slice_idx + 1} / {total_slices}"
        )
        
        # 画像を表示
        img = self.visualized_slices[self.current_slice_idx]
        self.display_image(img)
        
        # 面積テーブルを更新
        areas = self.slice_areas.get(self.current_slice_idx, {})
        self.update_area_table(areas)
    
    def display_image(self, img):
        """画像を表示"""
        # NumPy配列をQImageに変換
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # QPixmapに変換してスケーリング
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def update_area_table(self, label_areas):
        """面積テーブルを更新"""
        self.area_table.setRowCount(len(label_areas))
        
        for i, (label, area) in enumerate(sorted(label_areas.items())):
            self.area_table.setItem(i, 0, QTableWidgetItem(label))
            self.area_table.setItem(i, 1, QTableWidgetItem(f"{area:.2f}"))
    
    def update_volume_table(self, volumes):
        """体積テーブルを更新"""
        self.volume_table.setRowCount(len(volumes))
        
        for i, (label, volume) in enumerate(sorted(volumes.items())):
            self.volume_table.setItem(i, 0, QTableWidgetItem(label))
            self.volume_table.setItem(i, 1, QTableWidgetItem(f"{volume:.2f}"))


def main():
    app = QApplication(sys.argv)
    
    # スタイル設定
    app.setStyle("Fusion")
    
    window = EyeMuscleGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()