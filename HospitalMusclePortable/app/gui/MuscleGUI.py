"""
病院用筋肉セグメンテーションGUI - バージョン1
機能: DICOMフォルダ選択、NIfTI変換、予測、結果保存/レビュー
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QSpinBox, QComboBox, QMessageBox, QProgressBar, QSlider,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import SimpleITK as sitk
import cv2
import tempfile
import subprocess


def get_app_path():
    """アプリケーションのベースパスを取得（exe化対応）"""
    if getattr(sys, 'frozen', False):
        # PyInstallerでexe化された場合
        return Path(sys._MEIPASS)
    else:
        # 通常のPython実行
        return Path(__file__).resolve().parent.parent


def setup_nnunet_env():
    """nnU-Net用の環境変数を設定（exe化対応）
    
    注意: 既に環境変数が設定されている場合（バッチファイルから起動など）は上書きしない
    """
    # 既に環境変数が設定されている場合はスキップ
    if os.environ.get('nnUNet_results'):
        print(f"nnUNet_results (既存): {os.environ.get('nnUNet_results')}")
        return
    
    app_path = get_app_path()
    
    if getattr(sys, 'frozen', False):
        # exe化された場合、バンドルされたモデルを使用
        nnunet_results = app_path / 'nnUNet_results'
    else:
        # 通常実行時は親の親ディレクトリ（imageProcessing）を参照
        nnunet_results = app_path.parent / 'nnUNet_results'
    
    os.environ['nnUNet_results'] = str(nnunet_results)
    os.environ['nnUNet_raw'] = str(nnunet_results.parent / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(nnunet_results.parent / 'nnUNet_preprocessed')
    
    print(f"nnUNet_results (新規設定): {os.environ.get('nnUNet_results')}")


# 起動時にnnU-Net環境変数を設定
setup_nnunet_env()

# 同じディレクトリのモジュールをインポート
from dicom_handler import convert_dicom_folder_to_nifti, save_temp_nifti
from result_manager import ResultManager


class ScrollableImageLabel(QLabel):
    """マウスホイールでスクロール可能な画像ラベル"""
    wheel_scrolled = pyqtSignal(int)  # delta
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
    
    def wheelEvent(self, event):
        """画像上でのマウスホイールイベント"""
        delta = event.angleDelta().y()
        self.wheel_scrolled.emit(delta)
        event.accept()


class PredictionThread(QThread):
    """予測処理を別スレッドで実行"""
    finished = pyqtSignal(object, dict, dict, object, object, object)
    error = pyqtSignal(str)
    
    def __init__(self, nifti_path, predictor):
        super().__init__()
        self.nifti_path = nifti_path
        self.predictor = predictor
    
    def run(self):
        try:
            result = self.predictor.predict_from_nifti(self.nifti_path)
            img_slices, slice_areas, volumes, image_array, pred_array, spacing = result
            self.finished.emit(img_slices, slice_areas, volumes, image_array, pred_array, spacing)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class NnUNetPredictor:
    """nnU-Net推論クラス（固定設定版）"""
    
    def __init__(self):
        # 固定設定
        self.task_id = 119
        self.configuration = "2d"
        self.fold = 0
        self.checkpoint = "checkpoint_best.pth"
        self.dataset_name = f"Dataset{self.task_id:03d}_EyeMuscleSegmentation"
        
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
            "-chk", self.checkpoint,
            "-device", "cpu"  # CPU版PyTorch用
        ]
        
        try:
            # Windowsではコンソールウィンドウを非表示にする
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                                   creationflags=creationflags)
            print(f"推論成功: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = f"nnUNetコマンドエラー:\n"
            error_msg += f"コマンド: {' '.join(cmd)}\n"
            error_msg += f"終了コード: {e.returncode}\n"
            error_msg += f"標準エラー出力:\n{e.stderr}\n"
            error_msg += f"標準出力:\n{e.stdout}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def visualize_slice(self, image_slice, pred_slice, spacing, show_labels=True):
        """単一スライスの可視化と面積計算"""
        # 画像を8ビットに正規化
        img_normalized = ((image_slice - image_slice.min()) / 
                         (image_slice.max() - image_slice.min()) * 255).astype(np.uint8)
        
        # グレースケールからBGRに変換
        img_bgr = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
        
        img_height, img_width = image_slice.shape
        pixel_area_mm2 = spacing[1] * spacing[2]  # Y * X spacing (mm²)
        pixel_area_cm2 = pixel_area_mm2 / 100  # mm² → cm² (1cm² = 100mm²)
        
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
                
                # 面積計算（cm²）
                area_pixels = cv2.contourArea(contour)
                area_cm2 = area_pixels * pixel_area_cm2
                
                if side_label not in label_areas:
                    label_areas[side_label] = 0
                label_areas[side_label] += area_cm2
                
                # 描画
                cv2.polylines(img_bgr, [contour], isClosed=True, color=color, thickness=2)
                if show_labels:
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
                raise RuntimeError(f"nnU-Net推論に失敗しました。\n\n詳細:\n{str(e)}")
            
            # 予測結果を読み込み
            pred_path = output_dir / "case.nii.gz"
            if not pred_path.exists():
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
            
            # 体積計算（cm³）
            volumes = {}
            slice_thickness_mm = spacing[2]  # Z spacing (mm)
            slice_thickness_cm = slice_thickness_mm / 10  # mm → cm
            
            # 各ラベルの体積を計算
            for label_id, label_name in self.label_names.items():
                for side in ['l', 'r']:
                    side_label = f"{side}_{label_name}"
                    total_volume = 0
                    
                    for slice_idx, areas in slice_areas.items():
                        if side_label in areas:
                            # 面積(cm²) × 厚さ(cm) = 体積(cm³)
                            total_volume += areas[side_label] * slice_thickness_cm
                    
                    if total_volume > 0:
                        volumes[side_label] = total_volume
            
            return visualized_slices, slice_areas, volumes, image_array, pred_array, spacing


class MuscleSegmentationGUI(QMainWindow):
    """筋肉セグメンテーションGUIメインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.current_dicom_folder = None
        self.current_folder_name = None
        self.predictor = NnUNetPredictor()  # 起動時に初期化
        self.prediction_thread = None
        self.visualized_slices = None
        self.slice_areas = None
        self.volumes = None
        self.current_slice_idx = 0
        
        # 動的再描画用のデータ
        self.image_array = None
        self.pred_array = None
        self.spacing = None
        self.show_labels = True
        self.zoom_level = 100
        
        # 結果管理（exe化対応）
        if getattr(sys, 'frozen', False):
            # exe化された場合、exeと同じディレクトリに出力
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).resolve().parent.parent
        self.result_manager = ResultManager(app_dir)
        
        self.init_ui()
    
    def init_ui(self):
        """UIを初期化"""
        self.setWindowTitle("筋肉セグメンテーション GUI v1 (病院用)")
        self.setGeometry(50, 50, 1600, 1000)
        
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
        
        # DICOMフォルダ選択グループ
        dicom_group = QGroupBox("DICOMフォルダ")
        dicom_layout = QVBoxLayout()
        
        self.select_folder_btn = QPushButton("📁 DICOMフォルダを選択")
        self.select_folder_btn.clicked.connect(self.select_dicom_folder)
        self.select_folder_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        dicom_layout.addWidget(self.select_folder_btn)
        
        self.folder_label = QLabel("フォルダ: 未選択")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #666;")
        dicom_layout.addWidget(self.folder_label)
        
        dicom_group.setLayout(dicom_layout)
        layout.addWidget(dicom_group)
        
        # 予測ボタン
        self.predict_btn = QPushButton("🔍 予測を実行")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.predict_btn)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 結果保存グループ
        save_group = QGroupBox("結果の保存")
        save_layout = QVBoxLayout()
        
        # 保存ボタン
        self.save_btn = QPushButton("✓ 結果を保存")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;"
        )
        save_layout.addWidget(self.save_btn)
        
        # レビューボタン
        self.review_btn = QPushButton("✗ 要手動レビュー")
        self.review_btn.clicked.connect(self.mark_for_review)
        self.review_btn.setEnabled(False)
        self.review_btn.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #f44336; color: white;"
        )
        save_layout.addWidget(self.review_btn)
        
        # レビュー待ち件数
        self.review_count_label = QLabel("")
        self.review_count_label.setStyleSheet("color: #666; font-size: 12px;")
        self.update_review_count()
        save_layout.addWidget(self.review_count_label)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # 表示設定グループ
        display_group = QGroupBox("表示設定")
        display_layout = QVBoxLayout()
        
        self.show_labels_checkbox = QCheckBox("画像上にラベル名を表示")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.stateChanged.connect(self.on_show_labels_changed)
        display_layout.addWidget(self.show_labels_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 筋肉の順序と色定義（凡例順）
        self.muscle_order = [
            ("so", "上斜筋", "#FF0000"),  # 赤
            ("io", "下斜筋", "#00FF00"),  # 緑
            ("sr", "上直筋", "#0000FF"),  # 青
            ("ir", "下直筋", "#FFFF00"),  # 黄
            ("lr", "外直筋", "#FF00FF"),  # マゼンタ
            ("mr", "内直筋", "#00FFFF"),  # シアン
        ]
        
        # 体積ツリー
        volume_group = QGroupBox("筋肉の体積 (cm³)")
        volume_layout = QVBoxLayout()
        
        self.volume_tree = QTreeWidget()
        self.volume_tree.setHeaderLabels(["筋肉", "体積 (cm³)"])
        self.volume_tree.setColumnCount(2)
        self.volume_tree.header().setStretchLastSection(True)
        self.volume_tree.setIndentation(20)
        volume_layout.addWidget(self.volume_tree)
        
        volume_group.setLayout(volume_layout)
        layout.addWidget(volume_group)
        
        # 面積ツリー
        area_group = QGroupBox("現在のスライスの面積 (cm²)")
        area_layout = QVBoxLayout()
        
        self.area_tree = QTreeWidget()
        self.area_tree.setHeaderLabels(["筋肉", "面積 (cm²)"])
        self.area_tree.setColumnCount(2)
        self.area_tree.header().setStretchLastSection(True)
        self.area_tree.setIndentation(20)
        area_layout.addWidget(self.area_tree)
        
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)
        
        layout.addStretch()
        
        return panel
    
    def create_image_panel(self):
        """画像表示パネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # タイトルとスライス情報
        header_layout = QHBoxLayout()
        title = QLabel("予測結果")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        self.slice_info_label = QLabel("スライス: - / -")
        header_layout.addWidget(self.slice_info_label)
        layout.addLayout(header_layout)
        
        # ズームスライダー
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("ズーム:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(40)
        zoom_layout.addWidget(self.zoom_label)
        layout.addLayout(zoom_layout)
        
        # 画像表示ラベル
        self.image_label = ScrollableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #1a1a1a;")
        self.image_label.setMinimumSize(700, 700)
        self.image_label.wheel_scrolled.connect(self.on_image_wheel_scrolled)
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
    
    def select_dicom_folder(self):
        """DICOMフォルダを選択"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "DICOMフォルダを選択 (SE○フォルダ)", ""
        )
        
        if folder_path:
            self.current_dicom_folder = folder_path
            # フォルダ名を取得
            from dicom_handler import get_folder_name
            self.current_folder_name = get_folder_name(folder_path)
            
            self.folder_label.setText(f"フォルダ: {self.current_folder_name}")
            self.folder_label.setStyleSheet("color: #000;")
            self.predict_btn.setEnabled(True)
            
            # 保存ボタンは予測完了まで無効
            self.save_btn.setEnabled(False)
            self.review_btn.setEnabled(False)
    
    def run_prediction(self):
        """予測を実行"""
        if not self.current_dicom_folder:
            return
        
        # UIを無効化
        self.predict_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.review_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            # DICOMをNIfTIに変換
            sitk_image, folder_name = convert_dicom_folder_to_nifti(self.current_dicom_folder)
            
            # 一時ディレクトリを作成（インスタンス変数として保持）
            self._temp_dir = tempfile.TemporaryDirectory()
            nifti_path = save_temp_nifti(sitk_image, self._temp_dir.name)
            
            # 予測スレッドを開始
            self.prediction_thread = PredictionThread(nifti_path, self.predictor)
            self.prediction_thread.finished.connect(self.on_prediction_finished)
            self.prediction_thread.error.connect(self.on_prediction_error)
            self.prediction_thread.start()
                
        except Exception as e:
            self._cleanup_temp_dir()
            self.on_prediction_error(str(e))
    
    def _cleanup_temp_dir(self):
        """一時ディレクトリをクリーンアップ"""
        if hasattr(self, '_temp_dir') and self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None
    
    def on_prediction_finished(self, visualized_slices, slice_areas, volumes, image_array, pred_array, spacing):
        """予測完了時の処理"""
        # 一時ディレクトリをクリーンアップ
        self._cleanup_temp_dir()
        
        self.visualized_slices = visualized_slices
        self.slice_areas = slice_areas
        self.volumes = volumes
        
        self.image_array = image_array
        self.pred_array = pred_array
        self.spacing = spacing
        
        # スライダーを設定
        num_slices = len(visualized_slices)
        self.slice_slider.setRange(0, num_slices - 1)
        self.slice_slider.setValue(num_slices // 2)
        self.slice_slider.setEnabled(True)
        
        self.current_slice_idx = num_slices // 2
        self.update_display()
        
        self.update_volume_tree(volumes)
        
        # UIを有効化
        self.predict_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.review_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "完了", 
            f"予測が完了しました\n総スライス数: {num_slices}")
    
    def on_prediction_error(self, error_msg):
        """予測エラー時の処理"""
        # 一時ディレクトリをクリーンアップ
        self._cleanup_temp_dir()
        
        QMessageBox.critical(self, "エラー", f"予測に失敗しました:\n{error_msg}")
        
        self.predict_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def save_results(self):
        """結果を保存"""
        if not self.current_folder_name or not self.volumes:
            return
        
        if self.pred_array is None or self.image_array is None:
            QMessageBox.warning(self, "警告", "予測データがありません")
            return
        
        try:
            # NIfTI形式で予測マスクを保存
            nifti_path = self.result_manager.save_prediction_nifti(
                self.current_folder_name,
                self.pred_array,
                self.spacing
            )
            
            # CSV追記（1行 = 1検査）
            csv_path = self.result_manager.append_to_csv(
                self.current_folder_name,
                self.volumes
            )
            
            QMessageBox.information(self, "保存完了", 
                f"結果を保存しました\n\n"
                f"予測マスク: {nifti_path.name}\n"
                f"CSV: {csv_path.name}")
            
            # ボタンを無効化（同じデータを重複保存しない）
            self.save_btn.setEnabled(False)
            self.review_btn.setEnabled(False)
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "エラー", f"保存に失敗しました:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def mark_for_review(self):
        """手動レビューリストに追加"""
        if not self.current_folder_name:
            return
        
        try:
            review_path = self.result_manager.add_to_review_list(
                self.current_folder_name,
                "ユーザーが手動確認を選択"
            )
            
            self.update_review_count()
            
            QMessageBox.information(self, "レビュー追加", 
                f"手動レビューリストに追加しました\n\n"
                f"フォルダ: {self.current_folder_name}")
            
            # ボタンを無効化
            self.save_btn.setEnabled(False)
            self.review_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"追加に失敗しました:\n{str(e)}")
    
    def update_review_count(self):
        """レビュー待ち件数を更新"""
        count = self.result_manager.get_pending_review_count()
        if count > 0:
            self.review_count_label.setText(f"レビュー待ち: {count}件")
        else:
            self.review_count_label.setText("")
    
    def on_slice_changed(self, value):
        """スライダーの値が変更されたとき"""
        self.current_slice_idx = value
        self.update_display()
    
    def on_show_labels_changed(self, state):
        """ラベル表示切り替え時の処理"""
        self.show_labels = (state == Qt.Checked)
        
        if self.image_array is not None and self.predictor is not None:
            self.redraw_current_slice()
    
    def on_zoom_changed(self, value):
        """ズームスライダー変更時の処理"""
        self.zoom_level = value
        self.zoom_label.setText(f"{value}%")
        
        if self.image_array is not None and self.predictor is not None:
            self.redraw_current_slice()
        elif self.visualized_slices is not None:
            img = self.visualized_slices[self.current_slice_idx]
            self.display_image(img)
    
    def redraw_current_slice(self):
        """現在のスライスをラベル設定に応じて再描画"""
        if self.image_array is None or self.pred_array is None:
            return
        
        img_slice = self.image_array[self.current_slice_idx]
        pred_slice = self.pred_array[self.current_slice_idx]
        
        vis_img, _ = self.predictor.visualize_slice(
            img_slice, pred_slice, self.spacing, show_labels=self.show_labels
        )
        self.display_image(vis_img)
    
    def on_image_wheel_scrolled(self, delta):
        """画像上でのマウスホイールスクロール時の処理"""
        if self.visualized_slices is None:
            return
        
        if delta > 0:
            new_idx = min(self.current_slice_idx + 1, len(self.visualized_slices) - 1)
        else:
            new_idx = max(self.current_slice_idx - 1, 0)
        
        if new_idx != self.current_slice_idx:
            self.current_slice_idx = new_idx
            self.slice_slider.setValue(new_idx)
            self.update_display()
    
    def update_display(self):
        """画像と面積テーブルを更新"""
        if self.visualized_slices is None:
            return
        
        total_slices = len(self.visualized_slices)
        self.slice_info_label.setText(
            f"スライス: {self.current_slice_idx + 1} / {total_slices}"
        )
        
        if self.image_array is not None and self.predictor is not None:
            self.redraw_current_slice()
        else:
            img = self.visualized_slices[self.current_slice_idx]
            self.display_image(img)
        
        areas = self.slice_areas.get(self.current_slice_idx, {})
        self.update_area_tree(areas)
    
    def display_image(self, img):
        """画像を表示"""
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_img)
        base_size = 700
        target_size = int(base_size * self.zoom_level / 100)
        scaled_pixmap = pixmap.scaled(
            target_size, target_size,
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def update_area_tree(self, label_areas):
        """面積ツリーを更新"""
        self.area_tree.clear()
        
        muscle_groups = {}
        for label, area in label_areas.items():
            parts = label.split('_')
            if len(parts) == 2:
                side, muscle = parts
                if muscle not in muscle_groups:
                    muscle_groups[muscle] = {}
                muscle_groups[muscle][side] = area
        
        for muscle_key, muscle_name_jp, color in self.muscle_order:
            if muscle_key not in muscle_groups:
                continue
            
            sides = muscle_groups[muscle_key]
            total_area = sum(sides.values())
            
            parent = QTreeWidgetItem([f"■ {muscle_key} ({muscle_name_jp})", f"{total_area:.2f}"])
            parent.setForeground(0, self._get_brush_from_color(color))
            
            for side in ['l', 'r']:
                if side in sides:
                    side_name = "左 (L)" if side == 'l' else "右 (R)"
                    child = QTreeWidgetItem([f"    {side_name}", f"{sides[side]:.2f}"])
                    parent.addChild(child)
            
            self.area_tree.addTopLevelItem(parent)
        
        self.area_tree.resizeColumnToContents(0)
    
    def update_volume_tree(self, volumes):
        """体積ツリーを更新"""
        self.volume_tree.clear()
        
        muscle_groups = {}
        for label, volume in volumes.items():
            parts = label.split('_')
            if len(parts) == 2:
                side, muscle = parts
                if muscle not in muscle_groups:
                    muscle_groups[muscle] = {}
                muscle_groups[muscle][side] = volume
        
        for muscle_key, muscle_name_jp, color in self.muscle_order:
            if muscle_key not in muscle_groups:
                continue
            
            sides = muscle_groups[muscle_key]
            total_volume = sum(sides.values())
            
            parent = QTreeWidgetItem([f"■ {muscle_key} ({muscle_name_jp})", f"{total_volume:.2f}"])
            parent.setForeground(0, self._get_brush_from_color(color))
            
            for side in ['l', 'r']:
                if side in sides:
                    side_name = "左 (L)" if side == 'l' else "右 (R)"
                    child = QTreeWidgetItem([f"    {side_name}", f"{sides[side]:.2f}"])
                    parent.addChild(child)
            
            self.volume_tree.addTopLevelItem(parent)
        
        self.volume_tree.resizeColumnToContents(0)
    
    def _get_brush_from_color(self, hex_color):
        """16進カラーコードからQBrushを生成"""
        from PyQt5.QtGui import QBrush, QColor
        return QBrush(QColor(hex_color))


def load_env_file():
    """Project rootの.envファイルを読み込む"""
    try:
        env_path = Path(__file__).resolve().parent.parent.parent / '.env'
        
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value
            return True
    except Exception as e:
        print(f"Warning: Failed to load .env file: {e}")
    return False


def main():
    load_env_file()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MuscleSegmentationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()