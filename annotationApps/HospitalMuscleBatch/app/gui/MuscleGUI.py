"""
病院用筋肉セグメンテーションGUI - バージョン1
機能: DICOMフォルダ選択、NIfTI変換、予測、結果保存/レビュー
"""

import sys
import os
from pathlib import Path

# ポータブル環境用: DLL検索パスを設定（torch, PyQt5のDLLが見つかるように）
# これはimport torchより前に実行する必要がある
if sys.platform == 'win32':
    _base = os.path.dirname(os.path.abspath(__file__))
    _python_dir = os.path.normpath(os.path.join(_base, '..', '..', 'python311'))
    _dll_dirs = [
        os.path.join(_python_dir, 'Lib', 'site-packages', 'torch', 'lib'),
        os.path.join(_python_dir, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'bin'),
        _python_dir,
    ]
    for _d in _dll_dirs:
        if os.path.isdir(_d):
            os.add_dll_directory(_d)
            if _d not in os.environ.get('PATH', ''):
                os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QSpinBox, QComboBox, QMessageBox, QProgressBar, QSlider,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QFrame, QProgressDialog, QLineEdit,
    QListWidget, QListWidgetItem, QScrollArea, QTabWidget,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
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
    # 引数: visualized_slices, slice_areas, volumes_data, image_array, pred_array, spacing, used_device
    finished = pyqtSignal(object, object, object, object, object, object, str)
    error = pyqtSignal(str)

    def __init__(self, nifti_path, predictor):
        super().__init__()
        self.nifti_path = nifti_path
        self.predictor = predictor

    def run(self):
        try:
            result = self.predictor.predict_from_nifti(self.nifti_path)
            img_slices, slice_areas, volumes_data, image_array, pred_array, spacing, used_device = result
            self.finished.emit(img_slices, slice_areas, volumes_data, image_array, pred_array, spacing, used_device)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class LoadExistingThread(QThread):
    """保存済み予測NIfTIを再読み込みするスレッド（推論スキップ）"""
    finished = pyqtSignal(object, object, object, object, object, object, str)
    error = pyqtSignal(str)

    def __init__(self, image_nifti_path, pred_nifti_path, predictor):
        super().__init__()
        self.image_nifti_path = image_nifti_path
        self.pred_nifti_path = pred_nifti_path
        self.predictor = predictor

    def run(self):
        try:
            result = self.predictor.load_from_existing_prediction(
                self.image_nifti_path, self.pred_nifti_path
            )
            vis_slices, slice_areas, volumes_data, image_array, pred_array, spacing = result
            self.finished.emit(vis_slices, slice_areas, volumes_data, image_array, pred_array, spacing, "cached")
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class NnUNetPredictor:
    """nnU-Net推論クラス（固定設定版）"""
    
    def __init__(self):
        # 固定設定
        self.task_id = 119
        self.configuration = "2d"
        self.fold = 0
        self.checkpoint = "checkpoint_best.pth"
        self.dataset_name = f"Dataset{self.task_id:03d}_EyeMuscleSegmentation"

        # デバイス設定（GPU自動検出）
        self.device = self._detect_device()
        
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
    
    def _detect_device(self):
        """利用可能なデバイスを検出（torchインポートなし、DLL競合回避）

        torch_cuda.dll の存在でCUDA対応ビルドかどうかを判定する。
        NVIDIAドライバの有無は subprocess で nvidia-smi を呼んで確認する。
        """
        base = os.path.dirname(sys.executable)
        cuda_dll = os.path.join(base, 'Lib', 'site-packages', 'torch', 'lib', 'torch_cuda.dll')
        if not os.path.exists(cuda_dll):
            print("torch_cuda.dll が見つかりません。CPUモードを使用します。")
            return "cpu"

        # NVIDIAドライバが使えるか確認
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().splitlines()[0]
                print(f"GPU検出: {gpu_name} → cudaモードを使用します。")
                return "cuda"
        except Exception:
            pass

        print("NVIDIAドライバが見つかりません。CPUモードを使用します。")
        return "cpu"

    def run_nnunet_inference(self, input_dir, output_dir):
        """nnU-Net推論を実行（サブプロセス経由 - PyQt5とPyTorchのDLL競合回避）"""
        import subprocess
        
        # nnUNet_resultsからモデルフォルダパスを構築
        nnunet_results = os.environ.get('nnUNet_results')
        if not nnunet_results:
            raise RuntimeError("nnUNet_results環境変数が設定されていません")
        
        # モデルフォルダパス
        model_folder = os.path.join(
            nnunet_results,
            f"Dataset{self.task_id:03d}_EyeMuscleSegmentation",
            f"nnUNetTrainer__nnUNetPlans__{self.configuration}"
        )
        
        if not os.path.exists(model_folder):
            raise RuntimeError(
                f"モデルフォルダが見つかりません: {model_folder}\n"
                f"nnUNet_results: {nnunet_results}"
            )
        
        # run_nnunet.py のパス
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_nnunet.py')
        
        # predict_entry_point_modelfolder() 用のコマンド引数
        cmd = [
            sys.executable,
            script_path,
            "-i", input_dir,
            "-o", output_dir,
            "-m", model_folder,
            "-f", str(self.fold),
            "-chk", self.checkpoint,
            "-device", self.device,
            "--disable_tta",
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        print(f"モデルフォルダ: {model_folder}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            print(f"推論成功: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = f"nnUNetコマンドエラー:\n"
            error_msg += f"コマンド: {' '.join(cmd)}\n"
            error_msg += f"終了コード: {e.returncode}\n"
            error_msg += f"\n=== 標準エラー出力 ===\n"
            error_msg += f"{e.stderr if e.stderr else '(なし)'}\n"
            error_msg += f"\n=== 標準出力 ===\n"
            error_msg += f"{e.stdout if e.stdout else '(なし)'}"
            print(error_msg)
            raise RuntimeError(error_msg)


    
    def _compute_volumes(self, slice_areas, slice_thickness_cm, slice_indices=None):
        """指定スライスインデックスで各筋肉ラベルの体積(cm³)を計算する。

        Args:
            slice_areas: {slice_idx: {label: area_cm2}}
            slice_thickness_cm: スライス厚(cm)
            slice_indices: 使用するスライスのインデックスリスト。Noneなら全スライス。

        Returns:
            {label: volume_cm3}  体積0のラベルは含まない
        """
        if slice_indices is None:
            slice_indices = sorted(slice_areas.keys())

        volumes = {}
        for label_name in self.label_names.values():
            for side in ['l', 'r']:
                side_label = f"{side}_{label_name}"
                total = sum(
                    slice_areas[i][side_label]
                    for i in slice_indices
                    if side_label in slice_areas.get(i, {})
                )
                if total > 0:
                    volumes[side_label] = total * slice_thickness_cm
        return volumes

    def _get_valid_slices_dynamic(self, slice_areas):
        """so/sr/mr/lr/ir が両側(l・r)すべて検出されているスライスのインデックスを返す。

        io は範囲決定の条件に含めない（ioが無くても範囲を縮めない）。
        """
        required = [
            f"{side}_{name}"
            for name in ['so', 'sr', 'mr', 'lr', 'ir']
            for side in ['l', 'r']
        ]
        return sorted(
            idx for idx, areas in slice_areas.items()
            if all(label in areas and areas[label] > 0 for label in required)
        )

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
                
                # 放射線医学的表示規則: 画像左側 = 患者右(r)、画像右側 = 患者左(l)
                side = "r" if center_x < img_width // 2 else "l"
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
        """NIfTIファイルから予測を実行。CUDA失敗時はCPUで自動リトライ。

        Returns:
            tuple: (visualized_slices, slice_areas, volumes, image_array, pred_array, spacing, used_device)
        """
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            input_file = input_dir / "case_0000.nii.gz"
            shutil.copy(nifti_path, input_file)

            print(f"入力ファイル: {input_file}")
            print(f"出力ディレクトリ: {output_dir}")
            print(f"使用デバイス: {self.device}")

            # 推論実行（CUDA失敗時はCPUで自動リトライ）
            used_device = self.device
            try:
                self.run_nnunet_inference(str(input_dir), str(output_dir))
            except RuntimeError as e:
                if self.device == "cuda":
                    print(f"CUDAで推論に失敗しました。CPUで再試行します。\n原因: {e}")
                    # 以降の推論もCPUを使う
                    self.device = "cpu"
                    used_device = "cpu"
                    # 出力ディレクトリをクリアしてリトライ
                    for f in output_dir.iterdir():
                        f.unlink()
                    try:
                        self.run_nnunet_inference(str(input_dir), str(output_dir))
                    except RuntimeError as e2:
                        raise RuntimeError(f"nnU-Net推論に失敗しました（CPU再試行も失敗）。\n\n詳細:\n{str(e2)}")
                else:
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
            
            # 体積計算（3種類）
            slice_thickness_cm = spacing[2] / 10  # mm → cm

            # ① 全スライス（GUI表示・従来互換）
            volumes_all = self._compute_volumes(slice_areas, slice_thickness_cm)

            # ② 動的範囲: so/sr/mr/lr/ir が両側すべて検出されているスライスのみ
            dyn_indices = self._get_valid_slices_dynamic(slice_areas)
            volumes_dyn = self._compute_volumes(slice_areas, slice_thickness_cm, dyn_indices)
            if dyn_indices:
                dyn_range_str = f"{min(dyn_indices)+1}-{max(dyn_indices)+1}"
            else:
                dyn_range_str = ""

            # ③ 固定範囲: スライス 5〜11（0始まりインデックス）
            fix_indices = [i for i in range(5, 12) if i in slice_areas]
            volumes_fix = self._compute_volumes(slice_areas, slice_thickness_cm, fix_indices)

            volumes_data = {
                "all":       volumes_all,
                "dyn":       volumes_dyn,
                "fix":       volumes_fix,
                "dyn_range": dyn_range_str,
            }

            return visualized_slices, slice_areas, volumes_data, image_array, pred_array, spacing, used_device

    def load_from_existing_prediction(self, image_nifti_path, pred_nifti_path):
        """保存済み予測NIfTIを使って可視化・体積計算のみ実行（推論スキップ）。

        Args:
            image_nifti_path: 元画像の一時NIfTIパス
            pred_nifti_path: 保存済み予測NIfTIパス

        Returns:
            predict_from_nifti と同じ形式のタプル（used_device は "cached"）
        """
        image_sitk = sitk.ReadImage(str(image_nifti_path))
        image_array = sitk.GetArrayFromImage(image_sitk)
        spacing = image_sitk.GetSpacing()

        pred_sitk = sitk.ReadImage(str(pred_nifti_path))
        pred_array = sitk.GetArrayFromImage(pred_sitk)

        num_slices = image_array.shape[0]
        visualized_slices = []
        slice_areas = {}
        for i in range(num_slices):
            vis_img, areas = self.visualize_slice(image_array[i], pred_array[i], spacing)
            visualized_slices.append(vis_img)
            slice_areas[i] = areas

        slice_thickness_cm = spacing[2] / 10
        volumes_all = self._compute_volumes(slice_areas, slice_thickness_cm)
        dyn_indices = self._get_valid_slices_dynamic(slice_areas)
        volumes_dyn = self._compute_volumes(slice_areas, slice_thickness_cm, dyn_indices)
        fix_indices = [i for i in range(5, 12) if i in slice_areas]
        volumes_fix = self._compute_volumes(slice_areas, slice_thickness_cm, fix_indices)
        dyn_range_str = f"{min(dyn_indices)+1}-{max(dyn_indices)+1}" if dyn_indices else ""

        volumes_data = {
            "all": volumes_all, "dyn": volumes_dyn,
            "fix": volumes_fix, "dyn_range": dyn_range_str,
        }
        return visualized_slices, slice_areas, volumes_data, image_array, pred_array, spacing


class NoScrollTabWidget(QTabWidget):
    """ホイールスクロールでタブが切り替わらないQTabWidget"""
    def wheelEvent(self, event):
        event.ignore()


class MuscleSegmentationGUI(QMainWindow):
    """筋肉セグメンテーションGUIメインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.current_dicom_folder = None
        self.current_folder_name = None
        self.predictor = NnUNetPredictor()
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
        self.zoom_level = 130

        # バッチ完了後のレビュー管理
        self.completed_results = []   # [{id, image_array, pred_array, spacing, slice_areas, volumes, status}]
        self.current_review_index = -1  # レビュー中の患者インデックス

        # 結果管理（exe化対応）
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).resolve().parent.parent
        self.result_manager = ResultManager(app_dir, model_id=self.predictor.task_id)

        self.init_ui()
    
    def init_ui(self):
        """UIを初期化"""
        self.setWindowTitle("筋肉セグメンテーション GUI v2 (バッチ処理版)")
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
        panel.setMinimumWidth(380)
        panel.setMaximumWidth(420)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        panel.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        inner.setLayout(layout)
        scroll.setWidget(inner)

        # ── バッチ処理設定 ──────────────────────────────
        batch_group = QGroupBox("バッチ処理設定")
        batch_layout = QVBoxLayout()
        batch_layout.setSpacing(6)

        self.select_folder_btn = QPushButton("📁 親フォルダを選択")
        self.select_folder_btn.clicked.connect(self.select_root_folder)
        self.select_folder_btn.setStyleSheet("padding: 8px;")
        batch_layout.addWidget(self.select_folder_btn)

        self.folder_label = QLabel("フォルダ: 未選択")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #666; font-size: 11px;")
        batch_layout.addWidget(self.folder_label)

        self.batch_info_label = QLabel("検出されたシリーズ: 0 件")
        self.batch_info_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        batch_layout.addWidget(self.batch_info_label)

        # シリーズフィルタ
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("対象シリーズ:"))
        self.series_filter_input = QLineEdit("eT1W_SE_cor")
        filter_layout.addWidget(self.series_filter_input)
        batch_layout.addLayout(filter_layout)

        # デバイス選択
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("推論デバイス:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda (GPU)", "cpu"])
        current_device = self.predictor.device
        self.device_combo.setCurrentIndex(0 if current_device == "cuda" else 1)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_layout.addWidget(self.device_combo)
        self.device_status_label = QLabel(f"[{current_device.upper()}]")
        self.device_status_label.setStyleSheet(
            "color: #2e7d32; font-weight: bold;" if current_device == "cuda"
            else "color: #e65100; font-weight: bold;"
        )
        device_layout.addWidget(self.device_status_label)
        batch_layout.addLayout(device_layout)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # ── 処理実行コントロール ──────────────────────────
        self.predict_btn = QPushButton("一括処理を開始")
        self.predict_btn.clicked.connect(self.start_batch_processing)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet("""
            QPushButton {
                padding: 10px; font-size: 14px; font-weight: bold;
                background-color: #1565c0; color: white;
                border: none; border-radius: 6px;
            }
            QPushButton:hover { background-color: #1976d2; }
            QPushButton:pressed { background-color: #0d47a1; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
        """)
        layout.addWidget(self.predict_btn)

        self.status_label = QLabel("待機中")
        self.status_label.setStyleSheet("font-size: 12px; color: #555;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m 件")
        layout.addWidget(self.progress_bar)

        # ── 患者リスト（バッチ完了後に表示） ────────────
        self.review_panel = QWidget()
        self.review_panel.setVisible(False)
        review_layout = QVBoxLayout()
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.setSpacing(4)
        self.review_panel.setLayout(review_layout)

        review_header = QLabel("患者リスト")
        review_header.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #212121; padding: 4px 0;"
        )
        review_layout.addWidget(review_header)

        self.review_summary_label = QLabel("")
        self.review_summary_label.setStyleSheet("font-size: 11px; color: #666;")
        review_layout.addWidget(self.review_summary_label)

        self.patient_list = QListWidget()
        self.patient_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #ffffff;
                outline: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 7px 10px;
                border-bottom: 1px solid #f5f5f5;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #0d47a1;
            }
            QListWidget::item:hover:!selected {
                background-color: #fafafa;
            }
        """)
        self.patient_list.setMinimumHeight(180)
        self.patient_list.currentRowChanged.connect(self.switch_to_patient)
        review_layout.addWidget(self.patient_list)

        layout.addWidget(self.review_panel)

        # ── 表示設定 ──────────────────────────────────
        display_group = QGroupBox("表示設定")
        display_layout = QVBoxLayout()
        self.show_labels_checkbox = QCheckBox("画像上にラベル名を表示")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.stateChanged.connect(self.on_show_labels_changed)
        display_layout.addWidget(self.show_labels_checkbox)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # ── 筋肉の順序・色定義 ─────────────────────────
        self.muscle_order = [
            ("so", "上斜筋", "#FF0000"),
            ("io", "下斜筋", "#00FF00"),
            ("sr", "上直筋", "#0000FF"),
            ("ir", "下直筋", "#FFFF00"),
            ("lr", "外直筋", "#FF00FF"),
            ("mr", "内直筋", "#00FFFF"),
        ]

        # ── 体積ツリー（タブ切替）─────────────────────
        volume_group = QGroupBox("筋肉の体積 (cm³)")
        volume_layout = QVBoxLayout()
        self.volume_tabs = NoScrollTabWidget()

        # タブ0: 全スライス
        tab_all = QWidget()
        tab_all_layout = QVBoxLayout()
        tab_all_layout.setContentsMargins(0, 4, 0, 0)
        self.volume_tree_all = QTreeWidget()
        self.volume_tree_all.setHeaderLabels(["筋肉", "体積 (cm³)"])
        self.volume_tree_all.setColumnCount(2)
        self.volume_tree_all.header().setStretchLastSection(True)
        self.volume_tree_all.setIndentation(16)
        tab_all_layout.addWidget(self.volume_tree_all)
        tab_all.setLayout(tab_all_layout)
        self.volume_tabs.addTab(tab_all, "全スライス")

        # タブ1: 動的範囲
        tab_dyn = QWidget()
        tab_dyn_layout = QVBoxLayout()
        tab_dyn_layout.setContentsMargins(0, 4, 0, 0)
        self.dyn_range_label = QLabel("範囲: —")
        self.dyn_range_label.setStyleSheet("font-size: 11px; color: #555;")
        tab_dyn_layout.addWidget(self.dyn_range_label)
        self.volume_tree_dyn = QTreeWidget()
        self.volume_tree_dyn.setHeaderLabels(["筋肉", "体積 (cm³)"])
        self.volume_tree_dyn.setColumnCount(2)
        self.volume_tree_dyn.header().setStretchLastSection(True)
        self.volume_tree_dyn.setIndentation(16)
        tab_dyn_layout.addWidget(self.volume_tree_dyn)
        tab_dyn.setLayout(tab_dyn_layout)
        self.volume_tabs.addTab(tab_dyn, "動的範囲")

        # タブ2: 固定範囲
        tab_fix = QWidget()
        tab_fix_layout = QVBoxLayout()
        tab_fix_layout.setContentsMargins(0, 4, 0, 0)
        self.fix_range_label = QLabel("範囲: 5〜11")
        self.fix_range_label.setStyleSheet("font-size: 11px; color: #555;")
        tab_fix_layout.addWidget(self.fix_range_label)
        self.volume_tree_fix = QTreeWidget()
        self.volume_tree_fix.setHeaderLabels(["筋肉", "体積 (cm³)"])
        self.volume_tree_fix.setColumnCount(2)
        self.volume_tree_fix.header().setStretchLastSection(True)
        self.volume_tree_fix.setIndentation(16)
        tab_fix_layout.addWidget(self.volume_tree_fix)
        tab_fix.setLayout(tab_fix_layout)
        self.volume_tabs.addTab(tab_fix, "固定範囲")

        volume_layout.addWidget(self.volume_tabs)
        volume_group.setLayout(volume_layout)
        layout.addWidget(volume_group)

        # ── 面積ツリー ─────────────────────────────────
        area_group = QGroupBox("現在のスライスの面積 (cm²)")
        area_layout = QVBoxLayout()
        self.area_tree = QTreeWidget()
        self.area_tree.setHeaderLabels(["筋肉", "面積 (cm²)"])
        self.area_tree.setColumnCount(2)
        self.area_tree.header().setStretchLastSection(True)
        self.area_tree.setIndentation(16)
        area_layout.addWidget(self.area_tree)
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)

        # ── 操作一覧ボタン ──────────────────────────────
        help_btn = QPushButton("❓ 操作一覧")
        help_btn.setStyleSheet(
            "background-color: #424242; color: white; font-size: 12px; padding: 6px;"
        )
        help_btn.clicked.connect(self.show_help_dialog)
        layout.addWidget(help_btn)

        layout.addStretch()
        return panel


    def create_image_panel(self):
        """画像表示パネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        panel.setLayout(layout)

        # ── レビューアクションバー（バッチ完了後に表示） ──
        self.review_action_bar = QFrame()
        self.review_action_bar.setFrameShape(QFrame.NoFrame)
        self.review_action_bar.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-bottom: 2px solid #dee2e6;
                border-radius: 6px;
            }
        """)
        self.review_action_bar.setVisible(False)

        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(14, 10, 14, 10)
        bar_layout.setSpacing(10)

        self.review_patient_label = QLabel("—")
        self.review_patient_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #212121;"
        )
        bar_layout.addWidget(self.review_patient_label, 1)

        self.slice_info_label = QLabel("スライス: - / -")
        self.slice_info_label.setStyleSheet("font-size: 12px; color: #666;")
        bar_layout.addWidget(self.slice_info_label)

        # ✓ 確認ボタン
        self.confirm_btn = QPushButton("✓  確認")
        self.confirm_btn.setFixedHeight(38)
        self.confirm_btn.setMinimumWidth(110)
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; color: white;
                border: none; border-radius: 6px;
                font-size: 14px; font-weight: bold; padding: 0 20px;
            }
            QPushButton:hover   { background-color: #388e3c; }
            QPushButton:pressed { background-color: #1b5e20; }
            QPushButton:disabled { background-color: #c8e6c9; color: #a5d6a7; }
        """)
        self.confirm_btn.clicked.connect(self.mark_confirmed)
        bar_layout.addWidget(self.confirm_btn)

        # ✎ 修正ボタン
        self.correction_btn = QPushButton("✎  修正")
        self.correction_btn.setFixedHeight(38)
        self.correction_btn.setMinimumWidth(110)
        self.correction_btn.setStyleSheet("""
            QPushButton {
                background-color: #e65100; color: white;
                border: none; border-radius: 6px;
                font-size: 14px; font-weight: bold; padding: 0 20px;
            }
            QPushButton:hover   { background-color: #ef6c00; }
            QPushButton:pressed { background-color: #bf360c; }
            QPushButton:disabled { background-color: #ffe0b2; color: #ffcc80; }
        """)
        self.correction_btn.clicked.connect(self.mark_needs_correction)
        bar_layout.addWidget(self.correction_btn)

        self.review_action_bar.setLayout(bar_layout)
        layout.addWidget(self.review_action_bar)

        # ── ズームスライダー ──────────────────────────
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("ズーム:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(120, 200)
        self.zoom_slider.setValue(130)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("130%")
        self.zoom_label.setMinimumWidth(40)
        zoom_layout.addWidget(self.zoom_label)
        layout.addLayout(zoom_layout)

        # ── 画像表示 ──────────────────────────────────
        self.image_label = ScrollableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #1a1a1a;")
        self.image_label.setMinimumSize(700, 700)
        self.image_label.wheel_scrolled.connect(self.on_image_wheel_scrolled)
        layout.addWidget(self.image_label)

        # ── スライススライダー ─────────────────────────
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("スライス:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slider_layout.addWidget(self.slice_slider)
        layout.addLayout(slider_layout)

        return panel

    def select_root_folder(self):
        """ルートフォルダを選択し、Data.txtを探して解析"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Data.txtを含む親フォルダを選択", ""
        )
        
        if not folder_path:
            return
            
        self.current_root_folder = Path(folder_path)
        self.folder_label.setText(f"フォルダ: {self.current_root_folder.name}")
        
        # Data.txtを再帰的に検索（あるいは直下のみ？要件定義によるが、まずは直下とサブフォルダを想定）
        # ユーザー要件は「data.txtが入っているフォルダを選択することとする」なので、直下にあると仮定。
        # もし見つからなければサブフォルダも探すようにする。
        
        data_txt_candidates = list(self.current_root_folder.rglob("Data.txt"))
        if not data_txt_candidates:
             QMessageBox.warning(self, "エラー", "Data.txt が見つかりませんでした。")
             return

        self.batch_queue = []
        
        for data_txt_path in data_txt_candidates:
            self.batch_queue.extend(self.parse_data_txt(data_txt_path))
            
        self.batch_info_label.setText(f"検出されたシリーズ: {len(self.batch_queue)} 件")
        
        if self.batch_queue:
            self.predict_btn.setEnabled(True)
            self.status_label.setText("待機中")
        else:
            self.predict_btn.setEnabled(False)
            self.status_label.setText("有効なシリーズが見つかりません")

    def _cleanup_temp_dir(self):
        """一時ディレクトリをクリーンアップ"""
        if hasattr(self, '_temp_dir') and self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

    def parse_data_txt(self, file_path):
        """Data.txtを解析して処理対象のパスとIDを抽出
        
        Format example:
        PatientName:(ID:147757)[20230104]
            Series No.302(MR)[eT1W_SE_tra]		Directory: DATA\147757\20240508\112048\EX1\SE1
        """
        results = []
        target_series = self.series_filter_input.text().strip()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
        except UnicodeDecodeError:
             with open(file_path, 'r', encoding='cp932') as f:
                content = f.readlines()
                
        base_dir = file_path.parent
        
        for line in content:
            line = line.strip()
            if not line:
                continue
            
            # ディレクトリ指定行を探す
            if "Directory:" in line:
                # ターゲットシリーズが含まれているか確認
                if target_series and target_series not in line:
                    continue

                # "DATA\..." の部分を抽出
                parts = line.split("Directory:")
                if len(parts) < 2:
                    continue
                
                rel_path_str = parts[1].strip()
                # Windows path separator logic
                rel_path = Path(rel_path_str)
                full_path = base_dir / rel_path
                
                if full_path.exists():
                    # ID生成: Data/ID/Date/Time/EX/SE -> ID_Date_Time_EX_SE
                    # rel_path_str usually starts with DATA\
                    path_parts = rel_path_str.replace('\\', '/').split('/')
                    # remove 'DATA' or empty strings
                    filtered_parts = [p for p in path_parts if p.upper() != 'DATA' and p]
                    
                    if len(filtered_parts) >= 1:
                        unique_id = "_".join(filtered_parts)
                        results.append({
                            "path": full_path,
                            "id": unique_id
                        })
        return results

    def start_batch_processing(self):
        """一括処理を開始"""
        if not self.batch_queue:
            return
            
        self.current_batch_index = 0
        self.total_batch_count = len(self.batch_queue)
        
        self.completed_results = []  # 前回のバッチ結果をリセット
        self.current_review_index = -1
        self.review_panel.setVisible(False)
        self.review_action_bar.setVisible(False)

        self.predict_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.total_batch_count)
        self.progress_bar.setValue(0)

        self.process_next_series()

    def process_next_series(self):
        """次のシリーズを処理（キャッシュがあれば推論スキップ）"""
        if self.current_batch_index >= self.total_batch_count:
            self.on_batch_finished()
            return

        item = self.batch_queue[self.current_batch_index]
        series_path = item["path"]
        series_id = item["id"]

        self.progress_bar.setValue(self.current_batch_index)
        self.current_dicom_folder = str(series_path)
        self.current_folder_name = series_id

        try:
            # DICOMをNIfTIに変換（キャッシュ・新規問わず画像は必要）
            sitk_image, _ = convert_dicom_folder_to_nifti(self.current_dicom_folder)
            self._temp_dir = tempfile.TemporaryDirectory()
            nifti_path = save_temp_nifti(sitk_image, self._temp_dir.name)

            if self.result_manager.is_prediction_valid(series_id):
                # キャッシュ使用: 推論スキップ
                self.status_label.setText(
                    f"キャッシュ読み込み ({self.current_batch_index + 1}/{self.total_batch_count}): {series_id}"
                )
                pred_nifti_path = str(
                    self.result_manager.predictions_dir / f"{series_id}_pred.nii.gz"
                )
                self.prediction_thread = LoadExistingThread(
                    nifti_path, pred_nifti_path, self.predictor
                )
            else:
                # 新規推論
                self.status_label.setText(
                    f"処理中 ({self.current_batch_index + 1}/{self.total_batch_count}): {series_id}"
                )
                self.prediction_thread = PredictionThread(nifti_path, self.predictor)

            self.prediction_thread.finished.connect(self.on_prediction_finished)
            self.prediction_thread.error.connect(self.on_prediction_error)
            self.prediction_thread.start()

        except Exception as e:
            print(f"Error processing {series_id}: {e}")
            self.append_error_log(series_id, str(e))
            self.current_batch_index += 1
            self.process_next_series()

    def on_prediction_finished(self, visualized_slices, slice_areas, volumes_data, image_array, pred_array, spacing, used_device):
        """予測完了時の処理（一括処理版）"""
        self._cleanup_temp_dir()

        is_cached = (used_device == "cached")

        # CUDAフォールバック検出・UI同期（キャッシュ読み込み時はスキップ）
        if not is_cached:
            cuda_fallback = (self.predictor.device == "cpu" and self.device_combo.currentIndex() == 0)
            self._sync_device_ui()
            if cuda_fallback:
                self.status_label.setText("⚠ CUDAに失敗したためCPUで処理しています（以降もCPUを使用）")

        volumes_all = volumes_data["all"]
        volumes_dyn = volumes_data["dyn"]
        volumes_fix = volumes_data["fix"]
        dyn_range   = volumes_data["dyn_range"]

        # === 保存ロジック ===
        initial_status = "pending"
        if is_cached:
            # CSVの既存行からreview_statusを復元、行がなければCSVを補完
            existing_row = self.result_manager.get_csv_row(self.current_folder_name)
            if existing_row:
                initial_status = existing_row.get("review_status", "pending")
            else:
                # NIfTIはあるがCSV行がない（CSV削除・形式変更後など）→ 追記
                try:
                    max_areas = self._calculate_max_areas_from(slice_areas)
                    self.result_manager.append_to_csv(
                        self.current_folder_name,
                        volumes_all, volumes_dyn, volumes_fix, dyn_range,
                        max_areas,
                    )
                except Exception as e:
                    print(f"CSV save error {self.current_folder_name}: {e}")
                    self.append_error_log(self.current_folder_name, f"CSV Save Error: {str(e)}")
        else:
            # NIfTI保存
            try:
                self.result_manager.save_prediction_nifti(
                    self.current_folder_name, pred_array, spacing
                )
            except Exception as e:
                print(f"NIfTI save error {self.current_folder_name}: {e}")
                self.append_error_log(self.current_folder_name, f"NIfTI Save Error: {str(e)}")

            # CSV追記（同じ行が既に存在する場合はスキップ）
            existing_row = self.result_manager.get_csv_row(self.current_folder_name)
            if existing_row is None:
                try:
                    max_areas = self._calculate_max_areas_from(slice_areas)
                    self.result_manager.append_to_csv(
                        self.current_folder_name,
                        volumes_all, volumes_dyn, volumes_fix, dyn_range,
                        max_areas,
                    )
                except Exception as e:
                    print(f"CSV save error {self.current_folder_name}: {e}")
                    self.append_error_log(self.current_folder_name, f"CSV Save Error: {str(e)}")
            else:
                initial_status = existing_row.get("review_status", "pending")

        # 完了リストに追記（レビュー用）
        self.completed_results.append({
            "id":          self.current_folder_name,
            "image_array": image_array,
            "pred_array":  pred_array,
            "spacing":     spacing,
            "slice_areas": slice_areas,
            "volumes_all": volumes_all,
            "volumes_dyn": volumes_dyn,
            "volumes_fix": volumes_fix,
            "dyn_range":   dyn_range,
            "status":      initial_status,
        })

        # バッチ中プレビュー
        num_slices = image_array.shape[0]
        self.image_array = image_array
        self.pred_array  = pred_array
        self.spacing     = spacing
        self.slice_areas = slice_areas
        self.visualized_slices = [None] * num_slices
        self.slice_slider.setRange(0, num_slices - 1)
        self.slice_slider.setValue(num_slices // 2)
        self.slice_slider.setEnabled(True)
        self.current_slice_idx = num_slices // 2
        self.update_display()
        self.update_all_volume_trees(volumes_all, volumes_dyn, volumes_fix, dyn_range)

        # 次へ（QTimer経由でイベントループに戻してからGUIを更新する）
        self.current_batch_index += 1
        self.progress_bar.setValue(self.current_batch_index)
        QTimer.singleShot(100, self.process_next_series)
        
    def on_prediction_error(self, error_msg):
        """予測エラー時の処理（一括処理版）"""
        self._cleanup_temp_dir()
        self.append_error_log(self.current_folder_name, error_msg)

        # エラーでも次へ
        self.current_batch_index += 1
        self.progress_bar.setValue(self.current_batch_index)
        QTimer.singleShot(100, self.process_next_series)
        
    def _on_device_changed(self, index):
        """デバイス選択変更時の処理"""
        self.predictor.device = "cuda" if index == 0 else "cpu"

    def _sync_device_ui(self):
        """predictor.device の状態をUIに反映（フォールバック後の同期用）"""
        device = self.predictor.device
        # コンボボックスをブロックしてシグナルの二重発火を防ぐ
        self.device_combo.blockSignals(True)
        self.device_combo.setCurrentIndex(0 if device == "cuda" else 1)
        self.device_combo.blockSignals(False)
        if device == "cuda":
            self.device_status_label.setText("[CUDA]")
            self.device_status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        else:
            self.device_status_label.setText("[CPU (フォールバック)]")
            self.device_status_label.setStyleSheet("color: #e65100; font-weight: bold;")

    def on_batch_finished(self):
        """全バッチ処理完了 → レビューモードへ移行"""
        n = len(self.completed_results)
        self.status_label.setText(f"予測完了 — {n} 件をレビューしてください")
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)

        if n == 0:
            return

        # 患者リストを構築
        self.patient_list.blockSignals(True)
        self.patient_list.clear()
        for result in self.completed_results:
            item = self._make_patient_list_item(result["id"], result["status"])
            self.patient_list.addItem(item)
        self.patient_list.blockSignals(False)

        self._update_review_summary()
        self.review_panel.setVisible(True)
        self.review_action_bar.setVisible(True)

        # 先頭の患者を表示
        self.patient_list.setCurrentRow(0)

    def switch_to_patient(self, index):
        """患者リストの選択変更時 → 表示を切り替え"""
        if index < 0 or index >= len(self.completed_results):
            return

        self.current_review_index = index
        result = self.completed_results[index]

        # データをセット
        self.image_array = result["image_array"]
        self.pred_array = result["pred_array"]
        self.spacing = result["spacing"]
        self.slice_areas = result["slice_areas"]
        volumes_all = result["volumes_all"]
        volumes_dyn = result["volumes_dyn"]
        volumes_fix = result["volumes_fix"]
        dyn_range   = result["dyn_range"]

        num_slices = self.image_array.shape[0]
        self.visualized_slices = [None] * num_slices
        self.current_slice_idx = num_slices // 2

        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, num_slices - 1)
        self.slice_slider.setValue(self.current_slice_idx)
        self.slice_slider.setEnabled(True)
        self.slice_slider.blockSignals(False)

        self.update_display()
        self.update_all_volume_trees(volumes_all, volumes_dyn, volumes_fix, dyn_range)

        # アクションバーのラベルを更新
        self.review_patient_label.setText(result["id"])
        self._update_action_buttons(result["status"])

    def mark_confirmed(self):
        """現在の患者を「確認済み」にしてリストとCSVを更新"""
        if self.current_review_index < 0:
            return
        self._set_review_status(self.current_review_index, "confirmed")
        self._advance_to_next_pending()

    def mark_needs_correction(self):
        """現在の患者を「修正」にしてリストとCSVを更新"""
        if self.current_review_index < 0:
            return
        self._set_review_status(self.current_review_index, "needs_correction")
        self._advance_to_next_pending()

    def _set_review_status(self, index, status):
        """メモリとCSV両方にステータスを書き込む"""
        result = self.completed_results[index]
        result["status"] = status
        self._refresh_patient_item(index)
        self._update_review_summary()
        self._update_action_buttons(status)
        try:
            self.result_manager.update_review_status(result["id"], status)
        except PermissionError as e:
            QMessageBox.warning(self, "CSV書き込みエラー",
                f"ステータスのCSV保存に失敗しました（メモリ上の変更は保持されています）。\n\n{e}"
            )

    def _advance_to_next_pending(self):
        """次の未判定患者へフォーカスを移す"""
        n = len(self.completed_results)
        for offset in range(1, n + 1):
            idx = (self.current_review_index + offset) % n
            if self.completed_results[idx]["status"] == "pending":
                self.patient_list.setCurrentRow(idx)
                return

    def _make_patient_list_item(self, patient_id, status):
        """ステータスに応じた QListWidgetItem を生成"""
        from PyQt5.QtGui import QColor
        symbols = {
            "pending":          ("○", QColor("#757575")),
            "confirmed":        ("✓", QColor("#2e7d32")),
            "needs_correction": ("△", QColor("#e65100")),
        }
        symbol, color = symbols.get(status, ("○", QColor("#757575")))
        wi = QListWidgetItem(f"{symbol}  {patient_id}")
        wi.setForeground(color)
        return wi

    def _refresh_patient_item(self, index):
        """リストアイテムのテキスト・色を再描画"""
        result = self.completed_results[index]
        from PyQt5.QtGui import QColor
        symbols = {
            "pending":          ("○", QColor("#757575")),
            "confirmed":        ("✓", QColor("#2e7d32")),
            "needs_correction": ("△", QColor("#e65100")),
        }
        symbol, color = symbols.get(result["status"], ("○", QColor("#757575")))
        item = self.patient_list.item(index)
        if item:
            item.setText(f"{symbol}  {result['id']}")
            item.setForeground(color)

    def _update_review_summary(self):
        """レビューサマリーラベルを更新"""
        total = len(self.completed_results)
        confirmed = sum(1 for r in self.completed_results if r["status"] == "confirmed")
        correction = sum(1 for r in self.completed_results if r["status"] == "needs_correction")
        pending = total - confirmed - correction
        self.review_summary_label.setText(
            f"計 {total} 件  ／  ✓ {confirmed} 件  ／  △ {correction} 件  ／  ○ {pending} 件"
        )

    def _update_action_buttons(self, status):
        """現在のステータスに応じてボタンの見た目を変える"""
        confirmed = (status == "confirmed")
        correction = (status == "needs_correction")
        # 既に判定済みなら少し薄く表示（再クリック可能）
        self.confirm_btn.setStyleSheet(self.confirm_btn.styleSheet())  # refresh
        self.correction_btn.setStyleSheet(self.correction_btn.styleSheet())

    def append_error_log(self, series_id, msg):
        """エラーログをファイルに記録"""
        log_path = self.result_manager.base_dir / "output" / "error_log.txt"
        with open(log_path, 'a', encoding='utf-8') as f:
            import datetime
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {series_id}: {msg}\n")
            
    def _calculate_max_areas_from(self, slice_areas):
        """全スライスから各筋肉ラベルの最大断面積(cm²)を計算する"""
        max_areas = {}
        for areas in slice_areas.values():
            for label, area in areas.items():
                if area > max_areas.get(label, 0.0):
                    max_areas[label] = area
        return max_areas
    

    
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
    
    def update_volume_tree(self, volumes, tree):
        """体積ツリーを更新"""
        tree.clear()

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

            tree.addTopLevelItem(parent)

        tree.resizeColumnToContents(0)

    def update_all_volume_trees(self, volumes_all, volumes_dyn, volumes_fix, dyn_range):
        """3つの体積ツリーと動的範囲ラベルを一括更新"""
        self.update_volume_tree(volumes_all, self.volume_tree_all)
        self.update_volume_tree(volumes_dyn, self.volume_tree_dyn)
        self.update_volume_tree(volumes_fix, self.volume_tree_fix)
        range_text = f"範囲: {dyn_range}" if dyn_range else "範囲: —"
        self.dyn_range_label.setText(range_text)
    
    def _get_brush_from_color(self, hex_color):
        """16進カラーコードからQBrushを生成"""
        from PyQt5.QtGui import QBrush, QColor
        return QBrush(QColor(hex_color))

    def show_help_dialog(self):
        """操作一覧をポップアップ表示"""
        help_text = """
        <style>
            th { text-align: left; background-color: #444; color: white; padding: 4px 8px; }
            td { padding: 3px 12px 3px 0; }
        </style>
        <table cellspacing="0" cellpadding="0">
            <tr><th colspan="2">【画像表示】</th></tr>
            <tr><td>マウスホイール</td><td>スライス切替</td></tr>
            <tr><td>スライスバー</td><td>スライス位置指定</td></tr>
            <tr><td>ズームスライダー</td><td>表示倍率調整 (120〜200%)</td></tr>

            <tr><td colspan="2" height="8"></td></tr>
            <tr><th colspan="2">【体積表示】</th></tr>
            <tr><td>全スライス タブ</td><td>全スライスを使った体積</td></tr>
            <tr><td>動的範囲 タブ</td><td>筋肉が揃っているスライスのみで計算</td></tr>
            <tr><td>固定範囲 タブ</td><td>スライス 5〜11 固定で計算</td></tr>

            <tr><td colspan="2" height="8"></td></tr>
            <tr><th colspan="2">【レビュー】</th></tr>
            <tr><td>✓ 確認</td><td>確認済みに設定 → 次の未判定へ移動</td></tr>
            <tr><td>✎ 修正</td><td>要修正に設定 → 次の未判定へ移動</td></tr>
            <tr><td>患者リスト</td><td>クリックで患者を切替</td></tr>
        </table>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("操作一覧")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.exec_()


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