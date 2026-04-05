"""
AnnotationEditorGUI.py
眼筋 MRI アノテーション編集専用 GUI

フォルダ構造（HospitalMuscleBatch と同じ規則）:
    app/images/                          元画像 {id}_0000.nii.gz
    app/output/{model_id}/predictions/   AI予測 {id}_pred.nii.gz
    app/output/{model_id}/edited/        編集済み {id}_corrected.nii.gz

ケース ID（HospitalMuscleBatch と同じ命名規則）:
    Data.txt 経由: {PatientID}_{Date}_{Time}_EX1_SE{n}
    DICOM フォルダ直接: {親フォルダ名}{SEフォルダ名}
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── ポータブル Python 用 DLL パス設定 ────────────────────
if sys.platform == "win32":
    _base = os.path.dirname(os.path.abspath(__file__))
    for _candidate in [
        # 優先: annotationApps/python311（共有）
        os.path.normpath(os.path.join(_base, "..", "..", "..", "python311")),
        # フォールバック: アプリ内 python311
        os.path.normpath(os.path.join(_base, "..", "..", "python311")),
        # レガシー: HospitalMuscleBatch/python311
        os.path.normpath(os.path.join(_base, "..", "..", "..", "HospitalMuscleBatch", "python311")),
    ]:
        for _d in [
            os.path.join(_candidate, "Lib", "site-packages", "PyQt5", "Qt5", "bin"),
            _candidate,
        ]:
            if os.path.isdir(_d):
                try:
                    os.add_dll_directory(_d)
                except Exception:
                    pass
                if _d not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSlider, QSplitter,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

import numpy as np
import SimpleITK as sitk

# ── sys.path 設定 ──────────────────────────────────────────
_GUI_DIR = Path(__file__).resolve().parent   # app/gui/
_APP_DIR = _GUI_DIR.parent                   # app/
sys.path.insert(0, str(_APP_DIR))

from managers.annotation_io import (
    load_nifti, mask_to_polygons, polygons_to_mask, save_nifti,
)
from managers.label_config import LabelConfig
from managers.session_manager import SessionManager
from managers.prediction_filter import filter_predictions, stats_summary
from managers.auto_loader import AutoLoaderThread
from gui.dicom_handler import convert_dicom_folder_to_nifti
from gui.editor_canvas import EditorCanvas


# ════════════════════════════════════════════════════════════
# nnU-Net 推論クラス（HospitalMuscleBatch から移植・体積計算除去）
# ════════════════════════════════════════════════════════════

class NnUNetPredictor:
    """nnU-Net 推論クラス（体積計算なし）"""

    def __init__(self, task_id=119):
        self.task_id = task_id
        self.configuration = "2d"
        self.fold = 0
        self.checkpoint = "checkpoint_best.pth"
        self.dataset_name = f"Dataset{task_id:03d}_EyeMuscleSegmentation"
        self.device = self._detect_device()

    def _detect_device(self):
        """利用可能デバイスを検出（torch DLL なしで判定）"""
        base = os.path.dirname(sys.executable)
        cuda_dll = os.path.join(base, "Lib", "site-packages", "torch", "lib", "torch_cuda.dll")
        if not os.path.exists(cuda_dll):
            print("[Predictor] torch_cuda.dll なし → CPU モード")
            return "cpu"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"[Predictor] GPU 検出: {result.stdout.strip().splitlines()[0]} → CUDA")
                return "cuda"
        except Exception:
            pass
        print("[Predictor] GPU ドライバなし → CPU モード")
        return "cpu"

    def _get_model_folder(self):
        nnunet_results = os.environ.get("nnUNet_results", "")
        if not nnunet_results:
            raise RuntimeError("nnUNet_results 環境変数が設定されていません")
        folder = os.path.join(
            nnunet_results,
            f"Dataset{self.task_id:03d}_EyeMuscleSegmentation",
            f"nnUNetTrainer__nnUNetPlans__{self.configuration}",
        )
        if not os.path.exists(folder):
            raise RuntimeError(f"モデルフォルダが見つかりません: {folder}")
        return folder

    def run_nnunet_inference(self, input_dir, output_dir):
        """サブプロセス経由で nnU-Net 推論を実行する"""
        model_folder = self._get_model_folder()
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_nnunet.py")
        cmd = [
            sys.executable, script_path,
            "-i", input_dir,
            "-o", output_dir,
            "-m", model_folder,
            "-f", str(self.fold),
            "-chk", self.checkpoint,
            "-device", self.device,
            "--disable_tta",
        ]
        print(f"[nnUNet] コマンド: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
            print(f"[nnUNet] 推論成功: {result.stdout[:200]}")
            return True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"nnUNet エラー (code={e.returncode}):\n{e.stderr[-1000:]}"
            )

    def predict_from_nifti(self, nifti_path):
        """NIfTI から推論を実行して結果を返す。

        Returns:
            tuple: (image_array, pred_array, spacing, used_device)
                   shape=(Z,Y,X)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "input"
            out_dir = Path(tmpdir) / "output"
            in_dir.mkdir()
            out_dir.mkdir()

            in_file = in_dir / "case_0000.nii.gz"
            shutil.copy(str(nifti_path), str(in_file))

            used_device = self.device
            try:
                self.run_nnunet_inference(str(in_dir), str(out_dir))
            except RuntimeError as e:
                if self.device == "cuda":
                    print(f"[nnUNet] CUDA 失敗 → CPU でリトライ: {e}")
                    self.device = "cpu"
                    used_device = "cpu"
                    for f in out_dir.iterdir():
                        f.unlink()
                    self.run_nnunet_inference(str(in_dir), str(out_dir))
                else:
                    raise

            pred_path = out_dir / "case.nii.gz"
            if not pred_path.exists():
                raise FileNotFoundError(
                    f"予測結果が見つかりません: {pred_path}\n"
                    f"出力: {[f.name for f in out_dir.iterdir()]}"
                )

            img_sitk = sitk.ReadImage(str(nifti_path))
            image_array = sitk.GetArrayFromImage(img_sitk)
            spacing = img_sitk.GetSpacing()

            pred_sitk = sitk.ReadImage(str(pred_path))
            pred_array = sitk.GetArrayFromImage(pred_sitk)

            return image_array, pred_array, spacing, used_device


# ════════════════════════════════════════════════════════════
# 推論スレッド
# ════════════════════════════════════════════════════════════

class PredictionThread(QThread):
    """単一ケースの推論スレッド"""
    finished = pyqtSignal(object, object, object, str)  # img, pred, spacing, device
    error = pyqtSignal(str)

    def __init__(self, nifti_path, predictor):
        super().__init__()
        self.nifti_path = nifti_path
        self.predictor = predictor

    def run(self):
        try:
            img, pred, spacing, device = self.predictor.predict_from_nifti(self.nifti_path)
            self.finished.emit(img, pred, spacing, device)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ════════════════════════════════════════════════════════════
# アプリ起動ヘルパー
# ════════════════════════════════════════════════════════════

def get_app_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent  # AnnotationEditor/


def setup_nnunet_env(app_dir: Path):
    """nnUNet_results 環境変数を設定する（既存値は上書きしない）"""
    if os.environ.get("nnUNet_results"):
        return

    # 1) 共有: annotationApps/nnUNet_results
    shared = app_dir.parent / "nnUNet_results"
    # 2) アプリ内: app/nnUNet_results
    local = app_dir / "app" / "nnUNet_results"
    # 3) レガシー: HospitalMuscleBatch/app/nnUNet_results
    sibling = app_dir.parent / "HospitalMuscleBatch" / "app" / "nnUNet_results"

    for candidate in [shared, local, sibling]:
        if candidate.exists():
            os.environ["nnUNet_results"] = str(candidate)
            os.environ.setdefault("nnUNet_raw", str(candidate.parent / "nnUNet_raw"))
            os.environ.setdefault("nnUNet_preprocessed", str(candidate.parent / "nnUNet_preprocessed"))
            print(f"[nnUNet] nnUNet_results = {candidate}")
            return

    print("[nnUNet] nnUNet_results が見つかりません。推論機能は無効です。")


# ════════════════════════════════════════════════════════════
# メインウィンドウ
# ════════════════════════════════════════════════════════════

class AnnotationEditorApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.app_dir = get_app_dir()

        # config 読み込み
        cfg_path = self.app_dir / "config.json"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        self.model_id = cfg.get("model_id", 119)
        self.series_filter = cfg.get("series_filter", "eT1W_SE_cor")
        self.filter_cfg = cfg.get("prediction_filter", {})
        self.shared_folder = cfg.get("shared_folder_path", "")
        self.auto_load_on_startup = cfg.get("auto_load_on_startup", False)
        self._auto_loader_thread = None

        # nnUNet 環境変数
        setup_nnunet_env(self.app_dir)

        # 推論器（nnUNet_results がなければ None）
        self.predictor = None
        if os.environ.get("nnUNet_results"):
            try:
                self.predictor = NnUNetPredictor(task_id=self.model_id)
            except Exception as e:
                print(f"[Predictor] 初期化失敗: {e}")

        # フォルダパス（HospitalMuscleBatch と同じ構造）
        model_out = self.app_dir / "app" / "output" / str(self.model_id)
        self.images_dir = self.app_dir / "app" / "images"
        self.predictions_dir = model_out / "predictions"
        self.edited_dir = model_out / "edited"
        for d in [self.images_dir, self.predictions_dir, self.edited_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.label_config = LabelConfig(self.app_dir / "labels.json")
        self.session = SessionManager(self.app_dir / "session.json")

        # 現在のケース状態
        self.current_case_id = None
        self.current_image_array = None   # (Z, Y, X)
        self.current_spacing = None
        self.current_slice_polygons = {}  # {slice_idx: [polygon,...]}
        self.current_num_slices = 0
        self.current_slice_idx = 0
        self._orig_pred_polygons = {}     # リセット用

        # dirty トラッキング
        self.dirty_slices = set()
        self.is_saved = True

        # ステータスフィルタ
        self.status_filter = None

        # バッチ推論用
        self._pred_thread = None
        self._batch_queue = []
        self._batch_idx = 0

        self._init_ui()

        # 起動時: images フォルダをスキャン → 必要なら共有フォルダも自動取得
        QTimer.singleShot(300, self._startup_scan)

    # ════════════════════════════════════════════════════════
    # UI 構築
    # ════════════════════════════════════════════════════════

    def _init_ui(self):
        self.setWindowTitle("AnnotationEditor - 眼筋 MRI アノテーション編集")
        self.setGeometry(50, 50, 1750, 1020)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        central.setLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([370, 1380])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

    # ── 左パネル ──────────────────────────────────────────

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(460)
        panel.setStyleSheet("background-color: #f4f6f8;")

        lo = QVBoxLayout()
        lo.setContentsMargins(10, 10, 10, 10)
        lo.setSpacing(8)
        panel.setLayout(lo)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title = QLabel("AnnotationEditor")
        title.setStyleSheet("font-size: 17px; font-weight: bold; color: #1565c0;")
        title_row.addWidget(title)
        title_row.addStretch()
        btn_help = QPushButton("?")
        btn_help.setFixedSize(26, 26)
        btn_help.setToolTip("操作マニュアルを表示")
        btn_help.setStyleSheet("""
            QPushButton {
                font-size: 13px; font-weight: bold;
                color: white; background: #1565c0;
                border-radius: 13px; border: none;
            }
            QPushButton:hover { background: #1976d2; }
        """)
        btn_help.clicked.connect(self._show_help)
        title_row.addWidget(btn_help)
        lo.addLayout(title_row)
        sub = QLabel(f"眼筋 MRI アノテーション編集  [Dataset {self.model_id}]")
        sub.setStyleSheet("font-size: 10px; color: #888; margin-bottom: 2px;")
        lo.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ddd;")
        lo.addWidget(sep)

        # ── データ追加 ──
        grp_add = QGroupBox("データ追加")
        grp_add_lo = QVBoxLayout()
        grp_add_lo.setSpacing(5)

        btn_shared = QPushButton("📡 共有フォルダから自動取得")
        btn_shared.setToolTip(
            "config.json の shared_folder_path に設定した共有フォルダを\n"
            "DICOMDIR 形式でスキャンして新規データを取り込む"
        )
        btn_shared.setStyleSheet(self._btn_style("#6a1b9a"))
        btn_shared.clicked.connect(self._start_auto_load)
        grp_add_lo.addWidget(btn_shared)

        btn_datatxt = QPushButton("📋 Data.txt からインポート")
        btn_datatxt.setToolTip(
            "Data.txt を含む親フォルダを選択し DICOM → NIfTI 変換して追加\n"
            "（HospitalMuscleBatch と同じ形式）"
        )
        btn_datatxt.setStyleSheet(self._btn_style("#1565c0"))
        btn_datatxt.clicked.connect(self._add_from_data_txt)
        grp_add_lo.addWidget(btn_datatxt)

        btn_dicom = QPushButton("🏥 DICOM フォルダを直接変換して追加")
        btn_dicom.setToolTip("IMG 番号順のシングルフレーム DICOM フォルダを選択")
        btn_dicom.setStyleSheet(self._btn_style("#37474f"))
        btn_dicom.clicked.connect(self._add_from_dicom)
        grp_add_lo.addWidget(btn_dicom)

        grp_add.setLayout(grp_add_lo)
        lo.addWidget(grp_add)

        # ── 推論 ──
        grp_pred = QGroupBox("AI 推論")
        grp_pred_lo = QVBoxLayout()
        grp_pred_lo.setSpacing(5)

        from PyQt5.QtWidgets import QCheckBox
        self.chk_auto_filter = QCheckBox("推論後に異常排除")
        self.chk_auto_filter.setChecked(False)
        self.chk_auto_filter.setStyleSheet("font-size: 10px; color: #444;")
        self.chk_auto_filter.setToolTip(
            "推論完了後に自動でフィルタを適用する\n"
            "（極小・端部・過剰個数のコンポーネントを削除）"
        )
        grp_pred_lo.addWidget(self.chk_auto_filter)

        self.btn_predict = QPushButton("▶ 一括推論（未推論: 0 件）")
        self.btn_predict.setToolTip("images フォルダの未推論ケースを全件推論する")
        self.btn_predict.setStyleSheet(self._btn_style("#e65100"))
        self.btn_predict.setEnabled(self.predictor is not None)
        self.btn_predict.clicked.connect(self._start_batch_prediction)
        grp_pred_lo.addWidget(self.btn_predict)

        if self.predictor is None:
            no_model_lbl = QLabel("⚠ nnUNet_results が見つかりません（推論無効）")
            no_model_lbl.setStyleSheet("color: #c62828; font-size: 10px;")
            no_model_lbl.setWordWrap(True)
            grp_pred_lo.addWidget(no_model_lbl)

        self.pred_status_label = QLabel("")
        self.pred_status_label.setStyleSheet("font-size: 10px; color: #555;")
        self.pred_status_label.setWordWrap(True)
        grp_pred_lo.addWidget(self.pred_status_label)

        self.pred_progress = QProgressBar()
        self.pred_progress.setVisible(False)
        self.pred_progress.setTextVisible(True)
        self.pred_progress.setFormat("%v / %m 件")
        grp_pred_lo.addWidget(self.pred_progress)

        grp_pred.setLayout(grp_pred_lo)
        lo.addWidget(grp_pred)

        # ── ステータスサマリー ──
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-size: 11px; color: #555; padding: 3px 0;")
        self.summary_label.setWordWrap(True)
        lo.addWidget(self.summary_label)

        # ── フィルタ ──
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        self.btn_filter_all = QPushButton("すべて")
        self.btn_filter_pending = QPushButton("未推論")
        self.btn_filter_predicted = QPushButton("✎編集中")
        self.btn_filter_corrected = QPushButton("✓完了")
        for btn, fval in [
            (self.btn_filter_all, None),
            (self.btn_filter_pending, "pending"),
            (self.btn_filter_predicted, "editing"),
            (self.btn_filter_corrected, "corrected"),
        ]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton { padding: 3px 7px; border: 1px solid #ccc;
                              border-radius: 10px; font-size: 10px; background: #fff; }
                QPushButton:checked { background: #1565c0; color: white; border-color: #1565c0; }
            """)
            btn.clicked.connect(lambda _, v=fval: self._set_status_filter(v))
            filter_row.addWidget(btn)
        self.btn_filter_all.setChecked(True)
        filter_row.addStretch()
        lo.addLayout(filter_row)

        # ── 検索 ──
        search_row = QHBoxLayout()
        search_row.setSpacing(4)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("ケース ID で検索...")
        self.search_box.setStyleSheet("""
            QLineEdit { border: 1px solid #ddd; border-radius: 4px; padding: 5px 8px; font-size: 12px; }
            QLineEdit:focus { border-color: #1976d2; }
        """)
        self.search_box.textChanged.connect(self._filter_case_list)
        search_row.addWidget(self.search_box)
        btn_clr = QPushButton("×")
        btn_clr.setFixedSize(28, 28)
        btn_clr.setStyleSheet("border: 1px solid #ddd; border-radius: 4px; background: #f0f0f0; font-size: 14px;")
        btn_clr.clicked.connect(self.search_box.clear)
        search_row.addWidget(btn_clr)
        lo.addLayout(search_row)

        # ── ケースリスト ──
        self.case_list = QListWidget()
        self.case_list.setStyleSheet("""
            QListWidget { border: 1px solid #ddd; border-radius: 4px;
                          background: #fff; outline: none; font-size: 11px; }
            QListWidget::item { padding: 7px 10px; border-bottom: 1px solid #f0f0f0; }
            QListWidget::item:selected { background: #e3f2fd; color: #0d47a1; }
            QListWidget::item:hover:!selected { background: #fafafa; }
        """)
        self.case_list.currentRowChanged.connect(self._on_case_row_changed)
        lo.addWidget(self.case_list, 1)

        # ── 操作ボタン ──
        op_row = QHBoxLayout()
        op_row.setSpacing(6)
        self.btn_save = QPushButton("💾 保存  Ctrl+S")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(self._btn_style("#1565c0"))
        self.btn_save.clicked.connect(self.save_current_case)
        op_row.addWidget(self.btn_save)
        self.btn_complete = QPushButton("✓ 完了にする")
        self.btn_complete.setEnabled(False)
        self.btn_complete.setStyleSheet(self._btn_style("#2e7d32"))
        self.btn_complete.clicked.connect(self._mark_complete)
        op_row.addWidget(self.btn_complete)
        lo.addLayout(op_row)

        # ── 右クリックメニュー（削除） ──
        self.case_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.case_list.customContextMenuRequested.connect(self._on_case_list_context_menu)

        return panel

    # ── 右パネル ──────────────────────────────────────────

    def _build_right_panel(self):
        panel = QWidget()
        lo = QVBoxLayout()
        lo.setContentsMargins(4, 4, 4, 4)
        lo.setSpacing(4)
        panel.setLayout(lo)

        lo.addWidget(self._build_top_bar())

        self.canvas = EditorCanvas(self.label_config, parent=self)
        self.canvas.slice_changed.connect(self._on_slice_modified)
        lo.addWidget(self.canvas, 1)

        lo.addWidget(self._build_label_palette())
        return panel

    def _build_top_bar(self):
        bar = QWidget()
        bar.setFixedHeight(54)
        bar.setStyleSheet("background: #263238; border-radius: 4px;")
        lo = QHBoxLayout()
        lo.setContentsMargins(10, 4, 10, 4)
        lo.setSpacing(8)
        bar.setLayout(lo)

        lo.addWidget(self._icon_btn("◀", self._prev_slice))

        self.slice_label = QLabel("スライス: -/-")
        self.slice_label.setStyleSheet(
            "color: white; font-size: 13px; min-width: 120px; font-weight: bold;"
        )
        self.slice_label.setAlignment(Qt.AlignCenter)
        lo.addWidget(self.slice_label)

        lo.addWidget(self._icon_btn("▶", self._next_slice))
        lo.addWidget(self._vsep())

        wc_lbl = QLabel("WC")
        wc_lbl.setStyleSheet("color: #90caf9; font-size: 11px;")
        lo.addWidget(wc_lbl)
        self.wc_slider = self._make_slider(0, 255, 128)
        self.wc_slider.valueChanged.connect(self._on_wl_changed)
        lo.addWidget(self.wc_slider)
        self.wc_val = QLabel("128")
        self.wc_val.setStyleSheet("color: #90caf9; font-size: 11px; min-width: 28px;")
        lo.addWidget(self.wc_val)

        ww_lbl = QLabel("WW")
        ww_lbl.setStyleSheet("color: #90caf9; font-size: 11px;")
        lo.addWidget(ww_lbl)
        self.ww_slider = self._make_slider(1, 512, 256)
        self.ww_slider.valueChanged.connect(self._on_wl_changed)
        lo.addWidget(self.ww_slider)
        self.ww_val = QLabel("256")
        self.ww_val.setStyleSheet("color: #90caf9; font-size: 11px; min-width: 28px;")
        lo.addWidget(self.ww_val)

        lo.addWidget(self._vsep())

        btn_cp_prev = self._text_btn("← コピー", self._copy_from_prev)
        btn_cp_prev.setToolTip("前スライスのポリゴンをコピー (Ctrl+Left)")
        lo.addWidget(btn_cp_prev)

        btn_cp_next = self._text_btn("コピー →", self._copy_from_next)
        btn_cp_next.setToolTip("次スライスのポリゴンをコピー (Ctrl+Right)")
        lo.addWidget(btn_cp_next)

        lo.addWidget(self._vsep())

        self.btn_mask_toggle = QPushButton("マスク: ON")
        self.btn_mask_toggle.setCheckable(True)
        self.btn_mask_toggle.setChecked(True)
        self.btn_mask_toggle.setFixedHeight(32)
        self.btn_mask_toggle.setStyleSheet("""
            QPushButton        { color: white; background: #2e7d32; border: none;
                                 border-radius: 4px; padding: 0 10px; font-size: 11px; }
            QPushButton:!checked { background: #546e7a; }
        """)
        self.btn_mask_toggle.clicked.connect(self._on_mask_toggle)
        lo.addWidget(self.btn_mask_toggle)

        btn_reset = self._text_btn("リセット", self._reset_current_slice, color="#ffab40")
        btn_reset.setToolTip("現在スライスを AI 予測に戻す")
        lo.addWidget(btn_reset)

        btn_filter = self._text_btn("🧹 異常排除", self._apply_filter_to_current_case, color="#80cbc4")
        btn_filter.setToolTip(
            "現在ケースの予測マスクに異常排除フィルタを適用\n"
            "（極小コンポーネント / 画像端 / ラベル過剰個数）"
        )
        lo.addWidget(btn_filter)

        lo.addStretch()

        self.btn_undo = self._text_btn("↩ Undo", lambda: self.canvas.undo())
        self.btn_undo.setEnabled(False)
        lo.addWidget(self.btn_undo)

        self.btn_redo = self._text_btn("↪ Redo", lambda: self.canvas.redo())
        self.btn_redo.setEnabled(False)
        lo.addWidget(self.btn_redo)

        return bar

    def _build_label_palette(self):
        palette = QWidget()
        palette.setFixedHeight(64)
        palette.setStyleSheet("background: #37474f; border-radius: 4px;")

        lo = QHBoxLayout()
        lo.setContentsMargins(8, 5, 8, 5)
        lo.setSpacing(5)
        palette.setLayout(lo)

        self._label_buttons = {}

        btn = self._make_label_btn(0, (180, 180, 180), "選択", "0")
        self._label_buttons[0] = btn
        lo.addWidget(btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #546e7a; margin: 4px 2px;")
        lo.addWidget(sep)

        for lid, name, display, color, key in self.label_config.get_all_labels():
            btn = self._make_label_btn(lid, color, name, key)
            btn.setToolTip(display)  # 日本語名をツールチップで表示
            self._label_buttons[lid] = btn
            lo.addWidget(btn)

        lo.addStretch()

        hint = QLabel(
            "0=選択  1-9/q/w=ラベル  右クリック=確定/透明度  "
            "Tab=マスクON/OFF  Ctrl+Z/Y=Undo/Redo  Del=削除  ←→=スライス  Ctrl+←→=隣接コピー"
        )
        hint.setStyleSheet("color: #78909c; font-size: 9px;")
        lo.addWidget(hint)

        self._label_buttons[0].setChecked(True)
        return palette

    # ── UI ヘルパー ──────────────────────────────────────

    @staticmethod
    def _btn_style(color):
        return (
            f"QPushButton {{ padding: 7px; background: {color}; color: white; "
            "border: none; border-radius: 4px; font-size: 11px; }} "
            "QPushButton:disabled { background: #bdbdbd; color: #757575; }"
        )

    @staticmethod
    def _vsep():
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #546e7a;")
        return sep

    def _icon_btn(self, text, slot):
        btn = QPushButton(text)
        btn.setFixedSize(32, 32)
        btn.setStyleSheet(
            "color: white; background: #37474f; border: none; border-radius: 4px; font-size: 14px;"
        )
        btn.clicked.connect(slot)
        return btn

    def _text_btn(self, text, slot, color="white"):
        btn = QPushButton(text)
        btn.setFixedHeight(32)
        btn.setStyleSheet(
            f"color: {color}; background: #37474f; border: none; "
            "border-radius: 4px; padding: 0 8px; font-size: 11px;"
        )
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _make_slider(lo, hi, val):
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setFixedWidth(110)
        s.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #546e7a; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; background: #90caf9;
                border-radius: 7px; margin: -5px 0;
            }
        """)
        return s

    def _make_label_btn(self, label_id, color, display, key):
        r, g, b = color
        brightness = r * 0.299 + g * 0.587 + b * 0.114
        text_col = "black" if brightness > 160 else "white"
        # ホバー: 元色を少し明るく
        rh, gh, bh = min(r + 40, 255), min(g + 40, 255), min(b + 40, 255)
        btn = QPushButton(f"{display}\n[{key}]")
        btn.setCheckable(True)
        btn.setFixedSize(64, 52)
        btn.setStyleSheet(f"""
            QPushButton {{
                color: rgb({r},{g},{b});
                background: #2e3d45;
                border: 2px solid rgb({r},{g},{b});
                border-radius: 4px; font-size: 9px; font-weight: bold;
            }}
            QPushButton:checked {{
                background: rgb({r},{g},{b}); color: {text_col};
                border: 4px solid white;
            }}
            QPushButton:hover:!checked {{
                background: rgb({r//2},{g//2},{b//2});
                color: white;
                border: 2px solid rgb({rh},{gh},{bh});
            }}
        """)
        btn.clicked.connect(lambda _c, lid=label_id: self._on_label_btn_clicked(lid))
        return btn

    # ════════════════════════════════════════════════════════
    # 起動時スキャン
    # ════════════════════════════════════════════════════════

    def _startup_scan(self):
        """images フォルダをスキャンして session に追加 → ケースリスト表示"""
        added = self._scan_images_folder()
        self._load_session_cases()
        self._update_predict_btn()
        if added:
            print(f"[Scan] {added} 件の新規ケースを追加")
        # 最後のケースを復元
        last_id = self.session.get_last_case_id()
        if last_id:
            for i in range(self.case_list.count()):
                item = self.case_list.item(i)
                if item and item.data(Qt.UserRole) == last_id:
                    self.case_list.setCurrentRow(i)
                    break
        # 自動取得
        if self.auto_load_on_startup and self.shared_folder:
            self._start_auto_load(silent=True)

    def _scan_images_folder(self):
        """images フォルダの *_0000.nii.gz を session に登録する"""
        added = 0
        for img_path in sorted(self.images_dir.glob("*_0000.nii.gz")):
            case_id = img_path.name.replace("_0000.nii.gz", "")
            pred_path = self.predictions_dir / f"{case_id}_pred.nii.gz"
            corrected_path = self.edited_dir / f"{case_id}_corrected.nii.gz"
            ok = self.session.add_case(
                case_id, img_path,
                pred_path if pred_path.exists() else None,
                corrected_path,
            )
            if ok:
                added += 1
            else:
                # 既存ケースでも pred_path が増えていたら更新
                if pred_path.exists():
                    self.session.update_case_pred_path(case_id, pred_path)
        return added

    # ════════════════════════════════════════════════════════
    # ケース管理
    # ════════════════════════════════════════════════════════

    def _load_session_cases(self):
        self.case_list.clear()
        for case in self.session.get_cases():
            self._add_case_to_list(case)
        self._update_summary()

    def _add_case_to_list(self, case):
        case_id = case["case_id"]
        status = case.get("status", "pending")

        if self.status_filter and status != self.status_filter:
            return

        icons = {"pending": "○", "editing": "✎", "corrected": "✓"}
        icon = icons.get(status, "○")
        n_edited = len(case.get("edited_slices", []))
        edited_str = f"  ({n_edited}sl)" if n_edited else ""
        last = (case.get("last_edited") or "")[:10]
        last_str = f"  {last}" if last else ""

        item = QListWidgetItem(f"{icon}  {case_id}{edited_str}{last_str}")
        item.setData(Qt.UserRole, case_id)
        if status == "corrected":
            item.setForeground(QColor("#2e7d32"))
        elif status == "editing":
            item.setForeground(QColor("#e65100"))
        self.case_list.addItem(item)

    def _update_summary(self):
        cases = self.session.get_cases()
        total = len(cases)
        corrected = sum(1 for c in cases if c.get("status") == "corrected")
        editing  = sum(1 for c in cases if c.get("status") == "editing")
        pending  = total - corrected - editing
        self.summary_label.setText(
            f"合計 {total}件  ✓完了 {corrected}  ✎編集中 {editing}  ○未推論 {pending}"
        )

    def _update_predict_btn(self):
        n = len(self.session.get_unpredicted_ids())
        self.btn_predict.setText(f"▶ 一括推論（未推論: {n} 件）")
        self.btn_predict.setEnabled(self.predictor is not None and n > 0)

    def _set_status_filter(self, fval):
        self.status_filter = fval
        for btn, v in [
            (self.btn_filter_all, None),
            (self.btn_filter_pending, "pending"),
            (self.btn_filter_predicted, "editing"),
            (self.btn_filter_corrected, "corrected"),
        ]:
            btn.setChecked(v == fval)
        self._load_session_cases()

    def _filter_case_list(self, text):
        text = text.lower()
        for i in range(self.case_list.count()):
            item = self.case_list.item(i)
            item.setHidden(text not in item.data(Qt.UserRole).lower())

    # ════════════════════════════════════════════════════════
    # データ追加
    # ════════════════════════════════════════════════════════

    def _add_from_data_txt(self):
        """Data.txt を含む親フォルダを選択して DICOM → NIfTI 変換"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Data.txt を含む親フォルダを選択"
        )
        if not folder_path:
            return

        root = Path(folder_path)
        data_txts = list(root.rglob("Data.txt"))
        if not data_txts:
            QMessageBox.warning(self, "エラー", "Data.txt が見つかりませんでした。")
            return

        queue = []
        for dtxt in data_txts:
            queue.extend(self._parse_data_txt(dtxt))

        if not queue:
            QMessageBox.information(self, "対象なし",
                f"対象シリーズが見つかりませんでした。\nシリーズフィルタ: '{self.series_filter}'")
            return

        imported = 0
        errors = []
        for item in queue:
            series_path = item["path"]
            series_id = item["id"]
            try:
                if (self.images_dir / f"{series_id}_0000.nii.gz").exists():
                    print(f"[Import] スキップ（既存）: {series_id}")
                    # セッションには追加
                    self.session.add_case(
                        series_id,
                        self.images_dir / f"{series_id}_0000.nii.gz",
                        corrected_path=self.edited_dir / f"{series_id}_corrected.nii.gz",
                    )
                    continue
                sitk_image, _ = convert_dicom_folder_to_nifti(str(series_path))
                img_path = self.images_dir / f"{series_id}_0000.nii.gz"
                sitk.WriteImage(sitk_image, str(img_path))
                self.session.add_case(
                    series_id, img_path,
                    corrected_path=self.edited_dir / f"{series_id}_corrected.nii.gz",
                )
                imported += 1
                print(f"[Import] 完了: {series_id}")
            except Exception as e:
                errors.append(f"{series_id}: {e}")
                print(f"[Import] エラー: {series_id}: {e}")

        self._load_session_cases()
        self._update_predict_btn()
        msg = f"{imported} 件をインポートしました。"
        if errors:
            msg += f"\n\nエラー ({len(errors)} 件):\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "インポート完了", msg)

    def _parse_data_txt(self, file_path):
        """Data.txt を解析して (path, id) のリストを返す（HospitalMuscleBatch 互換）"""
        results = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp932") as f:
                content = f.readlines()

        base_dir = file_path.parent
        for line in content:
            line = line.strip()
            if "Directory:" not in line:
                continue
            if self.series_filter and self.series_filter not in line:
                continue
            parts = line.split("Directory:")
            if len(parts) < 2:
                continue
            rel_path_str = parts[1].strip()
            full_path = base_dir / Path(rel_path_str)
            if not full_path.exists():
                continue
            path_parts = rel_path_str.replace("\\", "/").split("/")
            filtered = [p for p in path_parts if p.upper() != "DATA" and p]
            if filtered:
                unique_id = "_".join(filtered)
                results.append({"path": full_path, "id": unique_id})
        return results

    def _add_from_dicom(self):
        """DICOM フォルダを直接変換して追加"""
        folder = QFileDialog.getExistingDirectory(self, "DICOM フォルダを選択")
        if not folder:
            return
        try:
            sitk_image, folder_name = convert_dicom_folder_to_nifti(str(folder))
        except Exception as e:
            QMessageBox.critical(self, "変換エラー", str(e))
            return

        case_id = folder_name
        img_path = self.images_dir / f"{case_id}_0000.nii.gz"
        sitk.WriteImage(sitk_image, str(img_path))
        corrected_path = self.edited_dir / f"{case_id}_corrected.nii.gz"
        ok = self.session.add_case(case_id, img_path, corrected_path=corrected_path)

        self._load_session_cases()
        self._update_predict_btn()
        if ok:
            QMessageBox.information(self, "変換完了", f"ケース {case_id} を追加しました。")
        else:
            QMessageBox.information(self, "既存", f"ケース {case_id} は既に存在します。")

    # ════════════════════════════════════════════════════════
    # 共有フォルダ自動取得
    # ════════════════════════════════════════════════════════

    def _start_auto_load(self, silent=False):
        """共有フォルダから自動取得を開始する"""
        if not self.shared_folder:
            if not silent:
                QMessageBox.warning(
                    self, "設定なし",
                    "config.json の shared_folder_path が設定されていません。\n"
                    "共有フォルダのパスを設定してください。"
                )
            return

        if self._auto_loader_thread and self._auto_loader_thread.isRunning():
            QMessageBox.information(self, "実行中", "自動取得は既に実行中です。")
            return

        if not silent:
            self.pred_status_label.setText("共有フォルダをスキャン中...")

        self._auto_loader_thread = AutoLoaderThread(
            shared_folder   = self.shared_folder,
            series_filter   = self.series_filter,
            images_dir      = self.images_dir,
            session         = self.session,
            predictions_dir = self.predictions_dir,
            edited_dir      = self.edited_dir,
        )
        self._auto_loader_thread.progress_updated.connect(self._on_autoload_progress)
        self._auto_loader_thread.scan_completed.connect(
            lambda n, e: self._on_autoload_scan_completed(n, e, silent)
        )
        self._auto_loader_thread.error_occurred.connect(
            lambda msg: self._on_autoload_error(msg, silent)
        )
        self._auto_loader_thread.load_completed.connect(self._on_autoload_completed)
        self._auto_loader_thread.start()

    def _on_autoload_progress(self, current, total, message):
        self.pred_status_label.setText(message)
        if total > 0:
            self.pred_progress.setRange(0, total)
            self.pred_progress.setValue(current)
            self.pred_progress.setVisible(True)

    def _on_autoload_scan_completed(self, new_count, existing_count, silent):
        msg = f"スキャン完了: 新規 {new_count} 件 / 既存 {existing_count} 件"
        print(f"[AutoLoad] {msg}")
        if not silent and new_count == 0:
            self.pred_progress.setVisible(False)
            self.pred_status_label.setText(msg)

    def _on_autoload_error(self, error_msg, silent):
        self.pred_progress.setVisible(False)
        self.pred_status_label.setText("自動取得エラー")
        print(f"[AutoLoad] エラー: {error_msg}")
        if not silent:
            QMessageBox.warning(self, "自動取得エラー", error_msg)

    def _on_autoload_completed(self):
        self.pred_progress.setVisible(False)
        self._scan_images_folder()
        self._load_session_cases()
        self._update_predict_btn()
        self.pred_status_label.setText(
            f"自動取得完了: {self.session.get_cases().__len__()} 件"
        )

    def _remove_current_case(self):
        if not self.current_case_id:
            return
        reply = QMessageBox.question(
            self, "確認",
            f"ケース {self.current_case_id} をリストから削除しますか？\n（ファイルは削除されません）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.session.remove_case(self.current_case_id)
            self.current_case_id = None
            self.canvas.load_empty()
            self.btn_save.setEnabled(False)
            self.btn_complete.setEnabled(False)
            self._load_session_cases()
            self._update_predict_btn()

    def _on_case_list_context_menu(self, pos):
        from PyQt5.QtWidgets import QMenu
        item = self.case_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        act_delete = menu.addAction("🗑 リストから削除（ファイルは残す）")
        action = menu.exec_(self.case_list.viewport().mapToGlobal(pos))
        if action == act_delete:
            case_id = item.data(Qt.UserRole)
            reply = QMessageBox.question(
                self, "確認",
                f"ケース {case_id} をリストから削除しますか？\n（ファイルは削除されません）",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.session.remove_case(case_id)
                if self.current_case_id == case_id:
                    self.current_case_id = None
                    self.canvas.load_empty()
                    self.btn_save.setEnabled(False)
                    self.btn_complete.setEnabled(False)
                self._load_session_cases()
                self._update_predict_btn()

    def _mark_complete(self):
        """現在ケースを完了状態にする"""
        if not self.current_case_id:
            return
        if self.dirty_slices and not self.is_saved:
            reply = QMessageBox.question(
                self, "未保存の変更",
                "未保存の変更があります。保存してから完了にしますか？",
                QMessageBox.Save | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.save_current_case()
            else:
                return
        self.session.mark_case_complete(self.current_case_id)
        self._load_session_cases()
        # リスト選択状態を維持
        for i in range(self.case_list.count()):
            item = self.case_list.item(i)
            if item and item.data(Qt.UserRole) == self.current_case_id:
                self.case_list.blockSignals(True)
                self.case_list.setCurrentRow(i)
                self.case_list.blockSignals(False)
                break

    # ════════════════════════════════════════════════════════
    # バッチ推論
    # ════════════════════════════════════════════════════════

    def _start_batch_prediction(self):
        unpredicted = self.session.get_unpredicted_ids()
        if not unpredicted:
            QMessageBox.information(self, "情報", "未推論のケースがありません。")
            return

        self._batch_queue = unpredicted[:]
        self._batch_idx = 0
        self.pred_progress.setRange(0, len(self._batch_queue))
        self.pred_progress.setValue(0)
        self.pred_progress.setVisible(True)
        self.btn_predict.setEnabled(False)
        self._predict_next_in_batch()

    def _predict_next_in_batch(self):
        if self._batch_idx >= len(self._batch_queue):
            self._on_batch_finished()
            return

        case_id = self._batch_queue[self._batch_idx]
        self.pred_status_label.setText(
            f"推論中 ({self._batch_idx + 1}/{len(self._batch_queue)}): {case_id}"
        )
        self.pred_progress.setValue(self._batch_idx)

        img_path = self.images_dir / f"{case_id}_0000.nii.gz"
        if not img_path.exists():
            print(f"[Batch] 画像なしでスキップ: {case_id}")
            self._batch_idx += 1
            QTimer.singleShot(10, self._predict_next_in_batch)
            return

        self._pred_thread = PredictionThread(img_path, self.predictor)
        self._pred_thread.finished.connect(
            lambda img, pred, sp, dev: self._on_batch_pred_finished(case_id, img, pred, sp, dev)
        )
        self._pred_thread.error.connect(
            lambda err: self._on_batch_pred_error(case_id, err)
        )
        self._pred_thread.start()

    def _on_batch_pred_finished(self, case_id, image_array, pred_array, spacing, device):
        pred_path = self.predictions_dir / f"{case_id}_pred.nii.gz"
        img_path = self.images_dir / f"{case_id}_0000.nii.gz"
        try:
            # 自動排除フィルタ
            if self.chk_auto_filter.isChecked():
                pred_array, fstats = filter_predictions(pred_array, self.filter_cfg)
                print(f"[Filter] {case_id}: {stats_summary(fstats)}")
            save_nifti(pred_array, spacing, pred_path)
            save_nifti(image_array, spacing, img_path)
        except Exception as e:
            print(f"[Batch] 保存エラー {case_id}: {e}")

        self.session.update_case_pred_path(case_id, pred_path)
        print(f"[Batch] 完了: {case_id}  device={device}")

        # 現在表示中のケースなら表示を更新
        if self.current_case_id == case_id:
            self._reload_current_case()

        self._batch_idx += 1
        QTimer.singleShot(100, self._predict_next_in_batch)

    def _on_batch_pred_error(self, case_id, error_msg):
        print(f"[Batch] エラー {case_id}: {error_msg[:200]}")
        self._batch_idx += 1
        QTimer.singleShot(100, self._predict_next_in_batch)

    def _on_batch_finished(self):
        self.pred_progress.setVisible(False)
        self.pred_status_label.setText("推論完了")
        self._scan_images_folder()
        self._load_session_cases()
        self._update_predict_btn()
        n = len(self._batch_queue)
        QMessageBox.information(self, "推論完了", f"{n} 件の推論が完了しました。")

    # ════════════════════════════════════════════════════════
    # ケース読み込み・スライス表示
    # ════════════════════════════════════════════════════════

    def _on_case_row_changed(self, row):
        if row < 0:
            return
        if self.dirty_slices and not self.is_saved:
            reply = QMessageBox.question(
                self, "未保存の変更",
                "現在のケースに未保存の変更があります。保存しますか？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.save_current_case()
            elif reply == QMessageBox.Cancel:
                return
        item = self.case_list.item(row)
        if item:
            self._load_case(item.data(Qt.UserRole))

    def _load_case(self, case_id):
        case = self.session.get_case(case_id)
        if not case:
            return

        img_path = case.get("image_path")
        if not img_path or not Path(img_path).exists():
            QMessageBox.warning(self, "ファイルなし", f"画像ファイルが見つかりません:\n{img_path}")
            return

        try:
            img_array, spacing = load_nifti(img_path)
        except Exception as e:
            QMessageBox.critical(self, "読み込みエラー", str(e))
            return

        self.current_case_id = case_id
        self.current_image_array = img_array
        self.current_spacing = spacing
        self.current_num_slices = img_array.shape[0]
        self.current_slice_polygons = {}
        self._orig_pred_polygons = {}
        self.dirty_slices = set()
        self.is_saved = True

        # W/L: 元画像のまま（実際の値域にフィット）
        arr_f = img_array.astype(np.float32)
        v_min = int(arr_f.min())
        v_max = int(arr_f.max())
        ww = max(1, v_max - v_min)
        wc = v_min + ww // 2
        self.wc_slider.setRange(v_min, v_max)
        self.ww_slider.setRange(1, max(512, ww * 2))
        self._set_wl_silent(wc, ww)

        # マスク（編集済み → 予測 → ゼロ）
        mask_array = self._load_best_mask(case)
        max_label = self.label_config.max_label_id()
        for z in range(self.current_num_slices):
            self.current_slice_polygons[z] = mask_to_polygons(mask_array[z], max_label=max_label)

        # リセット用（予測）
        pred_path = case.get("pred_path")
        if pred_path and Path(pred_path).exists():
            try:
                pred_arr, _ = load_nifti(pred_path)
                for z in range(self.current_num_slices):
                    self._orig_pred_polygons[z] = mask_to_polygons(pred_arr[z], max_label=max_label)
            except Exception:
                pass

        self.current_slice_idx = 0
        self._display_slice(0, reset_zoom=True)
        self.session.set_last_case(case_id)
        self.btn_save.setEnabled(True)
        self.btn_complete.setEnabled(True)
        self.setWindowTitle(f"AnnotationEditor — {case_id}")

        # 未推論のケースには注意表示
        if not pred_path or not Path(pred_path).exists():
            self.pred_status_label.setText(
                f"⚠ {case_id} は未推論です。一括推論 or 個別推論で予測してください。"
            )
        else:
            self.pred_status_label.setText("")

    def _reload_current_case(self):
        """現在のケースを再読み込みする（推論完了後など）"""
        if self.current_case_id:
            self._load_case(self.current_case_id)

    def _load_best_mask(self, case):
        """編集済み → 予測 → ゼロマスク の優先順でマスクを返す"""
        shape = self.current_image_array.shape
        for key in ("corrected_path", "pred_path"):
            p = case.get(key)
            if p and Path(p).exists():
                try:
                    arr, _ = load_nifti(p)
                    if arr.shape == shape:
                        return arr
                except Exception:
                    pass
        return np.zeros(shape, dtype=np.uint8)

    def _display_slice(self, slice_idx, reset_zoom=False):
        if self.current_image_array is None:
            return
        if self.current_case_id and slice_idx != self.current_slice_idx:
            self.current_slice_polygons[self.current_slice_idx] = self.canvas.get_polygons()
        self.current_slice_idx = slice_idx
        img_slice = self.current_image_array[slice_idx]
        polygons = self.current_slice_polygons.get(slice_idx, [])
        self.canvas.load_slice(
            img_slice, polygons, slice_idx,
            reset_zoom=reset_zoom,
            wc=self.wc_slider.value(),
            ww=self.ww_slider.value(),
        )
        self._update_slice_label()
        self._update_undo_redo_btns()

    def _set_wl_silent(self, wc, ww):
        self.wc_slider.blockSignals(True)
        self.ww_slider.blockSignals(True)
        self.wc_slider.setValue(wc)
        self.ww_slider.setValue(ww)
        self.wc_val.setText(str(wc))
        self.ww_val.setText(str(ww))
        self.wc_slider.blockSignals(False)
        self.ww_slider.blockSignals(False)
        self.canvas.wc = float(wc)
        self.canvas.ww = float(ww)

    def _update_slice_label(self):
        if self.current_num_slices == 0:
            self.slice_label.setText("スライス: -/-")
            return
        mark = " *" if self.current_slice_idx in self.dirty_slices else ""
        self.slice_label.setText(
            f"スライス: {self.current_slice_idx + 1}{mark} / {self.current_num_slices}"
        )

    def _update_undo_redo_btns(self):
        self.btn_undo.setEnabled(self.canvas.can_undo())
        self.btn_redo.setEnabled(self.canvas.can_redo())

    # ════════════════════════════════════════════════════════
    # スライス操作
    # ════════════════════════════════════════════════════════

    def _prev_slice(self):
        self._go_slice(-1)

    def _next_slice(self):
        self._go_slice(1)

    def _go_slice(self, delta):
        if self.current_image_array is None:
            return
        new_idx = self.current_slice_idx + delta
        if 0 <= new_idx < self.current_num_slices:
            self.current_slice_polygons[self.current_slice_idx] = self.canvas.get_polygons()
            self._display_slice(new_idx)

    def _copy_from_prev(self):
        self._copy_from_adjacent(-1)

    def _copy_from_next(self):
        self._copy_from_adjacent(1)

    def _copy_from_adjacent(self, direction):
        if self.current_image_array is None:
            return
        src_idx = self.current_slice_idx + direction
        if src_idx < 0 or src_idx >= self.current_num_slices:
            return
        self.current_slice_polygons[self.current_slice_idx] = self.canvas.get_polygons()
        src_polys = self.current_slice_polygons.get(src_idx, [])
        if not src_polys:
            QMessageBox.information(self, "コピー元なし",
                f"スライス {src_idx + 1} にポリゴンがありません。")
            return
        label = "前" if direction == -1 else "次"
        reply = QMessageBox.question(
            self, "コピー確認",
            f"スライス {src_idx + 1}（{label}）のポリゴン {len(src_polys)} 個をコピーしますか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.canvas._push_undo()
            self.canvas.polygons = copy.deepcopy(src_polys)
            self.canvas.update()

    def _reset_current_slice(self):
        if not self.current_case_id:
            return
        orig = self._orig_pred_polygons.get(self.current_slice_idx)
        if orig is None:
            QMessageBox.information(self, "予測なし", "このケースに予測ファイルがありません。")
            return
        reply = QMessageBox.question(
            self, "リセット確認",
            "現在スライスを AI 予測結果に戻しますか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.canvas._push_undo()
            self.canvas.polygons = copy.deepcopy(orig)
            self.canvas.update()

    # ════════════════════════════════════════════════════════
    # イベントハンドラ
    # ════════════════════════════════════════════════════════

    def _on_slice_modified(self):
        self.dirty_slices.add(self.current_slice_idx)
        self.is_saved = False
        self._update_slice_label()
        self._update_undo_redo_btns()

    def _on_wl_changed(self):
        wc = self.wc_slider.value()
        ww = self.ww_slider.value()
        self.wc_val.setText(str(wc))
        self.ww_val.setText(str(ww))
        self.canvas.set_window_level(wc, ww)

    def _on_mask_toggle(self):
        self.canvas.mask_visible = self.btn_mask_toggle.isChecked()
        self.btn_mask_toggle.setText(
            "マスク: ON" if self.canvas.mask_visible else "マスク: OFF"
        )
        self.canvas.update()

    def _on_label_btn_clicked(self, label_id):
        for lid, btn in self._label_buttons.items():
            btn.setChecked(lid == label_id)
        self.canvas.current_label = label_id
        self.canvas.current_polygon = []
        self.canvas.setFocus()

    def update_label_palette_selection(self, label_id):
        for lid, btn in self._label_buttons.items():
            btn.setChecked(lid == label_id)

    # ════════════════════════════════════════════════════════
    # 保存
    # ════════════════════════════════════════════════════════

    def _apply_filter_to_current_case(self):
        """現在ケースの予測マスクに異常排除フィルタを手動適用する"""
        if not self.current_case_id:
            QMessageBox.information(self, "情報", "ケースを選択してください。")
            return

        case = self.session.get_case(self.current_case_id)
        pred_path = case.get("pred_path")
        if not pred_path or not Path(pred_path).exists():
            QMessageBox.warning(self, "予測なし",
                "このケースに予測ファイルがありません。\n先に推論を実行してください。")
            return

        reply = QMessageBox.question(
            self, "異常排除の確認",
            "予測マスクに異常排除フィルタを適用しますか？\n"
            "（極小コンポーネント / 画像端 / ラベル過剰個数を削除）\n\n"
            "※ 元の予測ファイルが上書きされます。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            pred_array, spacing = load_nifti(pred_path)
            filtered, fstats = filter_predictions(pred_array, self.filter_cfg)
            save_nifti(filtered, spacing, pred_path)
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))
            return

        summary = stats_summary(fstats)
        print(f"[Filter] {self.current_case_id}: {summary}")

        # 表示を更新
        self._reload_current_case()
        QMessageBox.information(self, "異常排除完了", f"{summary}")

    def save_current_case(self):
        if not self.current_case_id:
            return

        self.current_slice_polygons[self.current_slice_idx] = self.canvas.get_polygons()

        case = self.session.get_case(self.current_case_id)
        corrected_path = self.edited_dir / f"{self.current_case_id}_corrected.nii.gz"

        z, h, w = self.current_image_array.shape
        mask_array = np.zeros((z, h, w), dtype=np.uint8)
        for zi in range(z):
            polys = self.current_slice_polygons.get(zi, [])
            if polys:
                mask_array[zi] = polygons_to_mask(polys, (h, w))

        try:
            save_nifti(mask_array, self.current_spacing, corrected_path)
        except Exception as e:
            QMessageBox.critical(self, "保存エラー", str(e))
            return

        self._write_edit_log()

        self.session.update_case_after_save(
            self.current_case_id, list(self.dirty_slices), corrected_path
        )

        self.dirty_slices = set()
        self.is_saved = True
        self._update_slice_label()
        self._load_session_cases()
        self._update_predict_btn()

        # リスト内の選択状態を維持
        for i in range(self.case_list.count()):
            item = self.case_list.item(i)
            if item and item.data(Qt.UserRole) == self.current_case_id:
                self.case_list.blockSignals(True)
                self.case_list.setCurrentRow(i)
                self.case_list.blockSignals(False)
                break

        self.pred_status_label.setText(f"保存完了: {datetime.now().strftime('%H:%M:%S')}")

    def _write_edit_log(self):
        log_path = self.edited_dir / f"{self.current_case_id}_edit_log.json"
        log = {}
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except Exception:
                pass
        log[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = {
            "case_id": self.current_case_id,
            "edited_slices": sorted(self.dirty_slices),
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    # ════════════════════════════════════════════════════════
    # キーボードショートカット / 終了
    # ════════════════════════════════════════════════════════

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key_Left and mods == Qt.ControlModifier:
            self._copy_from_prev()
        elif key == Qt.Key_Right and mods == Qt.ControlModifier:
            self._copy_from_next()
        elif key == Qt.Key_Left:
            self._prev_slice()
        elif key == Qt.Key_Right:
            self._next_slice()
        elif key == Qt.Key_S and mods == Qt.ControlModifier:
            self.save_current_case()
        else:
            super().keyPressEvent(event)

    # ════════════════════════════════════════════════════════
    # ヘルプ
    # ════════════════════════════════════════════════════════

    def _show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("操作マニュアル — AnnotationEditor")
        dlg.resize(800, 620)

        lo = QVBoxLayout()
        lo.setContentsMargins(10, 10, 10, 8)
        lo.setSpacing(6)
        dlg.setLayout(lo)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #ccc; border-radius: 4px; }
            QTabBar::tab {
                padding: 6px 18px; font-size: 11px;
                background: #e0e0e0; border: 1px solid #ccc;
                border-bottom: none; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: #fff; font-weight: bold; color: #1565c0; }
        """)

        browser_style = (
            "QTextBrowser { background:#fff; border:none; "
            "font-size:12px; padding:4px; }"
        )

        def make_browser(html):
            b = QTextBrowser()
            b.setStyleSheet(browser_style)
            b.setHtml(self._help_wrap(html))
            return b

        # ── タブ1: 基本の流れ ──────────────────────────────
        tabs.addTab(make_browser("""
<h2>基本の流れ</h2>
<table width="100%" style="margin:10px 0;">
  <tr>
    <td align="center" style="background:#e3f2fd;border-radius:8px;padding:12px 6px;width:22%;">
      <b style="font-size:13px;color:#1565c0;">① データ追加</b><br/>
      <span style="font-size:11px;color:#555;">左パネル上部<br/>Data.txt / DICOM</span>
    </td>
    <td align="center" style="font-size:20px;color:#999;width:6%;">▶</td>
    <td align="center" style="background:#e8f5e9;border-radius:8px;padding:12px 6px;width:22%;">
      <b style="font-size:13px;color:#2e7d32;">② AI 推論</b><br/>
      <span style="font-size:11px;color:#555;">左パネル中部<br/>一括推論ボタン</span>
    </td>
    <td align="center" style="font-size:20px;color:#999;width:6%;">▶</td>
    <td align="center" style="background:#fff8e1;border-radius:8px;padding:12px 6px;width:22%;">
      <b style="font-size:13px;color:#e65100;">③ ケース選択</b><br/>
      <span style="font-size:11px;color:#555;">左パネル下部<br/>リストをクリック</span>
    </td>
    <td align="center" style="font-size:20px;color:#999;width:6%;">▶</td>
    <td align="center" style="background:#f3e5f5;border-radius:8px;padding:12px 6px;width:22%;">
      <b style="font-size:13px;color:#6a1b9a;">④ 編集・保存</b><br/>
      <span style="font-size:11px;color:#555;">右パネルで編集<br/>Ctrl+S で保存</span>
    </td>
  </tr>
</table>

<h3>左パネルの見方</h3>
<table width="100%">
  <tr><th>アイコン</th><th>ステータス</th><th>意味</th></tr>
  <tr><td align="center"><b>○</b></td><td>未推論</td><td>AI予測がまだ実行されていない</td></tr>
  <tr><td align="center" style="color:#e65100;"><b>✎</b></td><td>編集中</td><td>推論済み or 保存済み（完了前）</td></tr>
  <tr><td align="center" style="color:#2e7d32;"><b>✓</b></td><td>完了</td><td>「完了にする」ボタンで明示的に確定</td></tr>
</table>

<h3>右パネルの構成</h3>
<ul>
  <li><b>トップバー</b>: スライス移動 / WC・WW スライダー / 隣接コピー / Undo・Redo</li>
  <li><b>キャンバス</b>: 画像表示・ポリゴン編集エリア</li>
  <li><b>ラベルパレット</b>: 11 ラベルの選択ボタン（最下部）</li>
</ul>
"""), "① 基本の流れ")

        # ── タブ2: 主要な操作 ──────────────────────────────
        tabs.addTab(make_browser("""
<h2>主要な操作</h2>

<h3>ラベルの選択</h3>
<p>画面下部のパレットから選ぶか、キーボードショートカットで選択します。</p>
<table width="100%">
  <tr><th>キー</th><th>ラベル</th><th>キー</th><th>ラベル</th></tr>
  <tr><td><b>1</b></td><td>ir（下直筋）</td><td><b>7</b></td><td>io（下斜筋）</td></tr>
  <tr><td><b>2</b></td><td>mr（内直筋）</td><td><b>8</b></td><td>lac（涙腺）</td></tr>
  <tr><td><b>3</b></td><td>sr（上直筋）</td><td><b>9</b></td><td>on（視神経）</td></tr>
  <tr><td><b>4</b></td><td>so（上斜筋）</td><td><b>q</b></td><td>ball（眼球）</td></tr>
  <tr><td><b>5</b></td><td>lr（外直筋）</td><td><b>w</b></td><td>orbit（眼窩）</td></tr>
  <tr><td><b>6</b></td><td>lev（挙筋）</td><td><b>0</b></td><td>選択ツール</td></tr>
</table>
<p style="font-size:11px;color:#555;">※ 選択中のラベルはパレットのボタンが明るく（フルカラーに）なります。</p>

<h3>ポリゴンの描画</h3>
<ol>
  <li>ラベルを選択（1〜9 / q / w キー）</li>
  <li>キャンバス上を <b>左クリック</b> で頂点を追加</li>
  <li><b>右クリック</b> でポリゴン確定（3点以上必要）</li>
  <li><b>Esc キー</b> で描画キャンセル</li>
</ol>

<h3>ポリゴンの編集（選択ツール: 0）</h3>
<ol>
  <li><b>左クリック</b> でポリゴンを選択 → 白い頂点が表示される</li>
  <li>頂点を <b>ドラッグ</b> して位置を調整</li>
  <li><b>Delete キー</b> で選択中のポリゴンを削除</li>
  <li><b>右クリック</b> でマスクの透明度を切り替え（通常 → 薄め → 輪郭のみ）</li>
  <li>何もない場所を <b>左クリック</b> または <b>Esc</b> で選択解除</li>
</ol>

<h3>Undo / Redo</h3>
<ul>
  <li><b>Ctrl + Z</b>（または ↩ Undo ボタン）: 直前の操作を取り消す</li>
  <li><b>Ctrl + Y</b>（または ↪ Redo ボタン）: 取り消した操作をやり直す</li>
  <li>スライスごとに独立して最大 30 ステップ保持</li>
</ul>

<h3>保存</h3>
<ul>
  <li><b>Ctrl + S</b> または 💾 保存ボタン でケース全体を保存</li>
  <li>編集したスライスはスライス番号に <b>*</b> が付く</li>
  <li>保存するとステータスが <b>✎ 編集中</b> に変わる（<b>完了にするボタン</b>で「✓完了」にする）</li>
</ul>
"""), "② 主要な操作")

        # ── タブ3: その他の操作 ────────────────────────────
        tabs.addTab(make_browser("""
<h2>その他の操作</h2>

<h3>スライス移動</h3>
<table width="100%">
  <tr><th>操作</th><th>動作</th></tr>
  <tr><td><b>← / →</b> キー</td><td>前 / 次のスライスへ移動</td></tr>
  <tr><td>トップバーの ◀ / ▶ ボタン</td><td>同上</td></tr>
</table>
<p style="font-size:11px;color:#555;">スライス番号の表示例: <b>スライス: 5* / 20</b>（* は未保存の編集あり）</p>

<h3>隣接スライスのコピー</h3>
<ul>
  <li><b>Ctrl + ←</b>（または ← コピーボタン）: 前のスライスのポリゴンを現在スライスにコピー</li>
  <li><b>Ctrl + →</b>（または コピー →ボタン）: 次のスライスのポリゴンを現在スライスにコピー</li>
  <li>コピー前に確認ダイアログが表示されます</li>
</ul>

<h3>ズーム・パン</h3>
<table width="100%">
  <tr><th>操作</th><th>動作</th></tr>
  <tr><td>マウスホイール</td><td>ズームイン / アウト</td></tr>
  <tr><td>中ボタン（ホイールクリック）ドラッグ</td><td>画像をパン（移動）</td></tr>
</table>

<h3>コントラスト調整（WC / WW）</h3>
<ul>
  <li>トップバーの <b>WC</b>（Window Center）スライダー: 輝度の中心値</li>
  <li><b>WW</b>（Window Width）スライダー: 輝度の幅（小さいほどコントラスト強）</li>
  <li>デフォルトはケースごとに元画像の最小・最大値に自動設定</li>
</ul>

<h3>マスク表示切り替え</h3>
<ul>
  <li><b>Tab キー</b> または <b>マスク: ON/OFF ボタン</b> でアノテーションの表示/非表示をトグル</li>
  <li>OFF にすると元画像のみ表示（アノテーションを確認せず画像を見たいとき）</li>
</ul>

<h3>AI予測へのリセット</h3>
<ul>
  <li>トップバーの <b>リセットボタン</b>: 現在スライスのアノテーションを AI 予測結果に戻す</li>
  <li>AI予測ファイルがない場合は使用不可</li>
</ul>

<h3>異常排除フィルタ</h3>
<p>推論結果から明らかに誤ったコンポーネントを自動削除します。フィルタは以下の順で適用されます。</p>
<table width="100%">
  <tr><th>フィルタ</th><th>内容</th></tr>
  <tr><td><b>小面積除去</b></td><td>面積が <code>min_area_px</code> 未満のコンポーネントを削除</td></tr>
  <tr><td><b>端部除去</b></td><td>画像端から <code>edge_margin_pct</code>% 以内のコンポーネントを削除</td></tr>
  <tr><td><b>個数上限</b></td><td>同ラベルが <code>max_keep_per_label</code> 個を超える場合、面積上位のみ残す</td></tr>
</table>
<ul>
  <li><b>手動適用</b>: トップバーの <b>🧹 異常排除</b> ボタン → 現在ケースの予測ファイルに上書き適用</li>
  <li><b>自動適用</b>: 左パネルの「推論後に異常排除」チェックを ON にしてから一括推論を実行</li>
  <li>フィルタは予測ファイル（<code>_pred.nii.gz</code>）を上書きします。元に戻すには再推論が必要です</li>
</ul>
<p style="font-size:11px;color:#555;">設定値は <code>config.json</code> の <code>prediction_filter</code> で変更できます（デフォルト: min_area_px=20, edge_margin_pct=12, max_keep_per_label=2）</p>

<h3>ケースのフィルタ・検索</h3>
<ul>
  <li>フィルタボタン（すべて / 未推論 / ✎編集中 / ✓完了）でリストを絞り込み</li>
  <li>検索ボックスにケース ID の一部を入力するとリアルタイムで絞り込み</li>
</ul>
"""), "③ その他の操作")

        # ── タブ4: データ追加・高度な設定 ──────────────────
        tabs.addTab(make_browser("""
<h2>データ追加・高度な設定</h2>

<h3>共有フォルダから自動取得</h3>
<ul>
  <li>左パネルの <b>📡 共有フォルダから自動取得</b> ボタンで DICOMDIR 形式の共有フォルダをスキャン</li>
  <li><code>config.json</code> の <code>shared_folder_path</code> に共有フォルダのパスを設定</li>
  <li><code>auto_load_on_startup: true</code> にすると起動時に自動スキャン</li>
  <li>既にセッションにあるケースはスキップ（重複なし）</li>
</ul>

<h3>Data.txt からインポート（推奨）</h3>
<ol>
  <li>左パネルの <b>📋 Data.txt からインポート</b> をクリック</li>
  <li><code>Data.txt</code> を含む親フォルダを選択</li>
  <li>自動的に DICOM → NIfTI 変換され、ケースが追加される</li>
</ol>
<p>ケース ID は <code>{PatientID}_{Date}_{Time}_EX1_SE{n}</code> 形式で生成されます。</p>
<p style="background:#fff8e1;padding:6px 10px;border-left:3px solid #f9a825;font-size:11px;">
  <b>シリーズフィルタ</b>: <code>config.json</code> の <code>series_filter</code> に一致するシリーズのみ取り込まれます。<br/>
  例: <code>"series_filter": "eT1W_SE_cor"</code>　→　シリーズ名に "eT1W_SE_cor" を含むもののみ対象
</p>

<h3>DICOM フォルダを直接変換</h3>
<ul>
  <li>左パネルの <b>🏥 DICOM フォルダを直接変換して追加</b> をクリック</li>
  <li>IMG 番号順のシングルフレーム DICOM または Enhanced DICOM フォルダを選択</li>
</ul>

<h3>AI 一括推論</h3>
<ul>
  <li>左パネルの <b>▶ 一括推論</b> ボタンで、未推論ケースをすべて推論</li>
  <li>GPU（CUDA）が利用可能な場合は自動的に GPU を使用</li>
  <li>CUDA で失敗した場合は自動的に CPU で再実行</li>
</ul>

<h3>フォルダ構造</h3>
<pre style="background:#f5f5f5;padding:8px;border-radius:4px;font-size:11px;">
app/
  images/                        元画像    {ID}_0000.nii.gz
  output/{model_id}/
    predictions/                 AI予測    {ID}_pred.nii.gz
    edited/                      編集済み  {ID}_corrected.nii.gz
</pre>

<h3>config.json の設定変更</h3>
<pre style="background:#f5f5f5;padding:8px;border-radius:4px;font-size:11px;">
{
  "model_id": 119,                    ← 使用する nnUNet モデルの DatasetID
  "series_filter": "eT1W_SE_cor",     ← インポート対象のシリーズ名（部分一致）
  "shared_folder_path": "\\\\192.168.0.81\\Export",  ← 共有フォルダのパス
  "auto_load_on_startup": false       ← 起動時に自動スキャンするか
}
</pre>

<h3>nnUNet モデルの配置</h3>
<p>以下のいずれかにモデルを配置してください（上が優先）:</p>
<ol>
  <li><code>annotationApps/nnUNet_results/</code>（共有・推奨）</li>
  <li><code>AnnotationEditor/app/nnUNet_results/</code></li>
  <li><code>HospitalMuscleBatch/app/nnUNet_results/</code></li>
</ol>
<p style="font-size:11px;color:#555;">または環境変数 <code>nnUNet_results</code> を直接設定しても動作します。</p>
"""), "④ データ追加・設定")

        lo.addWidget(tabs)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.button(QDialogButtonBox.Close).setText("閉じる")
        btn_box.rejected.connect(dlg.reject)
        lo.addWidget(btn_box)

        dlg.exec_()

    @staticmethod
    def _help_wrap(body: str) -> str:
        return (
            '<html><body style="font-family:\'Meiryo UI\',sans-serif;'
            'color:#222;padding:10px;">'
            '<style>'
            'h2{color:#0d47a1;border-bottom:2px solid #1565c0;'
            'padding-bottom:4px;margin:0 0 10px;}'
            'h3{color:#1565c0;margin:14px 0 5px;}'
            'table{border-collapse:collapse;width:100%;margin:6px 0;font-size:11px;}'
            'th{background:#e3f2fd;border:1px solid #bbb;padding:5px 8px;text-align:left;}'
            'td{border:1px solid #ddd;padding:4px 8px;}'
            'ul,ol{margin:4px 0;padding-left:22px;line-height:1.7;}'
            'li{margin-bottom:2px;}'
            'code{background:#eee;padding:1px 5px;border-radius:3px;font-size:11px;}'
            'pre{font-family:monospace;}'
            '</style>'
            + body
            + '</body></html>'
        )

    def closeEvent(self, event):
        if self.dirty_slices and not self.is_saved:
            reply = QMessageBox.question(
                self, "未保存の変更",
                "未保存の変更があります。保存して終了しますか？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.save_current_case()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()


# ════════════════════════════════════════════════════════════
# エントリーポイント
# ════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont()
    font.setFamily("Meiryo UI")
    font.setPointSize(9)
    app.setFont(font)

    window = AnnotationEditorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
