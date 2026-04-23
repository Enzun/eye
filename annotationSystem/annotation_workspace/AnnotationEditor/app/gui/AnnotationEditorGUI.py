"""
AnnotationEditorGUI.py  (Lite 版)
眼筋 MRI アノテーション補正専用 GUI

分散アノテーションワークフロー用ライト版:
  - AI推論・DICOM取り込み機能なし
  - assignments.json からグループ選択
  - session.json の相対パスを workspace_root で解決

workspace_root の決定:
  1. config.json["workspace_root"] が非空なら使用（例: \\192.168.0.81\shared）
  2. 未設定なら app_dir.parent（AnnotationEditor/ の親 = 02_shared/）を使用
"""

import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── ポータブル Python 用 DLL パス設定 ────────────────────
if sys.platform == "win32":
    _base = os.path.dirname(os.path.abspath(__file__))
    for _candidate in [
        # 病院annotationSystem/python311（app/gui/ から 4 段上）
        os.path.normpath(os.path.join(_base, "..", "..", "..", "..", "python311")),
        # フォールバック: AnnotationEditor 直下
        os.path.normpath(os.path.join(_base, "..", "..", "python311")),
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

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSlider, QSplitter,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

import numpy as np

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
from gui.editor_canvas import EditorCanvas


# ════════════════════════════════════════════════════════════
# 予測ラベルリマップ（推論番号 → labels.json 番号の変換）
# ════════════════════════════════════════════════════════════

def load_model_label_map(model_id: int) -> dict:
    """nnUNet_results の dataset.json からモデルのラベルマップを読み込む。
    nnUNet_results が未設定の場合は空辞書を返す。"""
    nnunet_results = os.environ.get("nnUNet_results", "")
    if not nnunet_results:
        return {}

    base = Path(nnunet_results) / f"Dataset{model_id:03d}_EyeMuscleSegmentation"
    candidates = [
        base / "nnUNetTrainer__nnUNetPlans__2d" / "dataset.json",
        base / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "dataset.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                labels = data.get("labels", {})
                return {int(v): k for k, v in labels.items() if k != "background"}
            except Exception:
                pass
    return {}


def build_label_remap(model_label_map: dict, label_config) -> dict:
    """{model_int: canonical_int} のリマップ辞書を生成する。変換不要なラベルは含まない。"""
    if not model_label_map:
        return {}
    name_to_id = {label_config.get_name(lid): lid for lid in label_config.get_all_ids()}
    remap = {}
    for model_int, name in model_label_map.items():
        canonical_int = name_to_id.get(name)
        if canonical_int is not None and model_int != canonical_int:
            remap[model_int] = canonical_int
    return remap


def apply_label_remap(pred_arr: np.ndarray, remap: dict) -> np.ndarray:
    """予測配列にラベルリマップを適用する。remap が空の場合は入力をそのまま返す。"""
    if not remap:
        return pred_arr
    result = pred_arr.copy()
    for src, dst in remap.items():
        result[pred_arr == src] = dst
    return result


# ════════════════════════════════════════════════════════════
# アプリ起動ヘルパー
# ════════════════════════════════════════════════════════════

def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    # resolve() は Windows ネットワークドライブで UNC パスに戻すため
    # os.path.abspath() (GetFullPathNameW) を使いドライブレターを維持する
    return Path(os.path.abspath(__file__)).parent.parent.parent  # AnnotationEditor/


def get_workspace_root(app_dir: Path, cfg: dict) -> Path:
    """workspace_root を決定する。優先順:
    1. 環境変数 ANNOTEDITOR_WORKSPACE (bat が drive-letter path で設定)
    2. config.json["workspace_root"] が非空
    3. app_dir.parent（AnnotationEditor/ の親）
    """
    # 1. bat ファイルがドライブレターパスを渡す場合（UNCパス問題回避）
    env_root = os.environ.get("ANNOTEDITOR_WORKSPACE", "").strip()
    if env_root:
        return Path(env_root)
    # 2. config.json の明示的指定
    override = cfg.get("workspace_root", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else (app_dir / p).resolve()
    # 3. 自動検出
    return app_dir.parent  # AnnotationEditor/ の親 = annotation_workspace/


# ════════════════════════════════════════════════════════════
# グループ選択ダイアログ
# ════════════════════════════════════════════════════════════

class GroupSelectionDialog(QDialog):
    """assignments.json からグループを選択するダイアログ"""

    def __init__(self, workspace_root: Path, last_group_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AnnotationEditor Lite — 作業グループの選択")
        self.setMinimumWidth(520)
        self.setMinimumHeight(300)
        self.selected_group_id = None

        lo = QVBoxLayout(self)
        lo.setSpacing(10)
        lo.setContentsMargins(16, 16, 16, 12)

        assignments_path = workspace_root / "assignments.json"

        if not assignments_path.exists():
            lo.addWidget(QLabel(
                f"⚠  assignments.json が見つかりません:\n{assignments_path}\n\n"
                "管理 PC で「匿名化と配布」を実行してください。"
            ))
            btn = QPushButton("閉じる")
            btn.clicked.connect(self.reject)
            lo.addWidget(btn)
            return

        try:
            with open(assignments_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            groups = data.get("groups", [])
        except Exception as e:
            lo.addWidget(QLabel(f"⚠  読み込みエラー: {e}"))
            btn = QPushButton("閉じる")
            btn.clicked.connect(self.reject)
            lo.addWidget(btn)
            return

        title = QLabel("担当グループを選択してください")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px 0;")
        lo.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { font-size: 13px; border: 1px solid #ccc; border-radius: 4px; }"
            "QListWidget::item { padding: 9px 12px; border-bottom: 1px solid #f0f0f0; }"
            "QListWidget::item:selected { background-color: #1976d2; color: white; }"
            "QListWidget::item:hover:!selected { background: #e3f2fd; }"
        )
        self.list_widget.itemDoubleClicked.connect(self._on_accept)

        for g in groups:
            grp_id   = g.get("group_id", "")
            grp_name = g.get("group_name", "Unknown")
            case_start = g.get("case_start", "")
            case_end   = g.get("case_end",   "")
            session_rel = g.get("session_file", "")

            completed, total = 0, 0
            if session_rel:
                sp = workspace_root / session_rel
                if sp.exists():
                    try:
                        with open(sp, "r", encoding="utf-8") as f:
                            sd = json.load(f)
                        cases = sd.get("cases", [])
                        total = len(cases)
                        completed = sum(
                            1 for c in cases if c.get("status") == "corrected"
                        )
                    except Exception:
                        pass

            if total > 0:
                progress_str = f"  [{completed} / {total} 完了]"
            else:
                progress_str = "  [データなし]"

            display = f"{grp_id}  |  {grp_name}  |  {case_start} 〜 {case_end}{progress_str}"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, grp_id)
            if completed == total and total > 0:
                item.setForeground(QColor("#2e7d32"))
            self.list_widget.addItem(item)

        # 前回選択グループを初期選択
        pre_selected = False
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole) == last_group_id:
                self.list_widget.setCurrentRow(i)
                pre_selected = True
                break
        if not pre_selected and self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

        lo.addWidget(self.list_widget)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("選択")
        btn_box.button(QDialogButtonBox.Cancel).setText("キャンセル")
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        lo.addWidget(btn_box)

    def _on_accept(self):
        item = self.list_widget.currentItem()
        if item:
            self.selected_group_id = item.data(Qt.UserRole)
            self.accept()


# ════════════════════════════════════════════════════════════
# メインウィンドウ
# ════════════════════════════════════════════════════════════

class AnnotationEditorApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self._ready = False  # main() が show() を呼ぶ前にチェックする

        self.app_dir = get_app_dir()

        # ── config 読み込み ───────────────────────────────
        cfg_path = self.app_dir / "config.json"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        self.filter_cfg = cfg.get("prediction_filter", {})
        workspace_root  = get_workspace_root(self.app_dir, cfg)
        last_group_id   = cfg.get("last_group_id", "")

        # ── グループ選択ダイアログ ─────────────────────────
        dlg = GroupSelectionDialog(workspace_root, last_group_id)
        if dlg.exec_() != QDialog.Accepted or not dlg.selected_group_id:
            return  # main() が _ready=False を見て sys.exit() する

        selected_group_id = dlg.selected_group_id
        self._save_config_field(cfg_path, cfg, "last_group_id", selected_group_id)

        # ── グループ情報の取得 ────────────────────────────
        assignments_path = workspace_root / "assignments.json"
        try:
            with open(assignments_path, "r", encoding="utf-8") as f:
                assignments = json.load(f)
        except Exception as e:
            QMessageBox.critical(None, "エラー",
                f"assignments.json の読み込みに失敗しました:\n{e}")
            return

        group_info = next(
            (g for g in assignments.get("groups", [])
             if g.get("group_id") == selected_group_id),
            None,
        )
        if not group_info:
            QMessageBox.critical(None, "エラー",
                f"グループ {selected_group_id} が見つかりません。")
            return

        self.workspace_root = workspace_root
        self.group_id   = selected_group_id
        self.group_name = group_info.get("group_name", selected_group_id)
        self.case_start = group_info.get("case_start", "")
        self.case_end   = group_info.get("case_end",   "")
        self._group_session_rel = group_info.get("session_file", "")

        # ── ラベル設定 ────────────────────────────────────
        self.label_config = LabelConfig(self.app_dir / "labels.json")
        self._pred_remap = build_label_remap(
            load_model_label_map(119),  # nnUNet_results 未設定 → {} → リマップなし
            self.label_config,
        )

        # ── セッション読み込み ────────────────────────────
        session_path = workspace_root / self._group_session_rel
        self.session = SessionManager(session_path)

        # ── ケース状態 ────────────────────────────────────
        self.current_case_id      = None
        self.current_image_array  = None   # (Z, Y, X)
        self.current_spacing      = None
        self.current_slice_polygons = {}   # {slice_idx: [polygon,...]}
        self.current_num_slices   = 0
        self.current_slice_idx    = 0
        self._orig_pred_polygons  = {}     # リセット用（AI 予測）

        self.dirty_slices = set()
        self.is_saved     = True
        self.status_filter = None

        self._ready = True
        self._init_ui()

        QTimer.singleShot(300, self._startup_scan)

    # ════════════════════════════════════════════════════════
    # UI 構築
    # ════════════════════════════════════════════════════════

    def _init_ui(self):
        self.setWindowTitle(
            f"AnnotationEditor Lite — {self.group_id}  ({self.group_name})"
        )
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

        # ── タイトル行 ────────────────────────────────────
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

        sub = QLabel("眼筋 MRI アノテーション補正  (Lite 版)")
        sub.setStyleSheet("font-size: 12px; color: #888; margin-bottom: 2px;")
        lo.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ddd;")
        lo.addWidget(sep)

        # ── グループ情報 ──────────────────────────────────
        grp_frame = QFrame()
        grp_frame.setStyleSheet(
            "QFrame { background: #e8f0fe; border: 1px solid #b0c4f5;"
            " border-radius: 6px; }"
        )
        grp_lo = QVBoxLayout(grp_frame)
        grp_lo.setContentsMargins(10, 8, 10, 8)
        grp_lo.setSpacing(4)

        self.grp_id_label = QLabel(f"グループ: {self.group_id}  |  {self.group_name}")
        self.grp_id_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1a237e; border: none;"
        )
        self.grp_id_label.setWordWrap(True)
        grp_lo.addWidget(self.grp_id_label)

        self.grp_range_label = QLabel(f"担当: {self.case_start} 〜 {self.case_end}")
        self.grp_range_label.setStyleSheet("font-size: 13px; color: #3949ab; border: none;")
        grp_lo.addWidget(self.grp_range_label)

        btn_change = QPushButton("グループを変更")
        btn_change.setFixedHeight(26)
        btn_change.setStyleSheet(
            "QPushButton { background: #3f51b5; color: white; border: none;"
            " border-radius: 4px; font-size: 13px; padding: 0 10px; }"
            "QPushButton:hover { background: #303f9f; }"
        )
        btn_change.clicked.connect(self._change_group)
        grp_lo.addWidget(btn_change)
        lo.addWidget(grp_frame)

        # ── ステータスサマリー ──────────────────────────
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-size: 13px; color: #555; padding: 3px 0;")
        self.summary_label.setWordWrap(True)
        lo.addWidget(self.summary_label)

        # ── フィルタ ──────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        self.btn_filter_all       = QPushButton("すべて")
        self.btn_filter_pending   = QPushButton("未編集")
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
                QPushButton { padding: 4px 10px; border: 1px solid #ccc;
                              border-radius: 10px; font-size: 12px; background: #fff; }
                QPushButton:checked { background: #1565c0; color: white; border-color: #1565c0; }
            """)
            btn.clicked.connect(lambda _, v=fval: self._set_status_filter(v))
            filter_row.addWidget(btn)
        self.btn_filter_all.setChecked(True)
        filter_row.addStretch()
        lo.addLayout(filter_row)

        # ── 検索 ──────────────────────────────────────
        search_row = QHBoxLayout()
        search_row.setSpacing(4)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("ケース ID で検索...")
        self.search_box.setStyleSheet("""
            QLineEdit { border: 1px solid #ddd; border-radius: 4px;
                        padding: 5px 8px; font-size: 12px; }
            QLineEdit:focus { border-color: #1976d2; }
        """)
        self.search_box.textChanged.connect(self._filter_case_list)
        search_row.addWidget(self.search_box)
        btn_clr = QPushButton("×")
        btn_clr.setFixedSize(28, 28)
        btn_clr.setStyleSheet(
            "border: 1px solid #ddd; border-radius: 4px;"
            " background: #f0f0f0; font-size: 14px;"
        )
        btn_clr.clicked.connect(self.search_box.clear)
        search_row.addWidget(btn_clr)
        lo.addLayout(search_row)

        # ── ケースリスト ──────────────────────────────
        self.case_list = QListWidget()
        self.case_list.setStyleSheet("""
            QListWidget { border: 1px solid #ddd; border-radius: 4px;
                          background: #fff; outline: none; font-size: 13px; }
            QListWidget::item { padding: 7px 10px; border-bottom: 1px solid #f0f0f0; }
            QListWidget::item:selected { background: #e3f2fd; color: #0d47a1; }
            QListWidget::item:hover:!selected { background: #fafafa; }
        """)
        self.case_list.currentRowChanged.connect(self._on_case_row_changed)
        lo.addWidget(self.case_list, 1)

        # ── 操作ボタン ────────────────────────────────
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

        # ── ステータスバー（保存完了・警告など） ─────
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            "font-size: 12px; color: #555; padding: 2px 0;"
        )
        self.status_label.setWordWrap(True)
        lo.addWidget(self.status_label)

        # ── 右クリックメニュー（削除） ────────────────
        self.case_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.case_list.customContextMenuRequested.connect(
            self._on_case_list_context_menu
        )

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
        wc_lbl.setStyleSheet("color: #90caf9; font-size: 13px;")
        lo.addWidget(wc_lbl)
        self.wc_slider = self._make_slider(0, 255, 128)
        self.wc_slider.valueChanged.connect(self._on_wl_changed)
        lo.addWidget(self.wc_slider)
        self.wc_val = QLabel("128")
        self.wc_val.setStyleSheet("color: #90caf9; font-size: 13px; min-width: 32px;")
        lo.addWidget(self.wc_val)

        ww_lbl = QLabel("WW")
        ww_lbl.setStyleSheet("color: #90caf9; font-size: 13px;")
        lo.addWidget(ww_lbl)
        self.ww_slider = self._make_slider(1, 512, 256)
        self.ww_slider.valueChanged.connect(self._on_wl_changed)
        lo.addWidget(self.ww_slider)
        self.ww_val = QLabel("256")
        self.ww_val.setStyleSheet("color: #90caf9; font-size: 13px; min-width: 32px;")
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
                                 border-radius: 4px; padding: 0 10px; font-size: 13px; }
            QPushButton:!checked { background: #546e7a; }
        """)
        self.btn_mask_toggle.clicked.connect(self._on_mask_toggle)
        lo.addWidget(self.btn_mask_toggle)

        btn_reset = self._text_btn("リセット", self._reset_current_slice, color="#ffab40")
        btn_reset.setToolTip("現在スライスを AI 予測に戻す")
        lo.addWidget(btn_reset)

        btn_filter = self._text_btn(
            "🧹 異常排除", self._apply_filter_to_current_case, color="#80cbc4"
        )
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
            btn.setToolTip(display)
            self._label_buttons[lid] = btn
            lo.addWidget(btn)

        lo.addStretch()

        hint = QLabel(
            "左クリック=頂点追加  右クリック=ポリゴン確定  ホイール=ズーム  "
            "中ボタンドラッグ=パン  M=マスクON/OFF  ←→=スライス  Ctrl+Z/Y=Undo/Redo"
        )
        hint.setStyleSheet("color: #eceff1; font-size: 12px;")
        lo.addWidget(hint)

        self._label_buttons[0].setChecked(True)
        return palette

    # ── UI ヘルパー ──────────────────────────────────────

    @staticmethod
    def _btn_style(color):
        return (
            f"QPushButton {{ padding: 7px; background: {color}; color: white; "
            "border: none; border-radius: 4px; font-size: 13px; }} "
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
            "border-radius: 4px; padding: 0 8px; font-size: 13px;"
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
        rh, gh, bh = min(r + 40, 255), min(g + 40, 255), min(b + 40, 255)
        btn = QPushButton(f"{display}\n[{key}]")
        btn.setCheckable(True)
        btn.setFixedSize(64, 52)
        btn.setStyleSheet(f"""
            QPushButton {{
                color: rgb({r},{g},{b});
                background: #2e3d45;
                border: 2px solid rgb({r},{g},{b});
                border-radius: 4px; font-size: 12px; font-weight: bold;
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
    # 設定保存ヘルパー
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _save_config_field(cfg_path: Path, cfg: dict, key: str, value):
        """config.json の特定フィールドを更新して保存する"""
        cfg[key] = value
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Config] {key} 保存失敗: {e}")

    # ════════════════════════════════════════════════════════
    # 起動時読み込み
    # ════════════════════════════════════════════════════════

    def _startup_scan(self):
        """session.json のケースをリストに展開し、最後のケースを復元する"""
        self._load_session_cases()

        last_id = self.session.get_last_case_id()
        if last_id:
            for i in range(self.case_list.count()):
                item = self.case_list.item(i)
                if item and item.data(Qt.UserRole) == last_id:
                    self.case_list.setCurrentRow(i)
                    break

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
        status  = case.get("status", "pending")

        if self.status_filter and status != self.status_filter:
            return

        icons = {"pending": "○", "editing": "✎", "corrected": "✓"}
        icon = icons.get(status, "○")
        n_edited  = len(case.get("edited_slices", []))
        edited_str = f"  ({n_edited}sl)" if n_edited else ""
        last       = (case.get("last_edited") or "")[:10]
        last_str   = f"  {last}" if last else ""

        item = QListWidgetItem(f"{icon}  {case_id}{edited_str}{last_str}")
        item.setData(Qt.UserRole, case_id)
        if status == "corrected":
            item.setForeground(QColor("#2e7d32"))
        elif status == "editing":
            item.setForeground(QColor("#e65100"))
        self.case_list.addItem(item)

    def _update_summary(self):
        cases    = self.session.get_cases()
        total    = len(cases)
        corrected = sum(1 for c in cases if c.get("status") == "corrected")
        editing   = sum(1 for c in cases if c.get("status") == "editing")
        pending   = total - corrected - editing
        self.summary_label.setText(
            f"合計 {total}件  ✓完了 {corrected}  ✎編集中 {editing}  ○未作業 {pending}"
        )

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
    # グループ変更
    # ════════════════════════════════════════════════════════

    def _change_group(self):
        if self.dirty_slices and not self.is_saved:
            reply = QMessageBox.question(
                self, "未保存の変更",
                "現在のケースに未保存の変更があります。保存してからグループを変更しますか？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.save_current_case()
            elif reply == QMessageBox.Cancel:
                return

        cfg_path = self.app_dir / "config.json"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        workspace_root = get_workspace_root(self.app_dir, cfg)

        dlg = GroupSelectionDialog(workspace_root, self.group_id, self)
        if dlg.exec_() != QDialog.Accepted or not dlg.selected_group_id:
            return
        if dlg.selected_group_id == self.group_id:
            return

        selected_group_id = dlg.selected_group_id
        self._save_config_field(cfg_path, cfg, "last_group_id", selected_group_id)

        assignments_path = workspace_root / "assignments.json"
        with open(assignments_path, "r", encoding="utf-8") as f:
            assignments = json.load(f)

        group_info = next(
            (g for g in assignments.get("groups", [])
             if g.get("group_id") == selected_group_id),
            None,
        )
        if not group_info:
            QMessageBox.critical(self, "エラー",
                f"グループ {selected_group_id} が見つかりません。")
            return

        self.workspace_root = workspace_root
        self.group_id   = selected_group_id
        self.group_name = group_info.get("group_name", selected_group_id)
        self.case_start = group_info.get("case_start", "")
        self.case_end   = group_info.get("case_end",   "")

        session_path = workspace_root / group_info.get("session_file", "")
        self.session  = SessionManager(session_path)

        # UI リセット
        self.current_case_id     = None
        self.current_image_array = None
        self.current_spacing     = None
        self.current_slice_polygons = {}
        self.current_num_slices  = 0
        self.current_slice_idx   = 0
        self._orig_pred_polygons = {}
        self.dirty_slices        = set()
        self.is_saved            = True
        self.status_filter       = None
        self.btn_filter_all.setChecked(True)
        self.canvas.load_empty()
        self.btn_save.setEnabled(False)
        self.btn_complete.setEnabled(False)
        self.status_label.setText("")

        # グループ情報ラベル更新
        self.grp_id_label.setText(
            f"グループ: {self.group_id}  |  {self.group_name}"
        )
        self.grp_range_label.setText(
            f"担当: {self.case_start} 〜 {self.case_end}"
        )

        self.setWindowTitle(
            f"AnnotationEditor Lite — {self.group_id}  ({self.group_name})"
        )
        self._load_session_cases()

    # ════════════════════════════════════════════════════════
    # ケースの操作
    # ════════════════════════════════════════════════════════

    def _mark_complete(self):
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
        for i in range(self.case_list.count()):
            item = self.case_list.item(i)
            if item and item.data(Qt.UserRole) == self.current_case_id:
                self.case_list.blockSignals(True)
                self.case_list.setCurrentRow(i)
                self.case_list.blockSignals(False)
                break

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

        img_rel = case.get("image_path")
        if not img_rel:
            QMessageBox.warning(self, "ファイルなし",
                f"ケース {case_id}: image_path が設定されていません。")
            return
        img_path = self.workspace_root / img_rel
        if not img_path.exists():
            QMessageBox.warning(self, "ファイルなし",
                f"画像ファイルが見つかりません:\n{img_path}")
            return

        try:
            img_array, spacing = load_nifti(str(img_path))
        except Exception as e:
            QMessageBox.critical(self, "読み込みエラー", str(e))
            return

        self.current_case_id        = case_id
        self.current_image_array    = img_array
        self.current_spacing        = spacing
        self.current_num_slices     = img_array.shape[0]
        self.current_slice_polygons = {}
        self._orig_pred_polygons    = {}
        self.dirty_slices           = set()
        self.is_saved               = True

        # WC/WW を画像の値域にフィット
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
        max_label  = self.label_config.max_label_id()
        for z in range(self.current_num_slices):
            self.current_slice_polygons[z] = mask_to_polygons(
                mask_array[z], max_label=max_label
            )

        # リセット用（AI 予測ポリゴン）
        pred_rel = case.get("pred_path")
        if pred_rel:
            pred_path = self.workspace_root / pred_rel
            if pred_path.exists():
                try:
                    pred_arr, _ = load_nifti(str(pred_path))
                    pred_arr = apply_label_remap(pred_arr, self._pred_remap)
                    for z in range(self.current_num_slices):
                        self._orig_pred_polygons[z] = mask_to_polygons(
                            pred_arr[z], max_label=max_label
                        )
                except Exception:
                    pass

        self.current_slice_idx = 0
        self._display_slice(0, reset_zoom=True)
        self.session.set_last_case(case_id)
        self.btn_save.setEnabled(True)
        self.btn_complete.setEnabled(True)
        self.setWindowTitle(
            f"AnnotationEditor Lite — {self.group_id} — {case_id}"
        )

        # 予測なしの場合に注記
        if not pred_rel or not (self.workspace_root / pred_rel).exists():
            self.status_label.setText(
                f"⚠ {case_id}: 予測ファイルなし。ゼロマスクで表示しています。"
            )
        else:
            self.status_label.setText("")

    def _reload_current_case(self):
        if self.current_case_id:
            self._load_case(self.current_case_id)

    def _load_best_mask(self, case):
        """編集済み → 予測 → ゼロマスク の優先順でマスクを返す"""
        shape = self.current_image_array.shape
        for key in ("corrected_path", "pred_path"):
            rel = case.get(key)
            if not rel:
                continue
            p = self.workspace_root / rel
            if p.exists():
                try:
                    arr, _ = load_nifti(str(p))
                    if arr.shape == shape:
                        if key == "pred_path":
                            arr = apply_label_remap(arr, self._pred_remap)
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
        polygons  = self.current_slice_polygons.get(slice_idx, [])
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
            QMessageBox.information(
                self, "コピー元なし",
                f"スライス {src_idx + 1} にポリゴンがありません。"
            )
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
            QMessageBox.information(
                self, "予測なし", "このケースに予測ファイルがありません。"
            )
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

        case     = self.session.get_case(self.current_case_id)
        pred_rel = case.get("pred_path")
        if not pred_rel:
            QMessageBox.warning(self, "予測なし",
                "このケースに予測ファイルがありません。")
            return
        pred_path = self.workspace_root / pred_rel
        if not pred_path.exists():
            QMessageBox.warning(self, "予測なし",
                f"予測ファイルが見つかりません:\n{pred_path}")
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
            pred_array, spacing = load_nifti(str(pred_path))
            filtered, fstats = filter_predictions(pred_array, self.filter_cfg)
            save_nifti(filtered, spacing, str(pred_path))
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))
            return

        summary = stats_summary(fstats)
        print(f"[Filter] {self.current_case_id}: {summary}")
        self._reload_current_case()
        QMessageBox.information(self, "異常排除完了", f"{summary}")

    def save_current_case(self):
        if not self.current_case_id:
            return

        self.current_slice_polygons[self.current_slice_idx] = self.canvas.get_polygons()

        # corrected ディレクトリを確保
        corrected_dir = self.workspace_root / "corrected"
        corrected_dir.mkdir(parents=True, exist_ok=True)

        corrected_rel = f"corrected/{self.current_case_id}_corrected.nii.gz"
        corrected_abs = self.workspace_root / corrected_rel

        z, h, w = self.current_image_array.shape
        mask_array = np.zeros((z, h, w), dtype=np.uint8)
        for zi in range(z):
            polys = self.current_slice_polygons.get(zi, [])
            if polys:
                mask_array[zi] = polygons_to_mask(polys, (h, w))

        try:
            save_nifti(mask_array, self.current_spacing, str(corrected_abs))
        except Exception as e:
            QMessageBox.critical(self, "保存エラー", str(e))
            return

        self._write_edit_log()

        # セッションに相対パスで記録
        self.session.update_case_after_save(
            self.current_case_id, list(self.dirty_slices), corrected_rel
        )

        self.dirty_slices = set()
        self.is_saved     = True
        self._update_slice_label()
        self._load_session_cases()

        # リスト内の選択状態を維持
        for i in range(self.case_list.count()):
            item = self.case_list.item(i)
            if item and item.data(Qt.UserRole) == self.current_case_id:
                self.case_list.blockSignals(True)
                self.case_list.setCurrentRow(i)
                self.case_list.blockSignals(False)
                break

        self.status_label.setText(
            f"保存完了: {datetime.now().strftime('%H:%M:%S')}"
        )

    def _write_edit_log(self):
        log_path = (
            self.workspace_root / "corrected"
            / f"{self.current_case_id}_edit_log.json"
        )
        log = {}
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except Exception:
                pass
        log[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = {
            "case_id":      self.current_case_id,
            "edited_slices": sorted(self.dirty_slices),
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    # ════════════════════════════════════════════════════════
    # キーボードショートカット / 終了
    # ════════════════════════════════════════════════════════

    def keyPressEvent(self, event):
        key  = event.key()
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
        dlg.setWindowTitle("操作マニュアル — AnnotationEditor Lite")
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
<h2>基本の流れ（Lite 版）</h2>
<table width="100%" style="margin:10px 0;">
  <tr>
    <td align="center" style="background:#e8f0fe;border-radius:8px;padding:12px 6px;width:30%;">
      <b style="font-size:13px;color:#1a237e;">① グループ選択</b><br/>
      <span style="font-size:11px;color:#555;">起動時に担当グループを選択</span>
    </td>
    <td align="center" style="font-size:20px;color:#999;width:5%;">▶</td>
    <td align="center" style="background:#fff8e1;border-radius:8px;padding:12px 6px;width:30%;">
      <b style="font-size:13px;color:#e65100;">② ケース選択</b><br/>
      <span style="font-size:11px;color:#555;">左パネルのリストをクリック</span>
    </td>
    <td align="center" style="font-size:20px;color:#999;width:5%;">▶</td>
    <td align="center" style="background:#f3e5f5;border-radius:8px;padding:12px 6px;width:30%;">
      <b style="font-size:13px;color:#6a1b9a;">③ 編集・保存・完了</b><br/>
      <span style="font-size:11px;color:#555;">右パネルで編集 / Ctrl+S 保存</span>
    </td>
  </tr>
</table>

<h3>左パネルの見方</h3>
<table width="100%">
  <tr><th>アイコン</th><th>ステータス</th><th>意味</th></tr>
  <tr><td align="center"><b>○</b></td><td>未作業</td><td>まだ編集していない</td></tr>
  <tr><td align="center" style="color:#e65100;"><b>✎</b></td><td>編集中</td><td>保存済み（完了マーク前）</td></tr>
  <tr><td align="center" style="color:#2e7d32;"><b>✓</b></td><td>完了</td><td>「完了にする」で明示的に確定</td></tr>
</table>

<h3>右パネルの構成</h3>
<ul>
  <li><b>トップバー</b>: スライス移動 / WC・WW スライダー / 隣接コピー / Undo・Redo</li>
  <li><b>キャンバス</b>: 画像表示・ポリゴン編集エリア</li>
  <li><b>ラベルパレット</b>: ラベルの選択ボタン（最下部）</li>
</ul>
"""), "① 基本の流れ")

        # ── タブ2: 主要な操作 ──────────────────────────────
        tabs.addTab(make_browser("""
<h2>主要な操作</h2>

<h3>ラベルの選択</h3>
<table width="100%">
  <tr><th>キー</th><th>ラベル</th><th>キー</th><th>ラベル</th></tr>
  <tr><td><b>1</b></td><td>ir（下直筋）</td><td><b>7</b></td><td>io（下斜筋）</td></tr>
  <tr><td><b>2</b></td><td>mr（内直筋）</td><td><b>8</b></td><td>lac（涙腺）</td></tr>
  <tr><td><b>3</b></td><td>sr（上直筋）</td><td><b>9</b></td><td>on（視神経）</td></tr>
  <tr><td><b>4</b></td><td>so（上斜筋）</td><td><b>q</b></td><td>ball（眼球）</td></tr>
  <tr><td><b>5</b></td><td>lr（外直筋）</td><td><b>w</b></td><td>orbit（眼窩）</td></tr>
  <tr><td><b>6</b></td><td>lev（挙筋）</td><td><b>0</b></td><td>選択ツール</td></tr>
</table>

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
  <li><b>右クリック</b> でマスクの透明度を切り替え</li>
</ol>

<h3>保存・完了</h3>
<ul>
  <li><b>Ctrl + S</b> または 💾 保存ボタン でケース全体を保存</li>
  <li>保存するとステータスが <b>✎ 編集中</b> になる</li>
  <li><b>✓ 完了にする</b> ボタンで「✓完了」に変更（管理 PC の進捗確認に反映）</li>
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

<h3>ズーム・パン</h3>
<table width="100%">
  <tr><th>操作</th><th>動作</th></tr>
  <tr><td>マウスホイール</td><td>ズームイン / アウト</td></tr>
  <tr><td>中ボタンドラッグ</td><td>画像をパン（移動）</td></tr>
</table>

<h3>コントラスト調整（WC / WW）</h3>
<ul>
  <li><b>WC</b>（Window Center）スライダー: 輝度の中心値</li>
  <li><b>WW</b>（Window Width）スライダー: 輝度の幅</li>
</ul>

<h3>AI予測へのリセット</h3>
<ul>
  <li>トップバーの <b>リセットボタン</b>: 現在スライスを AI 予測結果に戻す</li>
  <li>予測ファイルがない場合は使用不可</li>
</ul>

<h3>異常排除フィルタ</h3>
<ul>
  <li>トップバーの <b>🧹 異常排除</b>: 現在ケースの予測マスクにフィルタを適用</li>
  <li>フィルタ設定は <code>config.json</code> の <code>prediction_filter</code> で変更可能</li>
</ul>

<h3>グループ変更</h3>
<ul>
  <li>左パネルの <b>グループを変更</b> ボタンで別のグループに切り替え</li>
  <li>前回使ったグループは次回起動時に自動選択される</li>
</ul>
"""), "③ その他の操作")

        # ── タブ4: 設定・構成 ──────────────────────────────
        tabs.addTab(make_browser("""
<h2>設定・フォルダ構成</h2>

<h3>フォルダ構成（annotation_workspace/）</h3>
<pre style="background:#f5f5f5;padding:8px;border-radius:4px;font-size:11px;">
annotation_workspace/
  assignments.json         グループ定義
  sessions/
    grp001.json            グループ 1 のセッション
    grp002.json            グループ 2 のセッション
  images/
    raw/   CaseXXX_0000.nii.gz    元画像
    pred/  CaseXXX_pred.nii.gz    AI 予測
  corrected/
    CaseXXX_corrected.nii.gz      補正済みマスク
    CaseXXX_edit_log.json         編集ログ
  AnnotationEditor/
    config.json                   アプリ設定
</pre>

<h3>config.json の設定</h3>
<pre style="background:#f5f5f5;padding:8px;border-radius:4px;font-size:11px;">
{
  "workspace_root": "",      ← 空 = 自動検出（AnnotationEditor の親フォルダ）
                               例: "\\\\192.168.0.81\\shared" でネットワーク共有を指定
  "last_group_id": "grp001", ← 最後に使ったグループ（自動保存）
  "prediction_filter": {
    "min_area_px": 20,         ← この面積未満のコンポーネントを削除
    "edge_margin_pct": 12,     ← 端からこの割合(%)以内を削除
    "max_keep_per_label": 2,   ← 1ラベル当たりの最大個数
    "enable_count_filter": true
  }
}
</pre>

<p style="background:#e8f5e9;padding:8px;border-left:3px solid #4caf50;font-size:11px;">
  <b>病院 PC に持ち込む場合:</b> AnnotationEditor フォルダをそのままコピーし、
  <code>config.json</code> の <code>workspace_root</code> に共有フォルダのパスを設定してください。
</p>
"""), "④ 設定・構成")

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
    if not window._ready:
        sys.exit(0)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
