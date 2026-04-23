from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QProgressBar, QRadioButton,
    QButtonGroup, QGroupBox, QFrame,
)
from PyQt5.QtCore import Qt
import os
from pathlib import Path
from core.config_manager import load_config
from core.dicom_importer import DICOMImporterThread


class ImportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.refresh_status()

    # ─────────────────────────── UI構築 ──────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── ステータスバー ──────────────────────────────────────────
        status_frame = QFrame()
        status_frame.setStyleSheet(
            "QFrame { background-color: #e8f5e9; border: 1px solid #a5d6a7;"
            " border-radius: 6px; }"
        )
        sf_layout = QHBoxLayout(status_frame)
        sf_layout.setContentsMargins(14, 8, 10, 8)

        self.status_label = QLabel("取り込み済み: ─ 件")
        self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold; font-size: 13px;")
        sf_layout.addWidget(self.status_label)
        sf_layout.addStretch()

        btn_refresh = QPushButton("🔄 更新")
        btn_refresh.setFixedWidth(72)
        btn_refresh.setStyleSheet(
            "QPushButton { background: transparent; color: #388e3c;"
            " border: none; font-size: 12px; }"
            "QPushButton:hover { color: #1b5e20; }"
        )
        btn_refresh.clicked.connect(self.refresh_status)
        sf_layout.addWidget(btn_refresh)

        layout.addWidget(status_frame)

        # ── Section 1 : 自動取得（DICOMDIR） ────────────────────────
        auto_group = QGroupBox("MRI 共有フォルダから自動取得（DICOMDIR）")
        auto_group.setStyleSheet(self._group_style(accent="#1565c0", bg="#f3f8ff",
                                                    border="#bbdefb"))
        auto_layout = QVBoxLayout(auto_group)
        auto_layout.setSpacing(10)
        auto_layout.setContentsMargins(14, 18, 14, 14)

        config = load_config()
        mri_path = config.get("paths", {}).get(
            "mri_folder_path",
            "（未設定 — config.json の paths.mri_folder_path を設定してください）"
        )
        self.mri_path_label = QLabel(f"対象フォルダ:  {mri_path}")
        self.mri_path_label.setStyleSheet("color: #555; font-size: 12px;")
        self.mri_path_label.setWordWrap(True)
        auto_layout.addWidget(self.mri_path_label)

        self.btn_auto_load = QPushButton("🔄  共有フォルダから自動取得")
        self.btn_auto_load.setFixedHeight(46)
        self.btn_auto_load.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; border-radius: 6px;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:disabled { background-color: #90caf9; }"
        )
        self.btn_auto_load.clicked.connect(self.run_auto_load)
        auto_layout.addWidget(self.btn_auto_load)

        layout.addWidget(auto_group)

        # ── Section 2 : 既存 NIfTI コピー ───────────────────────────
        nifti_group = QGroupBox("既存 NIfTI フォルダからコピー")
        nifti_group.setStyleSheet(self._group_style(accent="#546e7a", bg="#fafafa",
                                                     border="#cfd8dc"))
        nifti_layout = QHBoxLayout(nifti_group)
        nifti_layout.setContentsMargins(14, 18, 14, 14)

        self.radio_raw = QRadioButton("Raw（元データ）")
        self.radio_pred = QRadioButton("Prediction（予測済み）")
        self.radio_raw.setChecked(True)
        self.nifti_type_group = QButtonGroup(self)
        self.nifti_type_group.addButton(self.radio_raw, 1)
        self.nifti_type_group.addButton(self.radio_pred, 2)

        self.btn_nifti = QPushButton("📁  フォルダを選択して追加")
        self.btn_nifti.setFixedHeight(38)
        self.btn_nifti.setStyleSheet(
            "QPushButton { background-color: #546e7a; color: white; border-radius: 5px;"
            " font-size: 12px; padding: 0 14px; }"
            "QPushButton:hover { background-color: #455a64; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.btn_nifti.clicked.connect(self.run_nifti_copy)

        nifti_layout.addWidget(self.radio_raw)
        nifti_layout.addWidget(self.radio_pred)
        nifti_layout.addStretch()
        nifti_layout.addWidget(self.btn_nifti)

        layout.addWidget(nifti_group)

        # ── ログ & プログレス ────────────────────────────────────────
        log_label = QLabel("ログ")
        log_label.setStyleSheet("font-weight: bold; color: #444; font-size: 13px;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QTextEdit { background-color: #263238; color: #b2dfdb;"
            " font-family: Consolas, monospace; font-size: 12px; border-radius: 4px; }"
        )
        layout.addWidget(self.log_text)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ccc; border-radius: 4px;"
            " text-align: center; font-size: 11px; }"
            "QProgressBar::chunk { background-color: #42a5f5; border-radius: 3px; }"
        )
        layout.addWidget(self.progress_bar)

    # ─────────────────────────── スタイルヘルパー ─────────────────────

    @staticmethod
    def _group_style(accent: str, bg: str, border: str) -> str:
        return (
            f"QGroupBox {{ font-weight: bold; border: 1px solid {border};"
            f" border-radius: 6px; margin-top: 8px; background-color: {bg}; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px;"
            f" padding: 0 6px; color: {accent}; }}"
        )

    # ─────────────────────────── ステータス更新 ───────────────────────

    def refresh_status(self):
        config = load_config()
        raw_rel = config.get("paths", {}).get("raw_nifti_dir", "../rawData_nifti")
        raw_path = (Path(__file__).parent.parent / raw_rel).resolve()
        if raw_path.exists():
            count = len(list(raw_path.glob("*_0000.nii.gz")))
            self.status_label.setText(f"取り込み済み: {count} 件  (rawData_nifti)")
        else:
            self.status_label.setText("取り込み済み: 0 件  （rawData_nifti フォルダ未作成）")

    # ─────────────────────────── ログ ────────────────────────────────

    def log(self, message: str):
        self.log_text.append(message)

    # ─────────────────────────── パス解決 ────────────────────────────

    def _output_dir(self, is_pred: bool = False) -> str:
        config = load_config()
        key = "predictions_dir" if is_pred else "raw_nifti_dir"
        default = "../predictions" if is_pred else "../rawData_nifti"
        rel = config.get("paths", {}).get(key, default)
        return str((Path(__file__).parent.parent / rel).resolve())

    # ─────────────────────────── スレッド起動 ────────────────────────

    def _start_thread(self, mode: str, path: str, output_dir: str):
        self.btn_auto_load.setEnabled(False)
        self.btn_nifti.setEnabled(False)
        self.progress_bar.setValue(0)
        self.thread = DICOMImporterThread(mode, path, output_dir)
        self.thread.progress.connect(self.update_progress)
        self.thread.log_msg.connect(self.log)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    # ─────────────────────────── アクション ──────────────────────────

    def run_auto_load(self):
        config = load_config()
        folder = config.get("paths", {}).get("mri_folder_path", "")
        if not folder or not os.path.exists(folder):
            self.log(f"⚠  MRI フォルダが見つかりません: '{folder}'")
            self.log("    → config.json の paths.mri_folder_path を確認してください。")
            return
        self.log(f"▶  自動取得を開始します: {folder}")
        self._start_thread("dicomdir", folder, self._output_dir(False))

    def run_nifti_copy(self):
        is_pred = self.radio_pred.isChecked()
        folder = QFileDialog.getExistingDirectory(self, "NIfTI フォルダを選択")
        if not folder:
            return
        label = "Prediction" if is_pred else "Raw"
        self.log(f"▶  NIfTI コピーを開始します [{label}]: {folder}")
        self._start_thread("niftidir", folder, self._output_dir(is_pred))

    # ─────────────────────────── シグナル受信 ────────────────────────

    def update_progress(self, current: int, total: int, msg: str):
        self.log(msg)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_finished(self, success: bool):
        self.btn_auto_load.setEnabled(True)
        self.btn_nifti.setEnabled(True)
        if success:
            self.log("✅  インポートが完了しました。")
        else:
            self.log("❌  エラーが発生しました。ログを確認してください。")
        self.refresh_status()
