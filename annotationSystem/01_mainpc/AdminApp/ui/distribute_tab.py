from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFrame, QGroupBox, QSpinBox, QLineEdit, QProgressBar,
)
from pathlib import Path
from core.config_manager import load_config
from core.distributor import DistributorThread


class DistributeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ───────���─────────────────── ヘルパー ────────────────────────────

    def _resolve(self, rel: str) -> Path:
        return (Path(__file__).parent.parent / rel).resolve()

    @staticmethod
    def _group_style(accent: str, bg: str, border: str) -> str:
        return (
            f"QGroupBox {{ font-weight: bold; border: 1px solid {border};"
            f" border-radius: 6px; margin-top: 8px; background-color: {bg}; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px;"
            f" padding: 0 6px; color: {accent}; }}"
        )

    # ────���────────────────────── UI構築 ──────────���───────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── 注意バナー ──────���────────────────────��──────────────────
        warn_frame = QFrame()
        warn_frame.setStyleSheet(
            "QFrame { background-color: #e3f2fd; border: 1px solid #90caf9;"
            " border-radius: 6px; }"
        )
        wl = QHBoxLayout(warn_frame)
        wl.setContentsMargins(14, 10, 14, 10)
        warn_label = QLabel(
            "ℹ  新規データのみ末尾にグループ追加されます。既存グループと session.json は変更されません。\n"
            "グループを完全に再構成したい場合は、assignments.json を削除してから実行してください。"
        )
        warn_label.setWordWrap(True)
        warn_label.setStyleSheet("color: #1565c0; font-size: 13px;")
        wl.addWidget(warn_label)
        layout.addWidget(warn_frame)

        # ── 設定セクション ─────���─────────────────────────────────────
        settings_group = QGroupBox("配布設定")
        settings_group.setStyleSheet(self._group_style("#546e7a", "#fafafa", "#cfd8dc"))
        sg = QVBoxLayout(settings_group)
        sg.setContentsMargins(14, 18, 14, 14)
        sg.setSpacing(10)

        row1 = QHBoxLayout()
        lbl1 = QLabel("1グループあたりの件数:")
        lbl1.setStyleSheet("font-size: 13px;")
        row1.addWidget(lbl1)
        self.spin_group_size = QSpinBox()
        self.spin_group_size.setRange(1, 100)
        self.spin_group_size.setValue(15)
        self.spin_group_size.setFixedWidth(80)
        row1.addWidget(self.spin_group_size)
        row1.addStretch()
        sg.addLayout(row1)

        row2 = QHBoxLayout()
        lbl2 = QLabel("匿名化プレフィックス:")
        lbl2.setStyleSheet("font-size: 13px;")
        row2.addWidget(lbl2)
        self.line_prefix = QLineEdit("Case")
        self.line_prefix.setFixedWidth(100)
        self.line_prefix.setStyleSheet("font-size: 13px;")
        row2.addWidget(self.line_prefix)
        row2.addStretch()
        sg.addLayout(row2)

        layout.addWidget(settings_group)

        # ── 実行ボタン ─────────���─────────────────────────────────────
        self.btn_run = QPushButton("🚀  匿名化と共有フォルダへの配布を実行")
        self.btn_run.setFixedHeight(46)
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; border-radius: 6px;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: #b71c1c; }"
            "QPushButton:disabled { background-color: #ef9a9a; }"
        )
        self.btn_run.clicked.connect(self.run_distribute)
        layout.addWidget(self.btn_run)

        # ── ログ & プログレス ────────────────────────────────────────
        log_label = QLabel("配布ログ")
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
            "QProgressBar::chunk { background-color: #ef5350; border-radius: 3px; }"
        )
        layout.addWidget(self.progress_bar)

    # ─────────���───────────────── ログ ───────────���────────────────────

    def log(self, message: str):
        self.log_text.append(message)

    # ─────────��───────────────── アクション ─────��────────────────────

    def run_distribute(self):
        config = load_config()
        raw_dir  = str(self._resolve(config.get("paths", {}).get("raw_nifti_dir",   "../rawData_nifti")))
        pred_dir = str(self._resolve(config.get("paths", {}).get("predictions_dir", "../predictions")))
        shared   = str(self._resolve(config.get("paths", {}).get("shared_dir",      "../../annotation_workspace")))
        mapping  = str(self._resolve(config.get("paths", {}).get("mapping_csv",     "../mapping/case_mapping.csv")))

        group_size = self.spin_group_size.value()
        prefix     = self.line_prefix.text()
        num_digits = config.get("settings", {}).get("case_digits", 3)

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log("▶  匿名化・配布処理を開始します...")

        self.thread = DistributorThread(raw_dir, pred_dir, shared, mapping,
                                        group_size, prefix, num_digits)
        self.thread.progress.connect(self.update_progress)
        self.thread.log_msg.connect(self.log)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    # ────────���────────────────── シグナル受信 ────────────────────────

    def update_progress(self, current: int, total: int, msg: str):
        self.log(msg)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_finished(self, success: bool):
        self.btn_run.setEnabled(True)
        if success:
            self.log("✅  配布処理が完了しました。")
        else:
            self.log("❌  エラーが発生しました。ログを確認してください。")
