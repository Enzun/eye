from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QGroupBox, QSpinBox, QLineEdit,
)
import os
import re
import json
import shutil
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from core.config_manager import load_config


class ExportThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, shared_dir: str, out_dir: str, dataset_id: int, dataset_name: str):
        super().__init__()
        self.shared_dir   = Path(shared_dir)
        self.out_dir      = Path(out_dir)
        self.dataset_id   = dataset_id
        self.dataset_name = dataset_name

    def run(self):
        try:
            assignments_file = self.shared_dir / "assignments.json"
            if not assignments_file.exists():
                self.progress.emit("⚠  assignments.json が見つかりません。")
                self.finished.emit(False)
                return

            with open(assignments_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            target_ds  = self.out_dir / f"Dataset{self.dataset_id:03d}_{self.dataset_name}"
            images_tr  = target_ds / "imagesTr"
            labels_tr  = target_ds / "labelsTr"
            images_tr.mkdir(parents=True, exist_ok=True)
            labels_tr.mkdir(parents=True, exist_ok=True)

            count = 0
            for g in data.get("groups", []):
                session_rel  = g.get("session_file", "")
                session_path = self.shared_dir / Path(session_rel)
                if not session_path.exists():
                    continue
                with open(session_path, "r", encoding="utf-8") as sf:
                    s_data = json.load(sf)
                for case in s_data.get("cases", []):
                    if case.get("status") != "corrected":
                        continue
                    cid      = case.get("case_id")
                    raw_src  = self.shared_dir / Path(case.get("image_path", ""))
                    corr_src = self.shared_dir / Path(case.get("corrected_path", ""))
                    if raw_src.exists() and corr_src.exists():
                        shutil.copy2(raw_src,  images_tr / f"{cid}_0000.nii.gz")
                        shutil.copy2(corr_src, labels_tr  / f"{cid}.nii.gz")
                        count += 1
                        self.progress.emit(f"収集完了: {cid}")
                    else:
                        self.progress.emit(
                            f"⚠  {cid} — 元ファイルまたは修正ファイルが見つかりません"
                        )

            # labels.json → dataset.json
            labels_json = self.shared_dir / "labels.json"
            labels_dict = {"background": 0}
            if labels_json.exists():
                with open(labels_json, "r", encoding="utf-8") as lf:
                    for item in json.load(lf):
                        labels_dict[item["name"]] = int(item["key"])
            else:
                labels_dict.update({"ir": 1, "mr": 2, "sr": 3, "so": 4, "lr": 5, "io": 6})

            dataset_info = {
                "channel_names": {"0": "MRI"},
                "labels":        labels_dict,
                "numTraining":   count,
                "file_ending":   ".nii.gz",
                "overwrite_image_reader_writer": "SimpleITKIO",
            }
            with open(target_ds / "dataset.json", "w", encoding="utf-8") as df:
                json.dump(dataset_info, df, indent=4)

            self.progress.emit(f"\n合計 {count} 件を出力しました。")
            self.progress.emit(f"出力先: {target_ds}")
            self.finished.emit(True)

        except Exception as e:
            import traceback
            self.progress.emit(f"❌  エラー:\n{traceback.format_exc()}")
            self.finished.emit(False)


class ExportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_output_dir = str(
            (Path(__file__).parent.parent / "../output").resolve()
        )
        self._build_ui()
        self._load_next_dataset_id()

    # ────────────────────────── ヘルパー ──────────────────────────────

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

    # ────────────────────────── UI構築 ────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── 設定セクション ──────────────────────────────────────────
        settings_group = QGroupBox("出力設定")
        settings_group.setStyleSheet(self._group_style("#e65100", "#fff8f0", "#ffcc80"))
        sg = QVBoxLayout(settings_group)
        sg.setContentsMargins(14, 18, 14, 14)
        sg.setSpacing(10)

        # Dataset ID / 名前
        row1 = QHBoxLayout()
        lbl_id = QLabel("Dataset ID:")
        lbl_id.setStyleSheet("font-size: 13px;")
        row1.addWidget(lbl_id)
        self.spin_dataset_id = QSpinBox()
        self.spin_dataset_id.setRange(1, 999)
        self.spin_dataset_id.setFixedWidth(80)
        self.spin_dataset_id.setStyleSheet("font-size: 13px;")
        self.spin_dataset_id.valueChanged.connect(self._update_output_path_label)
        row1.addWidget(self.spin_dataset_id)

        lbl_name = QLabel("タスク名:")
        lbl_name.setStyleSheet("font-size: 13px; margin-left: 16px;")
        row1.addWidget(lbl_name)
        self.line_dataset_name = QLineEdit("EyeMuscleSegmentation")
        self.line_dataset_name.setStyleSheet("font-size: 13px;")
        self.line_dataset_name.textChanged.connect(self._update_output_path_label)
        row1.addWidget(self.line_dataset_name)
        sg.addLayout(row1)

        # 出力先パス表示
        self.output_path_label = QLabel()
        self.output_path_label.setStyleSheet("color: #666; font-size: 12px;")
        self.output_path_label.setWordWrap(True)
        sg.addWidget(self.output_path_label)

        layout.addWidget(settings_group)

        # ── 実行ボタン ──────────────────────────────────────────────
        self.btn_run = QPushButton("📦  学習用データ (nnUNet_raw 形式) を出力")
        self.btn_run.setFixedHeight(46)
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #e65100; color: white; border-radius: 6px;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: #bf360c; }"
            "QPushButton:disabled { background-color: #ffcc80; }"
        )
        self.btn_run.clicked.connect(self.run_export)
        layout.addWidget(self.btn_run)

        # ── ログ ────────────────────────────────────────────────────
        log_label = QLabel("実行ログ")
        log_label.setStyleSheet("font-weight: bold; color: #444; font-size: 13px;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QTextEdit { background-color: #263238; color: #b2dfdb;"
            " font-family: Consolas, monospace; font-size: 12px; border-radius: 4px; }"
        )
        layout.addWidget(self.log_text)

    # ────────────────────────── ロジック ──────────────────────────────

    def _load_next_dataset_id(self):
        next_id = 100
        if os.path.exists(self._base_output_dir):
            for d in os.listdir(self._base_output_dir):
                m = re.match(r"Dataset(\d+)_", d)
                if m:
                    val = int(m.group(1))
                    if val >= next_id:
                        next_id = val + 1
        self.spin_dataset_id.setValue(next_id)
        self._update_output_path_label()

    def _update_output_path_label(self):
        ds_id   = self.spin_dataset_id.value()
        ds_name = self.line_dataset_name.text()
        out = Path(self._base_output_dir) / f"Dataset{ds_id:03d}_{ds_name}"
        self.output_path_label.setText(f"出力先:  {out}")

    def log(self, message: str):
        self.log_text.append(message)

    def run_export(self):
        config = load_config()
        shared_rel = config.get("paths", {}).get("shared_dir", "../../annotation_workspace")
        shared_path = str(self._resolve(shared_rel))

        ds_id   = self.spin_dataset_id.value()
        ds_name = self.line_dataset_name.text()

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        self.log("▶  学習データの抽出を開始します...")

        self.thread = ExportThread(shared_path, self._base_output_dir, ds_id, ds_name)
        self.thread.progress.connect(self.log)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, success: bool):
        self.btn_run.setEnabled(True)
        if success:
            self.log("✅  出力が完了しました。")
            self._load_next_dataset_id()
        else:
            self.log("❌  エラーが発生しました。ログを確認してください。")
