from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFrame, QGroupBox, QMessageBox,
)
from pathlib import Path
from core.config_manager import load_config
from core.inference_runner import InferenceThread


class InferenceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.refresh_status()

    # ─────────────────────────── ヘルパー ────────────────────────────

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

    # ────────────────��────────── UI構築 ──────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── ステータスバー ──────────────────────────────────────────
        status_frame = QFrame()
        status_frame.setStyleSheet(
            "QFrame { background-color: #e3f2fd; border: 1px solid #90caf9;"
            " border-radius: 6px; }"
        )
        sf_layout = QHBoxLayout(status_frame)
        sf_layout.setContentsMargins(14, 8, 10, 8)
        self.status_label = QLabel("─")
        self.status_label.setStyleSheet(
            "color: #1565c0; font-weight: bold; font-size: 13px;"
        )
        sf_layout.addWidget(self.status_label)
        sf_layout.addStretch()
        btn_refresh = QPushButton("🔄 更新")
        btn_refresh.setFixedWidth(72)
        btn_refresh.setStyleSheet(
            "QPushButton { background: transparent; color: #1976d2; border: none; font-size: 12px; }"
            "QPushButton:hover { color: #0d47a1; }"
        )
        btn_refresh.clicked.connect(self.refresh_status)
        sf_layout.addWidget(btn_refresh)
        layout.addWidget(status_frame)

        # ── 実行セクション ─────────────────────���─────────────────────
        run_group = QGroupBox("AI 推論実行 (nnUNet)")
        run_group.setStyleSheet(self._group_style("#2e7d32", "#f1f8e9", "#c8e6c9"))
        run_layout = QVBoxLayout(run_group)
        run_layout.setContentsMargins(14, 18, 14, 14)
        run_layout.setSpacing(10)

        info = QLabel(
            "rawData_nifti/ 内の未推論 NIfTI に対して nnUNet を一括実行します。\n"
            "推論済みのファイルはスキップされます。"
        )
        info.setStyleSheet("color: #555; font-size: 13px;")
        info.setWordWrap(True)
        run_layout.addWidget(info)

        self.btn_run = QPushButton("🧠  AI 推論を実行 (nnUNet)")
        self.btn_run.setFixedHeight(46)
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #388e3c; color: white; border-radius: 6px;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: #2e7d32; }"
            "QPushButton:disabled { background-color: #a5d6a7; }"
        )
        self.btn_run.clicked.connect(self.run_inference)
        run_layout.addWidget(self.btn_run)
        layout.addWidget(run_group)

        # ── ログ ───────────────────────���─────────────────────────���──
        log_label = QLabel("推論ログ")
        log_label.setStyleSheet("font-weight: bold; color: #444; font-size: 13px;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QTextEdit { background-color: #263238; color: #b2dfdb;"
            " font-family: Consolas, monospace; font-size: 12px; border-radius: 4px; }"
        )
        layout.addWidget(self.log_text)

    # ────────────────��────────── ステータス更新 ───────────────────────

    def refresh_status(self):
        config = load_config()
        raw_rel  = config.get("paths", {}).get("raw_nifti_dir",   "../rawData_nifti")
        pred_rel = config.get("paths", {}).get("predictions_dir", "../predictions")
        raw_path  = self._resolve(raw_rel)
        pred_path = self._resolve(pred_rel)

        raw_files = list(raw_path.glob("*_0000.nii.gz")) if raw_path.exists() else []
        raw_count = len(raw_files)

        predicted = 0
        if pred_path.exists() and raw_count > 0:
            predicted = sum(
                1 for f in raw_files
                if (pred_path / f.name.replace("_0000.nii.gz", ".nii.gz")).exists()
            )
        unpredicted = raw_count - predicted
        self.status_label.setText(
            f"全 {raw_count} 件  |  推論済み: {predicted} 件  |  未推論: {unpredicted} 件"
        )

    # ─────────────────────────── ログ ────────────────────────────────

    def log(self, message: str):
        self.log_text.append(message)

    # ─────────────────────────── アクション ──────────────────────────

    def run_inference(self):
        # ── CUDA 可否チェック ────────────────────────────────────────────
        cuda_available = False
        cuda_diag = []
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_diag.append(f"PyTorch: {torch.__version__}")
            if cuda_available:
                cuda_diag.append(f"CUDA: {torch.version.cuda}")
                cuda_diag.append(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                cuda_diag.append("※ CUDA利用不可（GPUドライバー未インストール / CUDAバージョン不一致の可能性）")
        except Exception as e:
            cuda_diag.append(f"[ERROR] torch import 失敗: {e}")

        device = "cuda" if cuda_available else "cpu"

        if not cuda_available:
            ret = QMessageBox.warning(
                self,
                "GPU が利用できません",
                "CUDA (GPU) が利用できません。\n\n"
                "CPU で推論を実行すると非常に長時間かかる可能性があります。\n"
                "それでも CPU で実行しますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return

        # ── 推論開始 ─────────────────────────────────────────────────
        config = load_config()
        raw_rel  = config.get("paths", {}).get("raw_nifti_dir",   "../rawData_nifti")
        pred_rel = config.get("paths", {}).get("predictions_dir", "../predictions")
        model_id = config.get("settings", {}).get("model_id", 119)

        dataset_name = config.get("settings", {}).get("model_dataset_name", "EyeMuscleSegmentation")
        raw_dir = str(self._resolve(raw_rel))
        out_dir = str(self._resolve(pred_rel))

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        device_str = "GPU (CUDA)" if cuda_available else "CPU"
        self.log(
            f"▶  推論を開始します  [{device_str}]\n"
            f"  モデル: Dataset{model_id:03d}_{dataset_name}\n"
            f"  入力: {raw_dir}\n  出力: {out_dir}\n"
            + "\n".join(f"  {d}" for d in cuda_diag)
        )

        self.thread = InferenceThread(raw_dir, out_dir, model_id, device=device,
                                      dataset_name=dataset_name)
        self.thread.log_msg.connect(self.log)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    # ────────────────────��────── シグナル受信 ────────────────────────

    def on_finished(self, success: bool):
        self.btn_run.setEnabled(True)
        if success:
            self.log("✅  推論が完了���ました。")
        else:
            self.log("❌  推論中にエラーが発生しました。ログを確認してくださ��。")
        self.refresh_status()
