from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QInputDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import json
from pathlib import Path
from core.config_manager import load_config


class ProgressTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ────────────────────────── ヘルパー ──────────────────────────────

    def _resolve(self, rel: str) -> Path:
        return (Path(__file__).parent.parent / rel).resolve()

    # ────────────────────────── UI構築 ────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── ヘッダー ───────────────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("作業グループの進捗状況")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #333;")
        header.addWidget(title)
        header.addStretch()
        self.btn_refresh = QPushButton("🔄  最新情報を取得")
        self.btn_refresh.setFixedHeight(36)
        self.btn_refresh.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; border-radius: 5px;"
            " font-size: 13px; padding: 0 14px; }"
            "QPushButton:hover { background-color: #1565c0; }"
        )
        self.btn_refresh.clicked.connect(self.refresh_progress)
        header.addWidget(self.btn_refresh)
        layout.addLayout(header)

        # ── エラー表示（通常は非表示） ────────────────────────────
        self.error_label = QLabel()
        self.error_label.setStyleSheet(
            "QLabel { background-color: #ffebee; border: 1px solid #ef9a9a;"
            " border-radius: 5px; padding: 8px 14px; color: #c62828; font-size: 13px; }"
        )
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        # ── リネームヒント ─────────────────────────────────────────
        hint = QLabel("💡 グループ名をダブルクリックすると名前を変更できます")
        hint.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(hint)

        # ── テーブル ───────────────────────────────────────────────
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["グループ ID", "グループ名 (ダブルクリックで変更)", "担当ケース", "完了 / 全体", "ステータス"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet(
            "QTableWidget { font-size: 13px; gridline-color: #e0e0e0; }"
            "QHeaderView::section { background-color: #eceff1; font-weight: bold;"
            " font-size: 13px; padding: 6px; border: none;"
            " border-bottom: 1px solid #bdbdbd; }"
            "QTableWidget::item { padding: 4px; }"
            "QTableWidget::item:alternate { background-color: #f5f5f5; }"
        )
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self.table)

    # ────────────────────────── 更新 ──────────────────────────────────

    def _on_cell_double_clicked(self, row: int, col: int):
        """グループ名列（col 1）のダブルクリックでリネームダイアログを開く"""
        if col != 1:
            return
        grp_id_item = self.table.item(row, 0)
        name_item   = self.table.item(row, 1)
        if not grp_id_item or not name_item:
            return

        grp_id       = grp_id_item.text()
        current_name = name_item.text()

        new_name, ok = QInputDialog.getText(
            self, "グループ名を変更",
            f"{grp_id} の新しいグループ名:",
            text=current_name,
        )
        if not ok or not new_name.strip() or new_name.strip() == current_name:
            return

        # assignments.json を更新
        config = load_config()
        shared_rel = config.get("paths", {}).get("shared_dir", "../../annotation_workspace")
        assignments_file = self._resolve(shared_rel) / "assignments.json"

        if not assignments_file.exists():
            self.error_label.setText("⚠  assignments.json が見つかりません。")
            self.error_label.show()
            return

        try:
            with open(assignments_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for g in data.get("groups", []):
                if g.get("group_id") == grp_id:
                    g["group_name"] = new_name.strip()
                    break
            with open(assignments_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            name_item.setText(new_name.strip())
            self.error_label.hide()
        except Exception as e:
            self.error_label.setText(f"⚠  グループ名の保存に失敗しました: {e}")
            self.error_label.show()

    def refresh_progress(self):
        config = load_config()
        shared_rel = config.get("paths", {}).get("shared_dir", "../../annotation_workspace")
        shared_path = self._resolve(shared_rel)
        assignments_file = shared_path / "assignments.json"

        self.table.setRowCount(0)
        self.error_label.hide()

        if not shared_path.exists():
            self.error_label.setText(
                f"⚠  共有フォルダが見つかりません:\n{shared_path}"
            )
            self.error_label.show()
            return

        if not assignments_file.exists():
            self.error_label.setText(
                "⚠  assignments.json が見つかりません。\n"
                "「匿名化と配布」タブで配布処理を実行してください。"
            )
            self.error_label.show()
            return

        try:
            with open(assignments_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for g in data.get("groups", []):
                grp_id    = g.get("group_id", "")
                grp_name  = g.get("group_name", "Unknown")
                case_range = f"{g.get('case_start')}  〜  {g.get('case_end')}"
                session_rel = g.get("session_file", "")

                session_path = shared_path / Path(session_rel)
                total_cases = completed_cases = 0
                if session_path.exists():
                    with open(session_path, "r", encoding="utf-8") as sf:
                        s_data = json.load(sf)
                    cases = s_data.get("cases", [])
                    total_cases = len(cases)
                    completed_cases = sum(
                        1 for c in cases if c.get("status") == "corrected"
                    )

                progress_str = f"{completed_cases}  /  {total_cases}"
                if total_cases == 0:
                    status_str   = "データなし"
                    status_color = QColor("#9e9e9e")
                elif completed_cases == total_cases:
                    status_str   = "✅  完了"
                    status_color = QColor("#2e7d32")
                else:
                    pct = int(completed_cases / total_cases * 100) if total_cases else 0
                    status_str   = f"🔄  作業中 ({pct}%)"
                    status_color = QColor("#1565c0")

                row = self.table.rowCount()
                self.table.insertRow(row)
                for col, text in enumerate(
                    [grp_id, grp_name, case_range, progress_str, status_str]
                ):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    if col == 4:
                        item.setForeground(status_color)
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    self.table.setItem(row, col, item)

        except Exception as e:
            self.error_label.setText(f"⚠  読み込みエラー: {e}")
            self.error_label.show()
