"""
session_manager.py
session.json によるケース状態の永続管理

ステータス値:
    pending   : AI推論なし（未推論）
    editing   : 推論済み or 編集保存済み（完了マーク前）
    corrected : 「完了にする」ボタンで明示的に完了にしたもの

※ 旧バージョンの "predicted" は読み込み時に "editing" として扱う
"""

import json
from datetime import datetime
from pathlib import Path


class SessionManager:
    """セッション状態を session.json で管理する"""

    def __init__(self, session_path=None):
        if session_path is None:
            session_path = Path(__file__).resolve().parent.parent.parent / "session.json"
        self.session_path = Path(session_path)
        self._data = self._load()

    # ── ファイル I/O ─────────────────────────────────────

    def _load(self):
        if self.session_path.exists():
            try:
                with open(self.session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 旧ステータス "predicted" を "editing" に移行
                for c in data.get("cases", []):
                    if c.get("status") == "predicted":
                        c["status"] = "editing"
                return data
            except Exception:
                pass
        return {"version": "1.0", "cases": [], "last_case_id": None}

    def _save(self):
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── ケース管理 ───────────────────────────────────────

    def get_cases(self):
        return list(self._data.get("cases", []))

    def get_case(self, case_id):
        for c in self._data["cases"]:
            if c["case_id"] == case_id:
                return c
        return None

    def add_case(self, case_id, image_path, pred_path=None, corrected_path=None):
        """ケースを追加する。既存の場合は pred_path のみ更新して False を返す"""
        existing = self.get_case(case_id)
        if existing:
            if pred_path and not existing.get("pred_path"):
                existing["pred_path"] = str(pred_path)
                if existing.get("status") == "pending":
                    existing["status"] = "editing"
                self._save()
            return False
        entry = {
            "case_id": case_id,
            "image_path": str(image_path),
            "pred_path": str(pred_path) if pred_path else None,
            "corrected_path": str(corrected_path) if corrected_path else None,
            "status": "pending",
            "edited_slices": [],
            "last_edited": None,
        }
        self._data["cases"].append(entry)
        self._save()
        return True

    def remove_case(self, case_id):
        self._data["cases"] = [c for c in self._data["cases"] if c["case_id"] != case_id]
        if self._data.get("last_case_id") == case_id:
            self._data["last_case_id"] = None
        self._save()

    def update_case_pred_path(self, case_id, pred_path):
        """推論完了後にパスとステータスを更新する"""
        case = self.get_case(case_id)
        if case:
            case["pred_path"] = str(pred_path)
            if case.get("status") == "pending":
                case["status"] = "editing"
            self._save()

    def update_case_status(self, case_id, status):
        case = self.get_case(case_id)
        if case:
            case["status"] = status
            self._save()

    def update_case_after_save(self, case_id, edited_slices, corrected_path=None):
        """保存完了後にケース情報を更新する（ステータスは editing のまま）"""
        case = self.get_case(case_id)
        if not case:
            return
        case["edited_slices"] = sorted(set(edited_slices))
        case["last_edited"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if corrected_path:
            case["corrected_path"] = str(corrected_path)
        # 保存したら常に editing に戻す（完了は mark_case_complete のみ）
        case["status"] = "editing"
        self._save()

    def mark_case_complete(self, case_id):
        """ケースを完了状態にする"""
        case = self.get_case(case_id)
        if case:
            case["status"] = "corrected"
            self._save()

    def get_unpredicted_ids(self):
        """pred_path が存在しない（または None）のケース ID リストを返す"""
        result = []
        for c in self._data["cases"]:
            pred = c.get("pred_path")
            if not pred or not Path(pred).exists():
                result.append(c["case_id"])
        return result

    # ── 最後に開いたケース ────────────────────────────────

    def set_last_case(self, case_id):
        self._data["last_case_id"] = case_id
        self._save()

    def get_last_case_id(self):
        return self._data.get("last_case_id")
