"""
label_config.py
labels.json を読み込み、ラベル情報を提供する
"""

import json
from pathlib import Path


class LabelConfig:
    """labels.json ベースのラベル設定管理"""

    SELECT_TOOL_ID = 0

    def __init__(self, labels_json_path=None):
        if labels_json_path is None:
            # labels.json は annotationApps/AnnotationEditor/labels.json
            labels_json_path = Path(__file__).resolve().parent.parent.parent / "labels.json"

        with open(labels_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._labels = {}   # {label_id: info_dict}
        self._key_map = {}  # {key_char: label_id}

        for id_str, info in data["labels"].items():
            lid = int(id_str)
            self._labels[lid] = info
            key = info.get("key", "").lower()
            if key:
                self._key_map[key] = lid

        self._label_ids = sorted(self._labels.keys())

    # ── 基本アクセサ ─────────────────────────────────────

    def get_all_ids(self):
        """全ラベルIDのリストを返す"""
        return list(self._label_ids)

    def get_name(self, label_id):
        """英名を返す (例: "ir")"""
        return self._labels.get(label_id, {}).get("name", str(label_id))

    def get_display(self, label_id):
        """表示名を返す (例: "下直筋")"""
        return self._labels.get(label_id, {}).get("display", str(label_id))

    def get_color(self, label_id):
        """RGB タプルを返す"""
        return tuple(self._labels.get(label_id, {}).get("color", [128, 128, 128]))

    def get_key(self, label_id):
        """キーボードショートカット文字を返す"""
        return self._labels.get(label_id, {}).get("key", "")

    def key_to_label_id(self, key_char):
        """キー文字 → ラベルID。対応なければ None"""
        return self._key_map.get(key_char.lower())

    def get_all_labels(self):
        """全ラベル情報を返す: [(id, name, display, color, key), ...]"""
        return [
            (lid,
             self.get_name(lid),
             self.get_display(lid),
             self.get_color(lid),
             self.get_key(lid))
            for lid in self._label_ids
        ]

    def max_label_id(self):
        """最大ラベルIDを返す"""
        return max(self._label_ids) if self._label_ids else 0
