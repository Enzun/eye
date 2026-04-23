import os
import shutil
import csv
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
import json

class DistributorThread(QThread):
    progress = pyqtSignal(int, int, str)
    log_msg = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, raw_dir, pred_dir, shared_dir, mapping_csv, group_size, prefix, num_digits):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.pred_dir = Path(pred_dir)
        self.shared_dir = Path(shared_dir)
        self.mapping_csv = Path(mapping_csv)
        self.group_size = group_size
        self.prefix = prefix
        self.num_digits = num_digits
        
    def run(self):
        try:
            self.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
            shared_images_raw = self.shared_dir / 'images' / 'raw'
            shared_images_pred = self.shared_dir / 'images' / 'pred'
            shared_corrected = self.shared_dir / 'corrected'
            shared_sessions = self.shared_dir / 'sessions'
            
            shared_images_raw.mkdir(parents=True, exist_ok=True)
            shared_images_pred.mkdir(parents=True, exist_ok=True)
            shared_corrected.mkdir(parents=True, exist_ok=True)
            shared_sessions.mkdir(parents=True, exist_ok=True)
            
            # 既存の対応表を読み込み
            mapping = {}
            max_index = 0
            if self.mapping_csv.exists():
                with open(self.mapping_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        case_id = row['case_id']
                        mapping[row['original_filename']] = row
                        
                        # Case001 の数値部分を抽出して最大値を更新
                        num_str = case_id.replace(self.prefix, "")
                        if num_str.isdigit():
                            max_index = max(max_index, int(num_str))

            # raw_dir のデータを走査
            raw_files = list(self.raw_dir.glob("*_0000.nii.gz"))
            if not raw_files:
                self.log_msg.emit("配布可能な NIfTI (*_0000.nii.gz) が見つかりません。")
                self.finished.emit(True)
                return

            new_cases = []
            all_cases_in_shared = [] # 共有フォルダに出ている全ケース
            
            total = len(raw_files)
            for i, rf in enumerate(raw_files):
                orig_name = rf.name.replace("_0000.nii.gz", "")
                # _pred.nii.gz / .nii.gz どちらの命名でも対応
                pred_file = self.pred_dir / f"{orig_name}.nii.gz"
                if not pred_file.exists():
                    pred_file_alt = self.pred_dir / f"{orig_name}_pred.nii.gz"
                    if pred_file_alt.exists():
                        pred_file = pred_file_alt
                
                # 予測結果がない場合はスキップ（Rawだけで送らない）
                if not pred_file.exists():
                    self.log_msg.emit(f"スキップ: {orig_name} (予測結果なし)")
                    continue
                    
                # 匿名化 ID生成・取得
                if orig_name in mapping:
                    case_id = mapping[orig_name]['case_id']
                else:
                    max_index += 1
                    case_id = f"{self.prefix}{max_index:0{self.num_digits}d}"
                    
                    parts = orig_name.split('_')
                    pat_id = parts[0] if parts else orig_name
                    ex_id = ""
                    se_id = ""
                    for p in parts:
                        if p.startswith("EX"): ex_id = p
                        elif p.startswith("SE"): se_id = p
                        
                    mapping[orig_name] = {
                        'case_id': case_id,
                        'original_filename': orig_name,
                        'patient_id': pat_id,
                        'exam_id': ex_id,
                        'series_id': se_id,
                        'convert_date': datetime.now().strftime("%Y-%m-%d")
                    }
                    new_cases.append(mapping[orig_name])
                
                all_cases_in_shared.append(case_id)
                self.progress.emit(i, total, f"配布中: {orig_name} -> {case_id}")
                
                # 共有フォルダへコピー
                out_raw = shared_images_raw / f"{case_id}_0000.nii.gz"
                out_pred = shared_images_pred / f"{case_id}_pred.nii.gz"
                
                # すでに両方とも共有フォルダにある場合はスキップ
                if out_raw.exists() and out_pred.exists():
                    self.log_msg.emit(f"スキップ: {case_id} (すでに共有フォルダに存在)")
                    continue
                
                if not out_raw.exists():
                    shutil.copy2(rf, out_raw)
                
                if pred_file.exists() and not out_pred.exists():
                    shutil.copy2(pred_file, out_pred)

            # マッピングCSVを保存
            if new_cases:
                file_exists = self.mapping_csv.exists()
                with open(self.mapping_csv, 'a', encoding='utf-8', newline='') as f:
                    fieldnames = ['case_id', 'original_filename', 'patient_id', 'exam_id', 'series_id', 'convert_date']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(new_cases)
                self.log_msg.emit(f"マッピングCSVを更新しました（新規 {len(new_cases)} 件）")

            # -----------------------------------------------------
            # グループ・Session の作成（新規のみ追記）
            # -----------------------------------------------------
            all_cases_in_shared = sorted(list(set(all_cases_in_shared)))
            assignments_path = self.shared_dir / 'assignments.json'

            # 既存の assignments.json を読み込み、割り当て済みケースを把握する
            if assignments_path.exists():
                with open(assignments_path, 'r', encoding='utf-8') as f:
                    assignments = json.load(f)
                existing_groups = assignments.get("groups", [])
                # 既存グループの最大インデックスを取得
                group_idx = max(
                    (int(g["group_id"].replace("grp", "")) for g in existing_groups
                     if g.get("group_id", "").startswith("grp")),
                    default=0
                ) + 1
                # 既に session.json に登録済みのケースIDを収集
                already_assigned: set = set()
                for g in existing_groups:
                    s_path = self.shared_dir / Path(g.get("session_file", ""))
                    if s_path.exists():
                        with open(s_path, 'r', encoding='utf-8') as sf:
                            s_data = json.load(sf)
                        for c in s_data.get("cases", []):
                            already_assigned.add(c["case_id"])
                self.log_msg.emit(
                    f"既存グループ: {len(existing_groups)} 件 / "
                    f"割り当て済みケース: {len(already_assigned)} 件"
                )
            else:
                assignments = {"version": "1.0", "groups": []}
                existing_groups = []
                group_idx = 1
                already_assigned = set()

            # 未割り当てのケースのみを新グループに振り分ける
            new_cases = [cid for cid in all_cases_in_shared if cid not in already_assigned]

            if not new_cases:
                self.log_msg.emit("新規の未割り当てケースはありません。assignments.json は変更しません。")
            else:
                self.log_msg.emit(f"新規 {len(new_cases)} 件を追加グループに割り当てます。")
                for chunk_start in range(0, len(new_cases), self.group_size):
                    chunk_cases = new_cases[chunk_start:chunk_start + self.group_size]
                    if not chunk_cases:
                        continue

                    grp_id = f"grp{group_idx:03d}"
                    session_filename = f"{grp_id}.json"

                    assignments["groups"].append({
                        "group_id":    grp_id,
                        "group_name":  f"担当グループ {group_idx}",
                        "case_start":  chunk_cases[0],
                        "case_end":    chunk_cases[-1],
                        "session_file": f"sessions/{session_filename}",
                        "note":        f"全 {len(chunk_cases)} 件",
                    })

                    # 新規グループの session.json を作成（既存は触らない）
                    session_data = {
                        "version": "1.0",
                        "cases": [
                            {
                                "case_id":        cid,
                                "image_path":     f"images/raw/{cid}_0000.nii.gz",
                                "pred_path":      f"images/pred/{cid}_pred.nii.gz",
                                "corrected_path": f"corrected/{cid}_corrected.nii.gz",
                                "status":         "pending",
                                "edited_slices":  [],
                                "last_edited":    None,
                            }
                            for cid in chunk_cases
                        ],
                        "last_case_id": None,
                    }
                    s_path = shared_sessions / session_filename
                    with open(s_path, 'w', encoding='utf-8') as f:
                        json.dump(session_data, f, ensure_ascii=False, indent=2)
                    self.log_msg.emit(
                        f"グループ作成: {grp_id}  ({chunk_cases[0]} 〜 {chunk_cases[-1]}, "
                        f"{len(chunk_cases)} 件)"
                    )
                    group_idx += 1

                with open(assignments_path, 'w', encoding='utf-8') as f:
                    json.dump(assignments, f, ensure_ascii=False, indent=2)
                self.log_msg.emit("assignments.json を更新しました。")

            self.progress.emit(total, total, "配布完了")
            self.finished.emit(True)

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.log_msg.emit(f"エラー発生: {e}\n{err}")
            self.finished.emit(False)
