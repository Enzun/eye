import os
import shutil
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


def _normalize_filters(value) -> list:
    """series_filter を文字列・リスト問わず空文字除去済みリストに正規化する"""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [f for f in value if isinstance(f, str) and f]
    return []


class DICOMImporterThread(QThread):
    progress = pyqtSignal(int, int, str)
    log_msg  = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, mode: str, target_path: str, output_dir: str):
        super().__init__()
        self.mode        = mode   # 'dicomdir' | 'datatxt' | 'niftidir'
        self.target_path = target_path
        self.output_dir  = Path(output_dir)

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.mode == "dicomdir":
                self._import_dicomdir()
            elif self.mode == "datatxt":
                self._import_datatxt()
            elif self.mode.startswith("niftidir"):
                self._import_niftidir()
            self.finished.emit(True)
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[ERROR] {e}\n{traceback.format_exc()}")
            self.finished.emit(False)

    # ─────────────────────────── DICOMDIR ────────────────────────────

    def _import_dicomdir(self):
        from utils.dicomdir_parser import DICOMDIRParser, find_dicomdir
        from utils.dicom_handler import convert_multiframe_dicom_to_nifti
        import SimpleITK as sitk
        from core.config_manager import load_config

        config = load_config()
        filters = _normalize_filters(config.get("settings", {}).get("series_filter", []))

        if not self.target_path or not os.path.isdir(self.target_path):
            self.log_msg.emit("エラー: 有効な MRI フォルダが指定されていません。")
            return

        self.log_msg.emit(f"DICOMDIR を検索中: {self.target_path}")
        dicomdir_path = find_dicomdir(self.target_path)
        if not dicomdir_path:
            self.log_msg.emit("DICOMDIR が見つかりません。")
            return

        parser = DICOMDIRParser(dicomdir_path)
        if not parser.load():
            self.log_msg.emit("DICOMDIR の読み込みに失敗しました。")
            return

        # 全シリーズを取得してから複数フィルタで絞り込む
        series_list = parser.parse_series(series_filter=None)
        if filters:
            series_list = [
                s for s in series_list
                if any(f in s.get("series_description", "") for f in filters)
            ]

        filter_msg = f" (フィルタ: {filters})" if filters else " (フィルタなし)"
        if not series_list:
            self.log_msg.emit(f"対象シリーズが見つかりませんでした{filter_msg}。")
            return

        self.log_msg.emit(f"{len(series_list)} シリーズを検出{filter_msg}。")
        total = len(series_list)

        for i, s_info in enumerate(series_list):
            entry_id = s_info["entry_id"]
            out_path = self.output_dir / f"{entry_id}_0000.nii.gz"

            if out_path.exists():
                self.progress.emit(i + 1, total, f"スキップ (既存): {out_path.name}")
                continue

            self.progress.emit(i, total, f"変換中 ({i+1}/{total}): {entry_id}")
            try:
                first_img = parser.get_first_image_path(s_info)
                if not first_img:
                    self.log_msg.emit(f"画像ファイルなし: {entry_id}")
                    continue
                sitk_image = convert_multiframe_dicom_to_nifti(first_img)
                sitk.WriteImage(sitk_image, str(out_path))
                self.log_msg.emit(f"変換成功: {out_path.name}")
            except Exception as e:
                self.log_msg.emit(f"変換エラー ({entry_id}): {e}")

        self.progress.emit(total, total, "DICOMDIR インポート完了")

    # ─────────────────────────── Data.txt ────────────────────────────

    def _import_datatxt(self):
        import codecs
        from utils.dicomdir_parser import DICOMDIRParser, find_dicomdir
        from utils.dicom_handler import convert_multiframe_dicom_to_nifti, convert_dicom_folder_to_nifti
        import SimpleITK as sitk
        from core.config_manager import load_config

        self.log_msg.emit(f"Data.txt の探索を開始: {self.target_path}")
        target_dir = Path(self.target_path)
        data_txt_files = list(target_dir.rglob("Data.txt")) + list(target_dir.rglob("data.txt"))

        if not data_txt_files:
            self.log_msg.emit("Data.txt が見つかりませんでした。")
            return

        config = load_config()
        filters = _normalize_filters(config.get("settings", {}).get("series_filter", []))

        all_dirs = []
        for file_path in data_txt_files:
            try:
                with codecs.open(file_path, "r", "utf-8") as f:
                    content = f.readlines()
            except UnicodeDecodeError:
                with codecs.open(file_path, "r", "cp932") as f:
                    content = f.readlines()

            base_dir = file_path.parent
            for line in content:
                line = line.strip()
                if "Directory:" not in line:
                    continue
                if filters and not any(f in line for f in filters):
                    continue
                d = line.split("Directory:")[-1].strip()
                full_path = base_dir / d
                if full_path.exists():
                    all_dirs.append(full_path)

        if not all_dirs:
            self.log_msg.emit("対象シリーズに一致する DICOM フォルダが見つかりませんでした。")
            return

        total = len(all_dirs)
        self.log_msg.emit(f"{total} 件のフォルダを処理します。")

        for i, d_path in enumerate(all_dirs):
            files = os.listdir(d_path)
            im_files  = [f for f in files if f.startswith("IM_")]
            img_files = [f for f in files if f.startswith("IMG") and not f.endswith(".json")]

            entry_id = f"{d_path.parent.parent.name}_{d_path.parent.name}_{d_path.name}"
            out_path = self.output_dir / f"{entry_id}_0000.nii.gz"

            if out_path.exists():
                self.progress.emit(i + 1, total, f"スキップ (既存): {out_path.name}")
                continue

            self.progress.emit(i, total, f"変換中 ({i+1}/{total}): {d_path.name}")
            try:
                sitk_image = None
                if im_files:
                    sitk_image = convert_multiframe_dicom_to_nifti(str(d_path / im_files[0]))
                elif img_files:
                    sitk_image, _ = convert_dicom_folder_to_nifti(str(d_path))
                else:
                    self.log_msg.emit(f"有効な DICOM ファイルなし: {d_path}")
                    continue

                if sitk_image is not None:
                    import SimpleITK as sitk
                    sitk.WriteImage(sitk_image, str(out_path))
                    self.log_msg.emit(f"変換成功: {out_path.name}")
            except Exception as e:
                import traceback
                self.log_msg.emit(f"変換エラー ({d_path}): {e}\n{traceback.format_exc()}")

        self.progress.emit(total, total, "Data.txt インポート完了")

    # ─────────────────────────── 既存 NIfTI ─────────────────────────

    def _import_niftidir(self):
        self.log_msg.emit(f"NIfTI フォルダからコピー開始: {self.target_path}")
        nifti_files = list(Path(self.target_path).glob("*.nii.gz"))
        total = len(nifti_files)

        if total == 0:
            self.log_msg.emit("対象の .nii.gz ファイルが見つかりませんでした。")
            return

        for i, src in enumerate(nifti_files):
            dst = self.output_dir / src.name
            self.progress.emit(i, total, f"コピー中 ({i+1}/{total}): {src.name}")
            if dst.exists():
                self.log_msg.emit(f"スキップ (既存): {src.name}")
            else:
                shutil.copy2(src, dst)
                self.log_msg.emit(f"コピー完了: {src.name}")

        self.progress.emit(total, total, "NIfTI コピー完了")
