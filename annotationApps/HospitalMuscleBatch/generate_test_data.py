
import os
import shutil
from pathlib import Path

def generate_test_data():
    base_dir = Path(r"C:\imageProcessing\DATA\2025_11_03\DATA")
    source_dir = base_dir / "147757" / "20240508" / "112048" / "EX1"
    data_txt_path = Path(r"C:\imageProcessing\DATA\2025_11_03\Data.txt")

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    # Define test cases
    test_cases = [
        # (ID, Date, Time, EX)
        ("147757", "20240601", "090000", "EX1"), # Different Date
        ("999999", "20240508", "112048", "EX1"), # Different ID
        ("147757", "20240508", "153000", "EX1"), # Different Time
        ("147757", "20240508", "112048", "EX2"), # Different EX
    ]

    series_info = [
        ("SE1", "302", "MR", "eT1W_SE_tra"),
        ("SE2", "401", "MR", "T2 TIME sue1 rfa180"),
        ("SE3", "502", "MR", "eT1W_SE_cor"),
        ("SE4", "602", "MR", "eT1W_SE_sag"),
        ("SE5", "701", "MR", "T2W_SPIR_cor"),
    ]

    new_entries = []

    for case in test_cases:
        p_id, date, time, ex = case
        target_dir = base_dir / p_id / date / time / ex

        print(f"Generating case: {target_dir}")
        
        # Create directory and copy data
        if target_dir.exists():
             print(f"  Target already exists, skipping copy: {target_dir}")
        else:
            try:
                shutil.copytree(source_dir, target_dir)
                print(f"  Copied data to {target_dir}")
            except Exception as e:
                print(f"  Error copying data: {e}")
                continue

        # Prepare Data.txt entry
        # Assuming [20230104] is constant or irrelevant for this test, keeping it simple.
        entry_header = f"\nPatientName:(ID:{p_id})[20230104]\n"
        new_entries.append(entry_header)

        for se, ser_no, modality, desc in series_info:
            rel_path = f"DATA\\{p_id}\\{date}\\{time}\\{ex}\\{se}"
            # Format: 	Series No.302(MR)[eT1W_SE_tra]		Directory: DATA\147757\20240508\112048\EX1\SE1
            line = f"\tSeries No.{ser_no}({modality})[{desc}]\t\tDirectory: {rel_path}\n"
            new_entries.append(line)

    # Append to Data.txt
    try:
        with open(data_txt_path, "a", encoding="utf-8") as f:
            f.writelines(new_entries)
        print(f"Updated {data_txt_path}")
    except Exception as e:
        print(f"Error updating Data.txt: {e}")

if __name__ == "__main__":
    generate_test_data()
