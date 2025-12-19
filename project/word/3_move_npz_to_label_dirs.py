# word/3_move_npz_to_label_dirs.py
# python word/3_move_npz_to_label_dirs.py
import shutil
from pathlib import Path

SRC = Path("data/clip_npz")
DST_BASE = Path("data/word_sequences")
DST_BASE.mkdir(parents=True, exist_ok=True)

npzs = sorted(SRC.glob("*.npz"))
print(f"Found {len(npzs)} files in {SRC}")

for p in npzs:
    name = p.stem  # no extension
    parts = name.split("_")
    if len(parts) >= 2:
        label = parts[1].lower()
    else:
        label = "unknown"
    dest_dir = DST_BASE / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / p.name
    shutil.move(str(p), str(dest_path))
    print(f"Moved {p.name} -> {dest_dir}/")
print("Done.")
