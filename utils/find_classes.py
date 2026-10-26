import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
keywords = [k.lower() for k in sys.argv[2:]]


info = json.load(open(root / "cls_info" / "class_info.json", encoding="utf-8"))
if isinstance(info, dict):
    info = [{"id":k, **v} if isinstance(v, dict) else {"id":k, "name":v} for k, v in info.items()]
print("first entry, raw:", info[0], "\n")

total = 0
for entry in info:
    text = " | ".join(str(v) for k, v in entry.items() if k != "id")
    if not any(k in text.lower() for k in keywords):
        continue
    cid = str(entry.get("id", ""))
    folder = root / "all_images" / (f"{int(cid):04d}" if cid.isdigit() else cid)
    n = len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0
    total +=n
    print(f"{folder.name} n={n:<4} {text}")

print(f"\nimages in matching classes: {total}")
