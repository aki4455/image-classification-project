import json
import shutil
import sys
from pathlib import Path


def strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, str):
        yield obj


root = Path(sys.argv[1])
move = "--move" in sys.argv

split = json.load(open(root / "dataset_split" / "dataset_split.json", encoding="utf-8"))
known = set(strings(split))
files = [p for p in (root / "all_images").rglob("*") if p.is_file()]
extra = [p for p in files if p.name not in known]
ours = [p for p in extra if ".jpg_" in p.name]

print(f"files on disk: {len(files)} | not in split: {len(extra)} | of those, made by the old bug: {len(ours)}")
for p in extra[:10]:
    print("   ", p.relative_to(root))

if move:
    quarantine = root / "_quarantine"
    for p in extra:
        target = quarantine / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))
    print(f"moved {len(extra)} files to {quarantine}")
else:
    print("dry run: nothing moved. Add --move to quarantine them.")