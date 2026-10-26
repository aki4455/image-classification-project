import json, sys

def find(obj, target, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            find(v, target, f"{path}[{k!r}]")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find(v, target, f"{path}[{i}]")
    elif isinstance(obj, str) and target in obj:
        print(path, "->", obj)

find(json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2])