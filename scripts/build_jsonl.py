import os, json, random

IN_DIR = "./dataset/datasets/synthetic"
OUT_DIR = "./dataset"
os.makedirs(OUT_DIR, exist_ok=True)

pairs = []
files = sorted([f for f in os.listdir(IN_DIR) if f.startswith("input_") and f.endswith(".json")])
for f in files:
    idx = f.split("_")[1].split(".")[0]
    inp = json.load(open(os.path.join(IN_DIR, f), "r", encoding="utf-8"))
    lp = os.path.join(IN_DIR, f"layout_{idx}.json")
    if os.path.exists(lp):
        layout = json.load(open(lp, "r", encoding="utf-8"))
        pairs.append({"params": inp, "layout": layout})

random.shuffle(pairs)
cut = int(0.9 * len(pairs)) if pairs else 0
train, val = pairs[:cut], pairs[cut:]

with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as wf:
    for r in train: wf.write(json.dumps(r) + "\n")
with open(os.path.join(OUT_DIR, "val.jsonl"), "w", encoding="utf-8") as wf:
    for r in val: wf.write(json.dumps(r) + "\n")

print(f"✅ Wrote {len(train)} train and {len(val)} val rows to {OUT_DIR}")
