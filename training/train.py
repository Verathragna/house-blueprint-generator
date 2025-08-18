import os, sys, json, argparse, re, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, repo_root)

from models.layout_transformer import LayoutTransformer
from tokenizer.tokenizer import BlueprintTokenizer, PAD_ID

CHECKPOINT_DIR = os.path.join(repo_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CKPT_LATEST = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
OPT_LATEST = os.path.join(CHECKPOINT_DIR, "optimizer_latest.pth")
LOG_FILE = os.path.join(repo_root, "training_log.txt")

class PairDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: BlueprintTokenizer):
        self.rows = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")] if os.path.exists(jsonl_path) else []
        self.tk = tokenizer

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x, y = self.tk.build_training_pair(row["params"], row["layout"])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=PAD_ID)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=PAD_ID)
    key_mask = (x_pad == PAD_ID)
    return x_pad, y_pad, key_mask

def list_and_cleanup(keep_last_n=3):
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch") and f.endswith(".pth")]
    def epnum(fn):
        m = re.search(r"model_epoch(\d+)\.pth", fn); return int(m.group(1)) if m else -1
    files.sort(key=epnum)
    if len(files) > keep_last_n:
        for f in files[:-keep_last_n]:
            os.remove(os.path.join(CHECKPOINT_DIR, f))
            print(f"🗑️ Deleted {f}")

def train(epochs=10, batch_size=16, lr=1e-4):
    tk = BlueprintTokenizer()
    vocab_size = tk.get_vocab_size()

    train_path = os.path.join(repo_root, "dataset", "train.jsonl")
    val_path = os.path.join(repo_root, "dataset", "val.jsonl")

    train_set = PairDataset(train_path, tk)
    val_set = PairDataset(val_path, tk)

    if len(train_set) == 0:
        print("⚠️ No training data found. Run: python dataset/generate_dataset.py && python scripts/build_jsonl.py")
        return

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LayoutTransformer(vocab_size=vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)

    if os.path.exists(CKPT_LATEST):
        print("🔁 Resuming from latest checkpoint...")
        model.load_state_dict(torch.load(CKPT_LATEST, map_location="cpu"))
        if os.path.exists(OPT_LATEST):
            opt.load_state_dict(torch.load(OPT_LATEST, map_location="cpu"))

    def run_epoch(loader, train_mode=True):
        model.train(train_mode)
        total = 0.0
        for x, y, mask in loader:
            logits = model(x, key_padding_mask=mask)
            loss = crit(logits.reshape(-1, vocab_size), y.reshape(-1))
            if train_mode:
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()
        return total / max(1, len(loader))

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        for ep in range(epochs):
            tr = run_epoch(train_loader, True)
            va = run_epoch(val_loader, False) if len(val_set)>0 else float('nan')
            print(f"✅ Epoch {ep}: train={tr:.4f}  val={va:.4f}")
            log.write(f"epoch {ep}, train {tr:.4f}, val {va:.4f}\n")
            list_and_cleanup(keep_last_n=3)
            torch.save(model.state_dict(), CKPT_LATEST)
            torch.save(opt.state_dict(), OPT_LATEST)
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_epoch{ep}.pth"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    train(args.epochs, args.batch, args.lr)
