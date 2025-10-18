import os, sys, json, argparse, re, torch, subprocess, math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, repo_root)

from models.layout_transformer import LayoutTransformer
from tokenizer.tokenizer import BlueprintTokenizer, PAD_ID

CHECKPOINT_DIR = os.path.join(repo_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LOG_FILE = os.path.join(repo_root, "training_log.txt")

class PairDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: BlueprintTokenizer):
        self.rows = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")] if os.path.exists(jsonl_path) else []
        self.tk = tokenizer

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        if "x" in row and "y" in row:
            return (
                torch.tensor(row["x"], dtype=torch.long),
                torch.tensor(row["y"], dtype=torch.long),
            )

        # ensure coordinate fields are present so position tokens can be learned
        for idx, room in enumerate(
            row.get("layout", {}).get("layout", {}).get("rooms", [])
        ):
            pos = room.get("position") or {}
            if "x" not in pos or "y" not in pos:
                raise ValueError(f"Room {idx} missing x or y position")
        x_ids, y_ids = self.tk.build_training_pair(row["params"], row["layout"])
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=PAD_ID)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=PAD_ID)
    key_mask = (x_pad == PAD_ID)
    return x_pad, y_pad, key_mask

def list_and_cleanup(keep_last_n=3):
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("epoch_") and f.endswith(".pt")]
    def epnum(fn):
        m = re.search(r"epoch_(\d+)\.pt", fn)
        return int(m.group(1)) if m else -1
    files.sort(key=epnum)
    if len(files) > keep_last_n:
        for f in files[:-keep_last_n]:
            os.remove(os.path.join(CHECKPOINT_DIR, f))
            print(f"Deleted {f}")


def ensure_dataset(train_path: str, val_path: str) -> bool:
    """Generate training data if missing."""
    if os.path.exists(train_path) and os.path.exists(val_path):
        return True

    print("No training data found. Generating synthetic dataset...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dataset.generate_dataset",
                "--augment",
                "--patterns",
                "chain,corridor,l_shape,central_corridor",
            ],
            check=True,
            cwd=repo_root,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.build_jsonl",
                "--augment",
            ],
            check=True,
            cwd=repo_root,
        )
    except subprocess.CalledProcessError as e:  # pragma: no cover - best effort generation
        cmd = e.cmd if isinstance(e.cmd, str) else " ".join(e.cmd)
        print(f"Failed to run '{cmd}' (exit code {e.returncode})")
        return False
    except OSError as e:  # pragma: no cover - missing executable
        print(f"Execution failed: {e}")
        return False
    return os.path.exists(train_path) and os.path.exists(val_path)

def train(
    epochs=20,
    batch_size=16,
    lr=2e-4,
    layers=6,
    hidden_size=256,
    device="cpu",
    resume=None,
    save_weights=False,
):
    tk = BlueprintTokenizer()
    vocab_size = tk.get_vocab_size()

    train_path = os.path.join(repo_root, "dataset", "train.jsonl")
    val_path = os.path.join(repo_root, "dataset", "val.jsonl")

    if not ensure_dataset(train_path, val_path):
        print("Dataset generation failed. Aborting training.")
        return

    train_set = PairDataset(train_path, tk)
    val_set = PairDataset(val_path, tk)

    if len(train_set) == 0:
        print("Dataset generation produced no data. Aborting training.")
        return

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(device)
    model = LayoutTransformer(
        vocab_size=vocab_size,
        d_model=hidden_size,
        num_layers=layers,
        dim_ff=hidden_size * 4,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    
    # Add constraint-aware loss
    from training.constraint_losses import ConstraintLoss
    constraint_loss = ConstraintLoss(
        area_weight=1.0,
        overlap_weight=2.0,
        boundary_weight=1.5,
        count_weight=0.5
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and str(device) != "cpu")

    start_epoch = 0
    if resume and os.path.exists(resume):
        print(f"Resuming from checkpoint {resume}...")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        model.to(device)
        if "optimizer" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        start_epoch = checkpoint.get("epoch", -1) + 1

    # LR scheduler: linear warmup then cosine decay across all steps
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    global_step = 0

    def run_epoch(loader, train_mode=True):
        nonlocal global_step
        model.train(train_mode)
        total = 0.0
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x, key_padding_mask=mask)
                base_loss = crit(logits.reshape(-1, vocab_size), y.reshape(-1))
                
                # Add constraint loss during training
                if train_mode:
                    loss, loss_dict = constraint_loss(logits, y, tk, base_loss)
                else:
                    loss = base_loss
            if train_mode:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                global_step += 1
            total += loss.item()
        return total / max(1, len(loader))

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        for ep in range(start_epoch, start_epoch + epochs):
            tr = run_epoch(train_loader, True)
            va = run_epoch(val_loader, False) if len(val_set) > 0 else float('nan')
            
            # Log with constraint loss information if available
            if hasattr(run_epoch, 'last_constraint_loss'):
                constraint_avg = getattr(run_epoch, 'last_constraint_loss', 0.0)
                print(f"Epoch {ep}: train={tr:.4f} (constraint={constraint_avg:.4f}) val={va:.4f}")
                log.write(f"epoch {ep}, train {tr:.4f}, constraint {constraint_avg:.4f}, val {va:.4f}\n")
            else:
                print(f"Epoch {ep}: train={tr:.4f}  val={va:.4f}")
                log.write(f"epoch {ep}, train {tr:.4f}, val {va:.4f}\n")
            log.flush()
            list_and_cleanup(keep_last_n=3)
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{ep}.pt")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep}, ckpt_path)
            latest_weights_path = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
            torch.save(model.state_dict(), latest_weights_path)
            if save_weights:
                torch.save(
                    model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, f"model_epoch_{ep}_weights.pth"),
                )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    ap.add_argument("--save_weights", action="store_true", help="also save per-epoch raw weight files")
    args = ap.parse_args()
    train(
        args.epochs,
        args.batch,
        args.learning_rate,
        args.layers,
        args.hidden_size,
        args.device,
        args.resume,
        args.save_weights,
    )
