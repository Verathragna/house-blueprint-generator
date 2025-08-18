import os, sys, json, argparse, torch
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, repo_root)

from models.layout_transformer import LayoutTransformer
from tokenizer.tokenizer import BlueprintTokenizer, BOS_ID, SEP_ID, EOS_ID
from dataset.render_svg import render_layout_svg

CKPT = os.path.join(repo_root, "checkpoints", "model_latest.pth")

def greedy_decode(model, prefix_ids, max_len=160):
    model.eval()
    seq = prefix_ids[:]
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)[:, -1, :]
            nxt = int(logits.argmax(dim=-1))
            seq.append(nxt)
            if nxt == EOS_ID: break
    if SEP_ID in seq:
        i = seq.index(SEP_ID) + 1
        return seq[i:]
    return seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="generated_blueprint")
    args = ap.parse_args()

    params = json.load(open(args.params_json, "r", encoding="utf-8"))
    tk = BlueprintTokenizer()
    model = LayoutTransformer(tk.get_vocab_size())
    if not os.path.exists(CKPT):
        raise FileNotFoundError("Checkpoint not found. Train first (checkpoints/model_latest.pth).")
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))

    prefix = tk.encode_params(params)
    layout_tokens = greedy_decode(model, prefix, max_len=160)
    layout_json = tk.decode_layout_tokens(layout_tokens)

    json_path = f"{args.out_prefix}.json"
    svg_path = f"{args.out_prefix}.svg"
    json.dump(layout_json, open(json_path, "w", encoding="utf-8"), indent=2)
    from dataset.render_svg import render_layout_svg
    render_layout_svg(layout_json, svg_path)
    print(f"✅ Wrote {json_path} and {svg_path}")

if __name__ == "__main__":
    main()
