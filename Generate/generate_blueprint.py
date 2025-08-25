import os, sys, json, argparse, torch
from pydantic import ValidationError
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, repo_root)

from models.layout_transformer import LayoutTransformer
from tokenizer.tokenizer import BlueprintTokenizer
from models.decoding import decode
from dataset.render_svg import render_layout_svg
from evaluation.validators import enforce_min_separation
from Generate.params import Params

CKPT = os.path.join(repo_root, "checkpoints", "model_latest.pth")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="generated_blueprint")
    ap.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    ap.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "sample", "beam"],
        help="Decoding strategy",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument(
        "--min_separation",
        type=float,
        default=1.0,
        help="Minimum room separation; 0 disables post-processing",
    )
    args = ap.parse_args()

    try:
        raw = json.load(open(args.params_json, "r", encoding="utf-8"))
        params = Params.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Invalid parameters: {e}", file=sys.stderr)
        sys.exit(1)

    tk = BlueprintTokenizer()
    model = LayoutTransformer(tk.get_vocab_size())
    if not os.path.exists(CKPT):
        raise FileNotFoundError("Checkpoint not found. Train first (checkpoints/model_latest.pth).")
    ckpt = torch.load(CKPT, map_location=args.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(args.device)

    prefix = tk.encode_params(params.model_dump())
    layout_tokens = decode(
        model,
        prefix,
        max_len=160,
        strategy=args.strategy,
        temperature=args.temperature,
        beam_size=args.beam_size,
    )
    layout_json = tk.decode_layout_tokens(layout_tokens)
    if args.min_separation > 0:
        layout_json = enforce_min_separation(layout_json, args.min_separation)

    json_path = f"{args.out_prefix}.json"
    svg_path = f"{args.out_prefix}.svg"
    json.dump(layout_json, open(json_path, "w", encoding="utf-8"), indent=2)
    render_layout_svg(layout_json, svg_path)
    print(f"Wrote {json_path} and {svg_path}")

if __name__ == "__main__":
    main()
