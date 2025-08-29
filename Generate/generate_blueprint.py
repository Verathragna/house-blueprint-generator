import os, sys, json, argparse, torch, logging
from pydantic import ValidationError
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, repo_root)

from models.layout_transformer import LayoutTransformer
from tokenizer.tokenizer import BlueprintTokenizer
from models.decoding import decode
from dataset.render_svg import render_layout_svg
from evaluation.validators import enforce_min_separation, clamp_bounds, validate_layout
from evaluation.evaluate_sample import assert_room_counts, BoundaryViolationError
from Generate.params import Params

log = logging.getLogger(__name__)

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
    ap.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Maximum decode retries before clamping",
    )
    args = ap.parse_args()

    try:
        with open(args.params_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.error("Failed to read params file %s: %s", args.params_json, e)
        sys.exit(1)

    try:
        params = Params.model_validate(raw)
    except ValidationError as e:
        log.error("Invalid parameters: %s", e)
        sys.exit(1)

    tk = BlueprintTokenizer()
    model = LayoutTransformer(tk.get_vocab_size())
    if not os.path.exists(CKPT):
        raise FileNotFoundError("Checkpoint not found. Train first (checkpoints/model_latest.pth).")
    ckpt = torch.load(CKPT, map_location=args.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(args.device)

    prefix = tk.encode_params(params.model_dump())
    adjacency = params.adjacency.root if params.adjacency else None
    room_counts = {}
    bias_tokens = {}
    if "bedrooms" in raw:
        room_counts[tk.token_to_id["BEDROOM"]] = params.bedrooms
    if "bathrooms" in raw:
        room_counts[tk.token_to_id["BATHROOM"]] = params.bathrooms.full + params.bathrooms.half
    # Core rooms are always required at least once
    room_counts[tk.token_to_id["KITCHEN"]] = params.kitchen
    room_counts[tk.token_to_id["LIVING"]] = params.livingRooms
    room_counts[tk.token_to_id["DINING"]] = params.diningRooms
    room_counts[tk.token_to_id["LAUNDRY"]] = params.laundryRooms
    if raw.get("garage"):
        room_counts[tk.token_to_id["GARAGE"]] = 1
        bias_tokens[tk.token_to_id["GARAGE"]] = 2.0

    dims = raw.get("dimensions") or {}
    max_w = float(dims.get("width", 40))
    max_h = float(dims.get("depth", dims.get("height", 40)))

    max_attempts = args.max_attempts
    layout_json = None
    missing = []
    for attempt in range(max_attempts):
        layout_tokens = decode(
            model,
            prefix,
            max_len=160,
            strategy=args.strategy,
            temperature=args.temperature,
            beam_size=args.beam_size,
            required_counts=room_counts,
            bias_tokens=bias_tokens,
            tokenizer=tk,
            max_width=max_w,
            max_length=max_h,
        )
        layout_json = tk.decode_layout_tokens(layout_tokens)

        issues = validate_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            min_separation=0,
            adjacency=adjacency,
        )
        if issues:
            if attempt < max_attempts - 1:
                print(
                    f"Layout validation failed: {'; '.join(issues)}. Regenerating...",
                    file=sys.stderr,
                )
                continue
            layout_json = clamp_bounds(layout_json, max_w, max_h)
            issues = validate_layout(
                layout_json,
                max_width=max_w,
                max_length=max_h,
                min_separation=0,
                adjacency=adjacency,
            )
            if issues:
                raise BoundaryViolationError(
                    "Layout validation failed after clamping: " + "; ".join(issues)
                )

        if args.min_separation > 0:
            layout_json = enforce_min_separation(
                layout_json, args.min_separation, adjacency=adjacency
            )
            issues = validate_layout(
                layout_json,
                max_width=max_w,
                max_length=max_h,
                min_separation=args.min_separation,
                adjacency=adjacency,
            )
            if issues:
                if attempt < max_attempts - 1:
                    print(
                        f"Layout validation failed: {'; '.join(issues)}. Regenerating...",
                        file=sys.stderr,
                    )
                    continue
                layout_json = clamp_bounds(layout_json, max_w, max_h)
                issues = validate_layout(
                    layout_json,
                    max_width=max_w,
                    max_length=max_h,
                    min_separation=args.min_separation,
                    adjacency=adjacency,
                )
                if issues:
                    raise BoundaryViolationError(
                        "Layout validation failed after clamping: "
                        + "; ".join(issues)
                    )

        missing = assert_room_counts(layout_json, raw)
        if not missing:
            break
        if attempt < max_attempts - 1:
            print(
                f"Missing rooms { [m['room_type'] for m in missing] }, regenerating...",
                file=sys.stderr,
            )
            continue
        # On final attempt, inject placeholder rooms
        rooms = layout_json.setdefault("layout", {}).setdefault("rooms", [])
        for miss in missing:
            token_key = miss["room_type"].upper()
            wtok, ltok = tk.default_room_dims.get(token_key, ("W12", "L12"))
            rooms.append(
                {
                    "type": miss["room_type"].capitalize(),
                    "position": {"x": 0, "y": 0},
                    "size": {"width": int(wtok[1:]), "length": int(ltok[1:])},
                }
            )
        layout_json = clamp_bounds(layout_json, max_w, max_h)
        issues = validate_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            min_separation=args.min_separation,
            adjacency=adjacency,
        )
        if issues:
            raise BoundaryViolationError(
                "Layout validation failed after injecting placeholders: "
                + "; ".join(issues)
            )
        break

    json_path = f"{args.out_prefix}.json"
    svg_path = f"{args.out_prefix}.svg"
    layout_json = clamp_bounds(layout_json, max_w, max_h)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(layout_json, f, indent=2)
    except OSError as e:
        log.error("Failed to write layout JSON to %s: %s", json_path, e)
        sys.exit(1)
    try:
        render_layout_svg(layout_json, svg_path, lot_dims=(max_w, max_h))
    except OSError as e:
        log.error("Failed to write SVG to %s: %s", svg_path, e)
        sys.exit(1)
    print(f"Wrote {json_path} and {svg_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except BoundaryViolationError as exc:
        log.error("Boundary violation: %s", exc)
        sys.exit(1)
