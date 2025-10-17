import os, sys, json, argparse, torch, logging, re
from glob import glob
from typing import Optional
from pydantic import ValidationError
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, repo_root)

from models.layout_transformer import LayoutTransformer
from tokenizer.tokenizer import BlueprintTokenizer
from models.decoding import decode
from dataset.render_svg import render_layout_svg
from evaluation.validators import (
    enforce_min_separation,
    clamp_bounds,
    validate_layout,
    check_adjacency,
    resolve_overlaps,
)
from evaluation.evaluate_sample import assert_room_counts, BoundaryViolationError
from Generate.params import Params
from evaluation.refinement import refine_layout

log = logging.getLogger(__name__)

CKPT = os.path.join(repo_root, "checkpoints", "model_latest.pth")


def resolve_checkpoint_path(requested_path: Optional[str]) -> str:
    """Pick the checkpoint file to load, preferring explicit paths, then model_latest, then latest epoch."""
    if requested_path:
        if os.path.exists(requested_path):
            return requested_path
        raise FileNotFoundError(f"Requested checkpoint {requested_path} not found.")

    if os.path.exists(CKPT):
        return CKPT

    epoch_pattern = os.path.join(repo_root, "checkpoints", "epoch_*.pt")
    candidates = []
    for path in glob(epoch_pattern):
        match = re.search(r"epoch_(\d+)\.pt$", os.path.basename(path))
        if not match:
            continue
        candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError("Checkpoint not found. Train first (checkpoints/model_latest.pth).")

    candidates.sort()
    latest_path = candidates[-1][1]
    log.info("model_latest.pth missing; using most recent epoch checkpoint %s", os.path.basename(latest_path))
    return latest_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="generated_blueprint")
    ap.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file. Defaults to checkpoints/model_latest.pth or newest epoch_*.pt",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "sample", "beam", "guided"],
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
    ap.add_argument(
        "--debug_dump",
        type=str,
        default=None,
        help="Directory to write intermediate layouts for debugging",
    )
    ap.add_argument(
        "--guided_topk",
        type=int,
        default=8,
        help="Top-K expansion per step when using guided decoding",
    )
    ap.add_argument(
        "--refine_iters",
        type=int,
        default=0,
        help="Number of refinement iterations to run post decoding (0 disables)",
    )
    ap.add_argument(
        "--refine_temp",
        type=float,
        default=5.0,
        help="Initial temperature for refinement search",
    )
    ap.add_argument(
        "--issues_log",
        type=str,
        default=None,
        help="Optional path to append validation issues encountered during decoding",
    )
    ap.add_argument(
        "--guided_beam",
        type=int,
        default=8,
        help="Beam width when using guided decoding",
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

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt_blob = torch.load(ckpt_path, map_location=args.device)
    state_dict = ckpt_blob["model"] if isinstance(ckpt_blob, dict) and "model" in ckpt_blob else ckpt_blob
    model.load_state_dict(state_dict)
    model.to(args.device)

    if ckpt_path != CKPT and not args.checkpoint:
        try:
            torch.save(state_dict, CKPT)
            log.info("Cached weights to %s for subsequent runs", CKPT)
        except OSError as exc:
            log.warning("Failed to cache weights to %s: %s", CKPT, exc)


    debug_dump_dir = getattr(args, "debug_dump", None)
    if debug_dump_dir:
        try:
            os.makedirs(debug_dump_dir, exist_ok=True)
        except OSError as exc:
            log.warning("Failed to ensure debug dump directory %s: %s", debug_dump_dir, exc)
            debug_dump_dir = None

    def dump_layout(layout_dict, label):
        if not debug_dump_dir:
            return
        filename = os.path.join(debug_dump_dir, f"{args.out_prefix}_{label}.json")
        try:
            with open(filename, "w", encoding="utf-8") as fh:
                json.dump(layout_dict, fh, indent=2)
        except OSError as exc:
            log.warning("Failed to write debug dump %s: %s", filename, exc)

    issues_log_path = args.issues_log

    def record_issues(stage, issues, attempt=None, extra=None):
        if not issues_log_path or not issues:
            return
        entry = {
            "stage": stage,
            "attempt": attempt,
            "issues": issues,
            "extra": extra or {},
        }
        try:
            with open(issues_log_path, "a", encoding="utf-8") as fh:
                json.dump(entry, fh)
                fh.write("\n")
        except OSError as exc:
            log.warning("Failed to append issues to %s: %s", issues_log_path, exc)

    prefix = tk.encode_params(params.model_dump())
    adjacency = params.adjacency.root if params.adjacency else None
    adjacency_requirements = tk.adjacency_requirements_from_params(adjacency)

    def has_overlap(issue_list):
        return any("overlap" in msg.lower() for msg in (issue_list or []))

    def collect_adjacency_issues(layout_dict):
        if not adjacency:
            return []
        rooms = (layout_dict.get("layout") or {}).get("rooms", [])
        return check_adjacency(rooms, adjacency)

    def partial_validator(layout_dict):
        rooms = (layout_dict.get("layout") or {}).get("rooms", [])
        if not rooms:
            return []
        return validate_layout(
            layout_dict,
            max_width=max_w,
            max_length=max_h,
            min_separation=args.min_separation,
            adjacency=adjacency,
            require_connectivity=False,
        )
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

    decode_kwargs = {
        "max_len": 160,
        "strategy": args.strategy,
        "temperature": args.temperature,
        "beam_size": args.beam_size,
        "required_counts": room_counts,
        "bias_tokens": bias_tokens,
        "tokenizer": tk,
        "max_width": max_w,
        "max_length": max_h,
        "adjacency_requirements": adjacency_requirements,
    }
    if args.strategy == "guided":
        decode_kwargs.update(
            {
                "constraint_validator": partial_validator,
                "validator_min_rooms": 1,
                "guided_top_k": args.guided_topk,
                "guided_beam_size": args.guided_beam,
            }
        )

    log.info(
        "Starting generation using strategy=%s, guided_topk=%s, guided_beam=%s, refine_iters=%s",
        args.strategy,
        decode_kwargs.get("guided_top_k"),
        decode_kwargs.get("guided_beam_size"),
        args.refine_iters,
    )

    max_attempts = args.max_attempts
    layout_json = None
    missing = []
    for attempt in range(max_attempts):
        overlap_fix_used = False
        log.info("Attempt %s/%s: decoding layout", attempt + 1, max_attempts)
        layout_tokens = decode(
            model,
            prefix,
            **decode_kwargs,
        )
        layout_json = tk.decode_layout_tokens(layout_tokens)
        dump_layout(layout_json, f"attempt{attempt + 1}_raw")

        adjacency_issues = collect_adjacency_issues(layout_json)
        issues = validate_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            min_separation=0,
            adjacency=adjacency,
        )
        if issues:
            log.info(
                "Attempt %s: initial validation found %s issues",
                attempt + 1,
                len(issues),
            )
        record_issues("initial_validation", issues, attempt=attempt + 1)
        if issues and not overlap_fix_used and has_overlap(issues):
            layout_json = resolve_overlaps(layout_json, adjacency=adjacency)
            dump_layout(layout_json, f"attempt{attempt + 1}_overlap_fix")
            overlap_fix_used = True
            adjacency_issues = collect_adjacency_issues(layout_json)
            issues = validate_layout(
                layout_json,
                max_width=max_w,
                max_length=max_h,
                min_separation=0,
                adjacency=adjacency,
            )
            record_issues("post_overlap_fix", issues, attempt=attempt + 1)
            if issues:
                if attempt < max_attempts - 1:
                    print(
                        f"Layout validation failed: {'; '.join(issues)}. Regenerating...",
                        file=sys.stderr,
                    )
                    log.info(
                        "Attempt %s: validation failed, requesting regeneration (%s issues)",
                        attempt + 1,
                        len(issues),
                    )
                    record_issues("regen_after_validation", issues, attempt=attempt + 1)
                    continue
            if adjacency_issues:
                dump_layout(layout_json, f"attempt{attempt + 1}_failed")
                record_issues("adjacency_failure", adjacency_issues, attempt=attempt + 1)
                raise BoundaryViolationError(
                    "Layout validation failed: " + "; ".join(issues)
                )
            layout_json = clamp_bounds(layout_json, max_w, max_h)
            dump_layout(layout_json, f"attempt{attempt + 1}_clamped")
            issues = validate_layout(
                layout_json,
                max_width=max_w,
                max_length=max_h,
                min_separation=0,
                adjacency=adjacency,
            )
            record_issues("post_clamp_validation", issues, attempt=attempt + 1)
            if issues:
                dump_layout(layout_json, f"attempt{attempt + 1}_failed")
                record_issues("final_failure_before_separation", issues, attempt=attempt + 1)
                raise BoundaryViolationError(
                    "Layout validation failed after clamping: " + "; ".join(issues)
                )

        if args.min_separation > 0:
            layout_json = enforce_min_separation(
                layout_json, args.min_separation, adjacency=adjacency
            )
            dump_layout(layout_json, f"attempt{attempt + 1}_post_sep")
            adjacency_issues = collect_adjacency_issues(layout_json)
            issues = validate_layout(
                layout_json,
                max_width=max_w,
                max_length=max_h,
                min_separation=args.min_separation,
                adjacency=adjacency,
            )
            record_issues("post_separation_validation", issues, attempt=attempt + 1)
            if issues and not overlap_fix_used and has_overlap(issues):
                layout_json = resolve_overlaps(layout_json, adjacency=adjacency)
                dump_layout(layout_json, f"attempt{attempt + 1}_overlap_fix")
                overlap_fix_used = True
                adjacency_issues = collect_adjacency_issues(layout_json)
                issues = validate_layout(
                    layout_json,
                    max_width=max_w,
                    max_length=max_h,
                    min_separation=args.min_separation,
                    adjacency=adjacency,
                )
                record_issues("post_separation_overlap_fix", issues, attempt=attempt + 1)
            if issues:
                if attempt < max_attempts - 1:
                    print(
                        f"Layout validation failed: {'; '.join(issues)}. Regenerating...",
                        file=sys.stderr,
                    )
                    record_issues("regen_after_separation", issues, attempt=attempt + 1)
                    continue
                if adjacency_issues:
                    dump_layout(layout_json, f"attempt{attempt + 1}_failed")
                    record_issues("adjacency_failure_after_separation", adjacency_issues, attempt=attempt + 1)
                    raise BoundaryViolationError(
                        "Layout validation failed: " + "; ".join(issues)
                    )
                layout_json = clamp_bounds(layout_json, max_w, max_h)
                dump_layout(layout_json, f"attempt{attempt + 1}_clamped")
                issues = validate_layout(
                    layout_json,
                    max_width=max_w,
                    max_length=max_h,
                    min_separation=args.min_separation,
                    adjacency=adjacency,
                )
                record_issues("post_separation_clamp_validation", issues, attempt=attempt + 1)
                if issues:
                    dump_layout(layout_json, f"attempt{attempt + 1}_failed")
                    record_issues("final_failure_after_separation", issues, attempt=attempt + 1)
                    raise BoundaryViolationError(
                        "Layout validation failed after clamping: "
                        + "; ".join(issues)
                    )

        missing = assert_room_counts(layout_json, raw)
        if not missing:
            log.info("Attempt %s succeeded; layout satisfies room counts", attempt + 1)
            break
        if attempt < max_attempts - 1:
            print(
                f"Missing rooms { [m['room_type'] for m in missing] }, regenerating...",
                file=sys.stderr,
            )
            record_issues(
                "missing_rooms",
                [f"{m['room_type']} expected {m['expected']} found {m['found']}" for m in missing],
                attempt=attempt + 1,
            )
            log.info(
                "Attempt %s: missing rooms detected (%s). Retrying.",
                attempt + 1,
                ", ".join(m['room_type'] for m in missing),
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
        dump_layout(layout_json, "placeholders_clamped")
        issues = validate_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            min_separation=args.min_separation,
            adjacency=adjacency,
        )
        if issues:
            dump_layout(layout_json, "placeholders_failed")
            record_issues("placeholders_failure", issues)
            raise BoundaryViolationError(
                "Layout validation failed after injecting placeholders: "
                + "; ".join(issues)
            )
        break

    json_path = f"{args.out_prefix}.json"
    svg_path = f"{args.out_prefix}.svg"
    layout_json = clamp_bounds(layout_json, max_w, max_h)
    dump_layout(layout_json, "final")
    if args.refine_iters > 0:
        log.info(
            "Starting refinement for %s iterations (temperature %.2f)",
            args.refine_iters,
            args.refine_temp,
        )
        refined_layout = refine_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            min_separation=args.min_separation,
            adjacency=adjacency,
            iterations=args.refine_iters,
            temperature=args.refine_temp,
        )
        refined_layout = clamp_bounds(refined_layout, max_w, max_h)
        refined_issues = validate_layout(
            refined_layout,
            max_width=max_w,
            max_length=max_h,
            min_separation=args.min_separation,
            adjacency=adjacency,
        )
        if not refined_issues:
            layout_json = refined_layout
            dump_layout(layout_json, "refined")
            log.info("Refinement completed without issues; refined layout adopted")
        else:
            dump_layout(refined_layout, "refined_failed")
            record_issues("refinement_failure", refined_issues)
            log.warning(
                "Refinement produced a layout with issues: %s. Keeping pre-refined layout.",
                "; ".join(refined_issues),
            )

    final_issues = validate_layout(
        layout_json,
        max_width=max_w,
        max_length=max_h,
        min_separation=args.min_separation,
        adjacency=adjacency,
    )
    if final_issues:
        raise BoundaryViolationError(
            "Layout validation failed before writing output: "
            + "; ".join(final_issues)
        )

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
