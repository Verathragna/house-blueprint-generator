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
    pack_layout,
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
        default="guided",
        choices=["greedy", "sample", "beam", "guided"],
        help="Decoding strategy",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument(
        "--min_separation",
        type=float,
        default=0.5,  # Reduced from 1.0 for better convergence
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

    # Load checkpoint first so we can configure the model to match its dimensions
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt_blob = torch.load(ckpt_path, map_location=args.device)
    state_dict = ckpt_blob["model"] if isinstance(ckpt_blob, dict) and "model" in ckpt_blob else ckpt_blob

    # Infer transformer dims from checkpoint to avoid size mismatches
    def _infer_d_model(sd):
        w = sd.get("embed.weight")
        return int(w.shape[1]) if w is not None else 128

    def _infer_dim_ff(sd):
        w = sd.get("encoder.layers.0.linear1.weight")
        return int(w.shape[0]) if w is not None else 4 * _infer_d_model(sd)

    def _infer_num_layers(sd):
        layers = [k for k in sd.keys() if k.startswith("encoder.layers.")]
        if not layers:
            return 4
        # keys like encoder.layers.{i}.something -> get max i + 1
        try:
            idxs = set(int(k.split(".")[2]) for k in layers)
            return max(idxs) + 1
        except Exception:
            return 4

    d_model = _infer_d_model(state_dict)
    dim_ff = _infer_dim_ff(state_dict)
    num_layers = _infer_num_layers(state_dict)

    # Use a safe default nhead that divides d_model (prefer 8; fallback to 4)
    nhead = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)

    model = LayoutTransformer(
        tk.get_vocab_size(),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
    )

    model.load_state_dict(state_dict, strict=False)
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

    def issues_are_separation_only(issues_list):
        if not issues_list:
            return False
        lowered = [str(i).lower() for i in issues_list]
        return all(("within" in i and "overlap" not in i and "exceed" not in i and "not connected" not in i and "must share a wall" not in i) for i in lowered)

    def prune_excess_rooms(layout_dict, max_counts_by_type):
        rooms = (layout_dict.get("layout") or {}).get("rooms", [])
        if not rooms:
            return layout_dict
        kept = []
        counters = {k: 0 for k in max_counts_by_type.keys()}
        # Normalize keys to lowercase room type strings
        def norm(t):
            return (t or "").strip().lower()
        # Prefer to keep the first instances; simple and deterministic
        for room in rooms:
            t = norm(room.get("type"))
            # If this type has a cap, enforce it; otherwise drop the room
            if t in counters:
                if counters[t] < max_counts_by_type[t]:
                    counters[t] += 1
                    kept.append(room)
            # Unknown types are dropped
        layout_dict.setdefault("layout", {})["rooms"] = kept
        return layout_dict

    def shrink_to_fit(layout_dict, max_width, max_length, target_fill=0.8):
        rooms = (layout_dict.get("layout") or {}).get("rooms", [])
        if not rooms:
            return layout_dict
        lot_area = float(max_width) * float(max_length)
        total_area = 0.0
        sizes = []
        for room in rooms:
            size = room.get("size") or {}
            w = float(size.get("width", 0))
            l = float(size.get("length", 0))
            sizes.append((room, w, l))
            total_area += max(0.0, w) * max(0.0, l)
        if lot_area <= 0:
            return layout_dict
        max_allowed = max(0.1, target_fill) * lot_area
        if total_area <= max_allowed:
            return layout_dict
        scale = (max_allowed / max(1e-6, total_area)) ** 0.5
        # Apply scaling with a minimum practical room size
        for room, w, l in sizes:
            new_w = max(4, int(round(w * scale)))
            new_l = max(4, int(round(l * scale)))
            room.setdefault("size", {})["width"] = new_w
            room["size"]["length"] = new_l
        return layout_dict

    def build_sep_exempt_adjacency(layout_dict):
        """Union of requested adjacency with currently observed wall-sharing pairs.
        This is used to exempt touching rooms from separation penalties while still
        enforcing required adjacencies.
        """
        rooms = (layout_dict.get("layout") or {}).get("rooms", [])
        # start with requested adjacency
        merged = {k: list(v) for k, v in ((adjacency or {}) or {}).items()}
        # compute observed
        def _bounds(r):
            pos = r.get("position") or {}
            size = r.get("size") or {}
            x, y = float(pos.get("x", 0)), float(pos.get("y", 0))
            w, l = float(size.get("width", 0)), float(size.get("length", 0))
            return x, y, x + w, y + l
        def _share_wall(a, b):
            ax1, ay1, ax2, ay2 = _bounds(a)
            bx1, by1, bx2, by2 = _bounds(b)
            vertical_touch = (abs(ax2 - bx1) < 1e-6 or abs(bx2 - ax1) < 1e-6) and (min(ay2, by2) - max(ay1, by1) > 0)
            horizontal_touch = (abs(ay2 - by1) < 1e-6 or abs(by2 - ay1) < 1e-6) and (min(ax2, bx2) - max(ax1, bx1) > 0)
            return vertical_touch or horizontal_touch
        for i, r1 in enumerate(rooms):
            t1 = (r1.get("type") or "").strip()
            if not t1:
                continue
            for j in range(i + 1, len(rooms)):
                r2 = rooms[j]
                t2 = (r2.get("type") or "").strip()
                if not t2:
                    continue
                if _share_wall(r1, r2):
                    merged.setdefault(t1, [])
                    merged.setdefault(t2, [])
                    if t2 not in merged[t1]:
                        merged[t1].append(t2)
                    if t1 not in merged[t2]:
                        merged[t2].append(t1)
        return merged

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
    # Also build a map of maximum allowed counts per normalized type string for pruning
    max_counts_by_type = {}
    if "bedrooms" in raw:
        room_counts[tk.token_to_id["BEDROOM"]] = params.bedrooms
        max_counts_by_type["bedroom"] = int(params.bedrooms)
    if "bathrooms" in raw:
        room_counts[tk.token_to_id["BATHROOM"]] = params.bathrooms.full + params.bathrooms.half
        max_counts_by_type["bathroom"] = int(params.bathrooms.full + params.bathrooms.half)
    # Core rooms are always required at least once
    room_counts[tk.token_to_id["KITCHEN"]] = params.kitchen
    room_counts[tk.token_to_id["LIVING"]] = params.livingRooms
    room_counts[tk.token_to_id["DINING"]] = params.diningRooms
    room_counts[tk.token_to_id["LAUNDRY"]] = params.laundryRooms
    max_counts_by_type.update(
        {
            "kitchen": int(params.kitchen),
            "living room": int(params.livingRooms),
            "dining room": int(params.diningRooms),
            "laundry room": int(params.laundryRooms),
        }
    )
    if raw.get("garage"):
        room_counts[tk.token_to_id["GARAGE"]] = 1
        bias_tokens[tk.token_to_id["GARAGE"]] = 2.0
        max_counts_by_type["garage"] = 1

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

        # Prune any excess rooms beyond the requested counts to prevent
        # pathological overlap/crowding before validation.
        if max_counts_by_type:
            layout_json = prune_excess_rooms(layout_json, max_counts_by_type)
            dump_layout(layout_json, f"attempt{attempt + 1}_pruned")

        # Optionally shrink if rooms are too large to reasonably fit the lot, then pre-pack onto grid
        # Use more conservative target fill to leave more space for separation
        layout_json = shrink_to_fit(layout_json, max_w, max_h, target_fill=0.65)
        # Zoning + adjacency-aware packing for more realistic layouts
        default_hints = {
            "Bedroom": ["Bathroom", "Hallway"],
            "Bathroom": ["Bedroom"],
            "Kitchen": ["Dining Room", "Living Room", "Laundry Room"],
            "Dining Room": ["Kitchen", "Living Room"],
            "Living Room": ["Dining Room", "Kitchen"],
            "Garage": ["Laundry Room"],
            "Laundry Room": ["Kitchen", "Garage"],
        }
        layout_json = pack_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            grid=1.5,  # Use larger grid for more spacing
            adjacency_hints=default_hints,
            zoning=True,
            min_hall_width=4.0,  # Increase minimum hallway width
        )
        dump_layout(layout_json, f"attempt{attempt + 1}_prepacked")

        adjacency_issues = collect_adjacency_issues(layout_json)
        issues = validate_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            min_separation=args.min_separation,
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
            layout_json = resolve_overlaps(
                layout_json,
                adjacency=build_sep_exempt_adjacency(layout_json),
                max_width=max_w,
                max_length=max_h,
                max_iterations=10,  # Reduced from 20
                separation_iterations=150,  # Reduced from 300
            )
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
                min_separation=args.min_separation,
                adjacency=adjacency,
            )
            record_issues("post_clamp_validation", issues, attempt=attempt + 1)
            if issues:
                dump_layout(layout_json, f"attempt{attempt + 1}_failed")
                # NEW: If validation after clamping still shows overlaps, try to resolve them again.
                if has_overlap(issues):
                    log.warning("Overlaps detected after clamping. Attempting to resolve again.")
                    layout_json = resolve_overlaps(
                        layout_json,
                        adjacency=build_sep_exempt_adjacency(layout_json),
                        max_width=max_w,
                        max_length=max_h,
                        max_iterations=8,  # Reduced from 20
                        separation_iterations=100,  # Reduced from 300
                    )
                    dump_layout(layout_json, f"attempt{attempt + 1}_clamped_re_overlap_fix")
                    issues = validate_layout(
                        layout_json,
                        max_width=max_w,
                        max_length=max_h,
                        min_separation=args.min_separation,
                        adjacency=adjacency,
                    )
                    record_issues("post_clamp_re_overlap_fix_validation", issues, attempt=attempt + 1)
            if issues: # Check issues again after potential re-resolution
                    # Fallback: grid-pack layout to eliminate overlaps/connectivity issues
                    layout_packed = pack_layout(layout_json, max_width=max_w, max_length=max_h, grid=1.0)
                    dump_layout(layout_packed, f"attempt{attempt + 1}_packed")
                    # First, ensure the packed layout is free of hard geometry violations (bounds/overlaps)
                    issues_packed_geom = validate_layout(
                        layout_packed,
                        max_width=max_w,
                        max_length=max_h,
                        min_separation=0,
                        adjacency=None,
                        require_connectivity=False,
                    )
                    record_issues("post_pack_geometry_validation", issues_packed_geom, attempt=attempt + 1)
                    if issues_packed_geom:
                        # As a last resort, shrink to fit and re-pack once
                        layout_shrunk = shrink_to_fit(layout_json, max_w, max_h, target_fill=0.7)
                        layout_packed2 = pack_layout(layout_shrunk, max_width=max_w, max_length=max_h, grid=1.0)
                        dump_layout(layout_packed2, f"attempt{attempt + 1}_packed_shrunk")
                        issues_packed_geom2 = validate_layout(
                            layout_packed2,
                            max_width=max_w,
                            max_length=max_h,
                            min_separation=0,
                            adjacency=None,
                            require_connectivity=False,
                        )
                        record_issues("post_pack_shrunk_geometry_validation", issues_packed_geom2, attempt=attempt + 1)
                        if issues_packed_geom2:
                            dump_layout(layout_packed2, f"attempt{attempt + 1}_failed")
                            record_issues("final_failure_before_separation", issues_packed_geom2, attempt=attempt + 1)
                            raise BoundaryViolationError(
                                "Layout validation failed after clamping: " + "; ".join(issues_packed_geom2)
                            )
                        layout_packed = layout_packed2
                    # Then enforce separation and re-validate fully
                    layout_repaired = enforce_min_separation(
                        layout_packed,
                        args.min_separation,
                        adjacency=build_sep_exempt_adjacency(layout_packed),
                        max_width=max_w,
                        max_length=max_h,
                        max_iterations=100,  # Reduced from 300
                    )
                    layout_repaired = clamp_bounds(layout_repaired, max_w, max_h)
                    dump_layout(layout_repaired, f"attempt{attempt + 1}_packed_separated")
                    issues_repaired = validate_layout(
                        layout_repaired,
                        max_width=max_w,
                        max_length=max_h,
                        min_separation=args.min_separation,
                        adjacency=adjacency,
                    )
                    record_issues("post_pack_full_validation", issues_repaired, attempt=attempt + 1)
                    if issues_repaired:
                        if issues_are_separation_only(issues_repaired):
                            # Try one more shrink + re-pack to create spacing, then enforce separation again
                            layout_shrunk2 = shrink_to_fit(layout_repaired, max_w, max_h, target_fill=0.6)
                            layout_packed3 = pack_layout(layout_shrunk2, max_width=max_w, max_length=max_h, grid=1.0)
                            layout_packed3 = enforce_min_separation(
                                layout_packed3,
                                args.min_separation,
                                adjacency=build_sep_exempt_adjacency(layout_packed3),
                                max_width=max_w,
                                max_length=max_h,
                                max_iterations=80,  # Reduced from 300
                            )
                            layout_packed3 = clamp_bounds(layout_packed3, max_w, max_h)
                            dump_layout(layout_packed3, f"attempt{attempt + 1}_packed_separated_shrunk2")
                            issues_repaired2 = validate_layout(
                                layout_packed3,
                                max_width=max_w,
                                max_length=max_h,
                                min_separation=args.min_separation,
                                adjacency=adjacency,
                            )
                            record_issues("post_pack_full_validation_shrunk2", issues_repaired2, attempt=attempt + 1)
                            if not issues_repaired2:
                                layout_json = layout_packed3
                            elif issues_are_separation_only(issues_repaired2):
                                log.warning(
                                    "Proceeding despite separation-only issues after pack+separate: %s",
                                    "; ".join(issues_repaired2),
                                )
                                layout_json = layout_packed3
                            else:
                                dump_layout(layout_repaired, f"attempt{attempt + 1}_failed")
                                raise BoundaryViolationError(
                                    "Layout validation failed after clamping: " + "; ".join(issues_repaired2)
                                )
                        else:
                            # Non-separation issues remain; fail
                            dump_layout(layout_repaired, f"attempt{attempt + 1}_failed")
                            raise BoundaryViolationError(
                                "Layout validation failed after clamping: " + "; ".join(issues_repaired)
                            )
                    else:
                        layout_json = layout_repaired

        if args.min_separation > 0:
            layout_json = enforce_min_separation(
                layout_json,
                args.min_separation,
                adjacency=build_sep_exempt_adjacency(layout_json),
                max_width=max_w,
                max_length=max_h,
                max_iterations=80,  # Reduced from 200
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
                layout_json = resolve_overlaps(
                    layout_json,
                    adjacency=build_sep_exempt_adjacency(layout_json),
                    max_width=max_w,
                    max_length=max_h,
                    max_iterations=8,  # Reduced from 20
                    separation_iterations=100,  # Reduced from 300
                )
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
                    # Fallback: grid-pack then validate
                    layout_packed = pack_layout(layout_json, max_width=max_w, max_length=max_h, grid=1.0)
                    dump_layout(layout_packed, f"attempt{attempt + 1}_packed_after_sep")
                    # Ensure packed layout has no hard geometry violations first
                    issues_packed_geom = validate_layout(
                        layout_packed,
                        max_width=max_w,
                        max_length=max_h,
                        min_separation=0,
                        adjacency=None,
                        require_connectivity=False,
                    )
                    record_issues("post_separation_pack_geometry_validation", issues_packed_geom, attempt=attempt + 1)
                    if issues_packed_geom:
                        # As a last resort, shrink to fit and re-pack once
                        layout_shrunk = shrink_to_fit(layout_json, max_w, max_h, target_fill=0.7)
                        layout_packed2 = pack_layout(layout_shrunk, max_width=max_w, max_length=max_h, grid=1.0)
                        dump_layout(layout_packed2, f"attempt{attempt + 1}_packed_after_sep_shrunk")
                        issues_packed_geom2 = validate_layout(
                            layout_packed2,
                            max_width=max_w,
                            max_length=max_h,
                            min_separation=0,
                            adjacency=None,
                            require_connectivity=False,
                        )
                        record_issues("post_separation_pack_shrunk_geometry_validation", issues_packed_geom2, attempt=attempt + 1)
                        if issues_packed_geom2:
                            dump_layout(layout_packed2, f"attempt{attempt + 1}_failed")
                            record_issues("final_failure_after_separation", issues_packed_geom2, attempt=attempt + 1)
                            raise BoundaryViolationError(
                                "Layout validation failed after clamping: "
                                + "; ".join(issues_packed_geom2)
                            )
                        layout_packed = layout_packed2
                    # Then enforce separation and validate fully
                    layout_repaired = enforce_min_separation(
                        layout_packed,
                        args.min_separation,
                        adjacency=build_sep_exempt_adjacency(layout_packed),
                        max_width=max_w,
                        max_length=max_h,
                        max_iterations=300,
                    )
                    layout_repaired = clamp_bounds(layout_repaired, max_w, max_h)
                    dump_layout(layout_repaired, f"attempt{attempt + 1}_packed_after_sep_separated")
                    issues_repaired = validate_layout(
                        layout_repaired,
                        max_width=max_w,
                        max_length=max_h,
                        min_separation=args.min_separation,
                        adjacency=adjacency,
                    )
                    record_issues("post_separation_pack_full_validation", issues_repaired, attempt=attempt + 1)
                    if issues_repaired:
                        if issues_are_separation_only(issues_repaired):
                            # Try one more shrink + re-pack to create spacing, then enforce separation again
                            layout_shrunk2 = shrink_to_fit(layout_repaired, max_w, max_h, target_fill=0.6)
                            layout_packed3 = pack_layout(layout_shrunk2, max_width=max_w, max_length=max_h, grid=1.0)
                            layout_packed3 = enforce_min_separation(
                                layout_packed3,
                                args.min_separation,
                                adjacency=build_sep_exempt_adjacency(layout_packed3),
                                max_width=max_w,
                                max_length=max_h,
                                max_iterations=300,
                            )
                            layout_packed3 = clamp_bounds(layout_packed3, max_w, max_h)
                            dump_layout(layout_packed3, f"attempt{attempt + 1}_packed_after_sep_separated_shrunk2")
                            issues_repaired2 = validate_layout(
                                layout_packed3,
                                max_width=max_w,
                                max_length=max_h,
                                min_separation=args.min_separation,
                                adjacency=adjacency,
                            )
                            record_issues("post_separation_pack_full_validation_shrunk2", issues_repaired2, attempt=attempt + 1)
                            if not issues_repaired2:
                                layout_json = layout_packed3
                            elif issues_are_separation_only(issues_repaired2):
                                log.warning(
                                    "Proceeding despite separation-only issues after sep-pack+separate: %s",
                                    "; ".join(issues_repaired2),
                                )
                                layout_json = layout_packed3
                            else:
                                dump_layout(layout_repaired, f"attempt{attempt + 1}_failed")
                                raise BoundaryViolationError(
                                    "Layout validation failed after clamping: "
                                    + "; ".join(issues_repaired2)
                                )
                        else:
                            dump_layout(layout_repaired, f"attempt{attempt + 1}_failed")
                            raise BoundaryViolationError(
                                "Layout validation failed after clamping: "
                                + "; ".join(issues_repaired)
                            )
                    else:
                        layout_json = layout_repaired

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
        # If only separation problems remain, attempt one last shrink+pack+separate cycle
        if issues_are_separation_only(final_issues):
            layout_shrunk_final = shrink_to_fit(layout_json, max_w, max_h, target_fill=0.6)
            layout_packed_final = pack_layout(layout_shrunk_final, max_width=max_w, max_length=max_h, grid=1.0)
            layout_packed_final = enforce_min_separation(
                layout_packed_final,
                args.min_separation,
                adjacency=build_sep_exempt_adjacency(layout_packed_final),
                max_width=max_w,
                max_length=max_h,
                max_iterations=400,
            )
            layout_packed_final = clamp_bounds(layout_packed_final, max_w, max_h)
            dump_layout(layout_packed_final, "final_shrunk_packed_separated")
            final_issues2 = validate_layout(
                layout_packed_final,
                max_width=max_w,
                max_length=max_h,
                min_separation=args.min_separation,
                adjacency=adjacency,
            )
            if not final_issues2:
                layout_json = layout_packed_final
            else:
                # If only separation issues remain even after our best effort, accept layout without min-separation.
                if issues_are_separation_only(final_issues2):
                    log.warning(
                        "Proceeding despite separation-only issues after best-effort repair: %s",
                        "; ".join(final_issues2),
                    )
                else:
                    raise BoundaryViolationError(
                        "Layout validation failed before writing output: "
                        + "; ".join(final_issues2)
                    )
        else:
            # As a last resort, if only separation issues remain, accept layout.
            if issues_are_separation_only(final_issues):
                log.warning(
                    "Proceeding despite separation-only issues: %s",
                    "; ".join(final_issues),
                )
            else:
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
