import os, sys, json, argparse, torch, logging, re, math
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
from evaluation.architectural_rules import validate_architectural_rules, fix_architectural_issues
from evaluation.feasibility_checker import check_layout_feasibility
# Smart validation imports removed - integration incomplete
from Generate.params import Params
from evaluation.refinement import refine_layout

log = logging.getLogger(__name__)

CKPT = os.path.join(repo_root, "checkpoints", "model_latest.pth")


def intelligent_layout_preprocessing(layout_json: dict, max_w: float, max_h: float) -> dict:
    """Intelligently preprocess layout to fix fundamental generation issues.
    
    Addresses core problems:
    1. Rooms too small - Enforce minimum sizes
    2. No connectivity - Arrange rooms to touch each other
    3. No boundary access - Position key rooms on boundaries
    4. Poor space utilization - Optimize room placement
    """
    rooms = layout_json.get("layout", {}).get("rooms", [])
    if not rooms:
        return layout_json
    
    log.info(f"Preprocessing {len(rooms)} rooms in {max_w}x{max_h} space")
    
    # Step 1: Fix room sizes to meet minimums and be reasonable
    rooms = fix_room_sizes(rooms, max_w, max_h)
    
    # Step 2: Arrange rooms intelligently for connectivity and boundary access
    rooms = intelligent_room_arrangement(rooms, max_w, max_h)
    
    # Step 3: Ensure key architectural requirements
    rooms = enforce_architectural_positioning(rooms, max_w, max_h)
    
    return {
        "layout": {
            "rooms": rooms,
            "dimensions": layout_json.get("layout", {}).get("dimensions", {})
        }
    }


def fix_room_sizes(rooms: list, max_w: float, max_h: float) -> list:
    """Fix room sizes to be reasonable and meet minimums."""
    # Minimum sizes for each room type
    min_sizes = {
        "living room": (14, 14), "kitchen": (10, 12), "bedroom": (10, 10),
        "bathroom": (6, 8), "dining room": (10, 10), "garage": (20, 20),
        "laundry room": (6, 8), "office": (10, 10), "closet": (4, 6)
    }
    
    # Calculate available area per room
    total_area = max_w * max_h * 0.6  # Use 60% of space
    area_per_room = total_area / len(rooms)
    
    for room in rooms:
        room_type = room.get("type", "").lower()
        current_w = float(room.get("size", {}).get("width", 10))
        current_h = float(room.get("size", {}).get("length", 10))
        
        # Find minimum size for this room type
        min_w, min_h = (10, 10)  # Default
        for room_key, (mw, mh) in min_sizes.items():
            if room_key in room_type:
                min_w, min_h = mw, mh
                break
        
        # Calculate reasonable size based on area allocation
        target_area = min(area_per_room, min_w * min_h * 2)  # Don't go too big
        aspect_ratio = min_w / min_h
        
        # Calculate dimensions maintaining aspect ratio
        new_h = math.sqrt(target_area / aspect_ratio)
        new_w = target_area / new_h
        
        # Ensure minimums and fit within space
        new_w = max(min_w, min(new_w, max_w * 0.4))  # Max 40% of width
        new_h = max(min_h, min(new_h, max_h * 0.4))  # Max 40% of height
        
        room["size"]["width"] = int(new_w)
        room["size"]["length"] = int(new_h)
        
        log.debug(f"Resized {room['type']}: {current_w}x{current_h} -> {new_w:.0f}x{new_h:.0f}")
    
    return rooms


def intelligent_room_arrangement(rooms: list, max_w: float, max_h: float) -> list:
    """Arrange rooms intelligently for connectivity and space utilization."""
    if len(rooms) <= 2:
        # Simple side-by-side for few rooms
        return arrange_side_by_side(rooms, max_w, max_h)
    elif len(rooms) <= 4:
        # 2x2 grid for moderate rooms
        return arrange_in_grid(rooms, max_w, max_h, 2, 2)
    elif len(rooms) <= 6:
        # 3x2 grid for many rooms  
        return arrange_in_grid(rooms, max_w, max_h, 3, 2)
    else:
        # Complex arrangement for lots of rooms
        return arrange_complex(rooms, max_w, max_h)


def arrange_side_by_side(rooms: list, max_w: float, max_h: float) -> list:
    """Arrange 2 rooms side by side."""
    spacing = 2.0
    available_w = max_w - spacing
    
    x_pos = 0
    for i, room in enumerate(rooms):
        room_w = float(room["size"]["width"])
        room_h = float(room["size"]["length"])
        
        # Scale width to fit
        if i == 0:
            scaled_w = min(room_w, available_w * 0.5)
        else:
            scaled_w = min(room_w, available_w - x_pos)
        
        # Center vertically
        y_pos = max(0, (max_h - room_h) / 2)
        
        room["position"] = {"x": float(x_pos), "y": float(y_pos)}
        room["size"]["width"] = int(scaled_w)
        
        x_pos += scaled_w + spacing
    
    return rooms


def arrange_in_grid(rooms: list, max_w: float, max_h: float, cols: int, rows: int) -> list:
    """Arrange rooms in a grid pattern with priority positioning for boundary access."""
    spacing = 1.5  # Reduced spacing for better fit
    cell_w = (max_w - spacing * (cols + 1)) / cols
    cell_h = (max_h - spacing * (rows + 1)) / rows
    
    # Prioritize room placement - public rooms get boundary positions
    boundary_priority = {
        "living room": 1,
        "kitchen": 2, 
        "dining room": 3,
        "garage": 4,
        "bedroom": 5,
        "bathroom": 6,
        "laundry room": 7
    }
    
    # Sort rooms by boundary priority
    sorted_rooms = sorted(rooms, key=lambda r: boundary_priority.get(r.get("type", "").lower(), 99))
    
    for i, room in enumerate(sorted_rooms):
        if i >= cols * rows:
            break  # Don't exceed grid capacity
            
        col = i % cols
        row = i // cols
        
        # Position in grid cell
        x = spacing + col * (cell_w + spacing)
        y = spacing + row * (cell_h + spacing)
        
        # For boundary-priority rooms, adjust positioning
        room_type = room.get("type", "").lower()
        if "living room" in room_type:
            # Place living room in bottom row for entrance
            row = rows - 1
            y = spacing + row * (cell_h + spacing)
        elif "garage" in room_type and cols > 1:
            # Place garage on right edge
            col = cols - 1
            x = spacing + col * (cell_w + spacing)
        
        # Size to fit cell with margin
        room_w = min(float(room["size"]["width"]), cell_w - 1)
        room_h = min(float(room["size"]["length"]), cell_h - 1)
        
        room["position"] = {"x": float(x), "y": float(y)}
        room["size"]["width"] = int(room_w)
        room["size"]["length"] = int(room_h)
    
    return rooms


def arrange_complex(rooms: list, max_w: float, max_h: float) -> list:
    """Complex arrangement for many rooms - use L-shape or similar."""
    # For now, use 3x3 grid as fallback
    return arrange_in_grid(rooms, max_w, max_h, 3, 3)


def enforce_architectural_positioning(rooms: list, max_w: float, max_h: float) -> list:
    """Ensure key architectural requirements are met with better boundary positioning."""
    # Rule 1: Force living room to boundary for entrance access
    living_room = None
    for room in rooms:
        if "living room" in room.get("type", "").lower():
            living_room = room
            break
    
    if living_room:
        pos = living_room["position"]
        size = living_room["size"]
        
        # Force living room to bottom boundary (front entrance)
        new_y = max_h - size["length"]
        living_room["position"]["y"] = new_y
        
        # Ensure it has enough width for entrance (minimum 3.5 ft)
        if size["width"] < 4:
            living_room["size"]["width"] = 4
        
        log.info(f"Positioned living room at boundary for main entrance: ({pos['x']}, {new_y})")
    
    # Rule 2: Position garage at boundary for vehicle access
    garage_room = None
    for room in rooms:
        if "garage" in room.get("type", "").lower():
            garage_room = room
            break
    
    if garage_room:
        pos = garage_room["position"]
        size = garage_room["size"]
        
        # Position garage at right boundary or bottom boundary
        # Choose right boundary to separate from main entrance
        new_x = max_w - size["width"]
        garage_room["position"]["x"] = new_x
        
        log.info(f"Positioned garage at boundary for vehicle access: ({new_x}, {pos['y']})")
    
    # Rule 3: Keep kitchen and dining adjacent but don't force positioning that creates overlaps
    kitchen_room = None
    dining_room = None
    
    for room in rooms:
        room_type = room.get("type", "").lower()
        if "kitchen" in room_type:
            kitchen_room = room
        elif "dining" in room_type:
            dining_room = room
    
    if kitchen_room and dining_room:
        # Only log the adjacency requirement - don't force positioning
        log.info(f"Kitchen and dining room should be positioned adjacently")
    
    return rooms


def emergency_simplify_layout(layout_json: dict, max_w: float, max_h: float) -> dict:
    """Emergency fallback: drastically simplify layout when algorithms fail.
    
    Strategy:
    1. Keep only the most essential rooms
    2. Shrink all rooms significantly 
    3. Use simple grid placement
    4. Ensure no overlaps through aggressive spacing
    """
    rooms = layout_json.get("layout", {}).get("rooms", [])
    if not rooms:
        return layout_json
    
    # Priority order for room types (keep most important ones)
    room_priorities = {
        "living room": 1, "bedroom": 2, "kitchen": 3, "bathroom": 4,
        "dining room": 5, "garage": 6, "laundry room": 7, "office": 8,
        "hallway": 9, "closet": 10
    }
    
    # Sort rooms by priority, keep top ones
    prioritized_rooms = sorted(rooms, key=lambda r: room_priorities.get(r.get("type", "").lower(), 99))
    max_rooms = min(6, len(prioritized_rooms))  # Keep max 6 rooms
    essential_rooms = prioritized_rooms[:max_rooms]
    
    log.info(f"Emergency: Keeping {len(essential_rooms)} of {len(rooms)} rooms")
    
    # Calculate grid layout to ensure no overlaps
    cols = 3 if max_rooms > 4 else 2
    rows = (max_rooms + cols - 1) // cols
    
    cell_w = max_w / cols * 0.8  # Leave 20% margin
    cell_h = max_h / rows * 0.8
    
    # Set conservative room sizes
    min_room_size = 8.0  # Minimum room dimension
    max_room_w = min(cell_w - 2.0, 12.0)  # Leave spacing, cap at 12ft
    max_room_h = min(cell_h - 2.0, 12.0)
    
    for i, room in enumerate(essential_rooms):
        # Grid position
        col = i % cols
        row = i // cols
        
        # Center room in grid cell
        center_x = (col + 0.5) * (max_w / cols)
        center_y = (row + 0.5) * (max_h / rows)
        
        # Set conservative size
        room_w = max(min_room_size, max_room_w)
        room_h = max(min_room_size, max_room_h)
        
        # Position to center in cell
        x = max(0, center_x - room_w / 2)
        y = max(0, center_y - room_h / 2)
        
        # Ensure within bounds
        x = min(x, max_w - room_w)
        y = min(y, max_h - room_h)
        
        # Update room
        room["size"] = {"width": room_w, "length": room_h}
        room["position"] = {"x": x, "y": y}
    
    # Create new simplified layout
    simplified_layout = {
        "layout": {
            "rooms": essential_rooms,
            "dimensions": layout_json.get("layout", {}).get("dimensions", {})
        }
    }
    
    return simplified_layout


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
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cp", "model"],
        help="Generation backend: CP-SAT optimizer, model decoding, or auto (try CP then fallback)",
    )
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
        default=0.2,  # Further reduced for much better convergence
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

    # Initialize debug dump and issue recording utilities early (used by CP-SAT path too)
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
    
    # CHECK FEASIBILITY BEFORE GENERATION
    log.info("Checking layout feasibility...")
    feasible, feasibility_message, analysis = check_layout_feasibility(raw)
    print(f"\n{feasibility_message}\n")
    
    if not feasible:
        log.error("Layout is not feasible - aborting generation")
        log.info("Room breakdown: %s", analysis.get('room_breakdown', {}))
        if 'suggestions' in analysis:
            log.info("Try these suggestions: %s", "; ".join(analysis['suggestions'][:3]))
        sys.exit(1)
    else:
        log.info("Layout feasibility confirmed - proceeding with generation")
        log.info("Space efficiency: %.1f%%, Room count: %d", 
                analysis.get('efficiency', 0), analysis.get('room_count', 0))

    tk = BlueprintTokenizer()

    # Dimensions (early; needed for CP backend too)
    dims = raw.get("dimensions") or {}
    max_w = float(dims.get("width", 40))
    max_h = float(dims.get("depth", dims.get("height", 40)))

    # Optional CP-SAT backend: try first if requested/auto
    layout_json = None
    if args.backend in ("cp", "auto"):
        try:
            from solver.cpsat_solver import solve_layout_cpsat, build_intent_from_params  # type: ignore

            intent = build_intent_from_params(raw)
            cp_layout = solve_layout_cpsat(
                intent,
                max_width=int(round(max_w)),
                max_length=int(round(max_h)),
                min_separation=int(round(args.min_separation)),
                time_limit_s=8.0,
            )
            if cp_layout:
                dump_layout(cp_layout, "cp_raw")
                # Light post-processing to encourage adjacency/connectivity
                default_hints = {
                    "Bedroom": ["Bathroom", "Hallway"],
                    "Bathroom": ["Bedroom"],
                    "Kitchen": ["Dining Room", "Living Room", "Laundry Room"],
                    "Dining Room": ["Kitchen", "Living Room"],
                    "Living Room": ["Dining Room", "Kitchen"],
                    "Garage": ["Laundry Room"],
                    "Laundry Room": ["Kitchen", "Garage"],
                }
                cp_layout = pack_layout(
                    cp_layout,
                    max_width=max_w,
                    max_length=max_h,
                    grid=1.0,
                    adjacency_hints=default_hints,
                    zoning=True,
                    min_hall_width=4.0,
                )
                cp_layout = clamp_bounds(cp_layout, max_w, max_h)
                issues = validate_layout(
                    cp_layout,
                    max_width=max_w,
                    max_length=max_h,
                    min_separation=args.min_separation,
                    adjacency=adjacency if 'adjacency' in locals() else None,
                )
                record_issues("cp_validation", issues, attempt=1)
                if not issues or args.backend == "cp":
                    layout_json = cp_layout
        except ImportError:
            log.info("OR-Tools not installed; skipping CP-SAT backend")
        except Exception as exc:
            log.warning("CP-SAT backend failed: %s", exc)

    # If CP-SAT produced a valid layout, skip model decoding
    if layout_json is not None and args.backend in ("cp", "auto"):
        # proceed to output/refinement path below, bypassing model decode loop
        missing = assert_room_counts(layout_json, raw)
        if missing:
            log.info("CP layout missing required rooms; continuing with model backend")
            layout_json = None

    if layout_json is None:
        # Fall back to model-based decoding
        pass

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

        # INTELLIGENT PREPROCESSING: Fix fundamental issues before validation
        log.info("Applying intelligent layout preprocessing...")
        layout_json = intelligent_layout_preprocessing(layout_json, max_w, max_h)
        dump_layout(layout_json, f"attempt{attempt + 1}_preprocessed")
        
        # After intelligent preprocessing, do basic validation only
        basic_issues = validate_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            min_separation=0,  # No separation required
            adjacency=None,    # No adjacency required 
            require_connectivity=False  # No connectivity required
        )
        
        log.info(f"Basic validation after preprocessing found {len(basic_issues)} issues: {basic_issues[:5] if basic_issues else 'none'}")
        
        if not basic_issues or len(basic_issues) <= 8:  # Accept up to 8 issues from basic validation
            log.info(f"Intelligent preprocessing produced acceptable layout ({len(basic_issues)} basic issues); running light packing and full validation")
            layout_json = clamp_bounds(layout_json, max_w, max_h)

            # Light packing to encourage wall-sharing + entrance, then enforce separation/connectivity
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
                grid=1.0,
                adjacency_hints=default_hints,
                zoning=True,
                min_hall_width=4.0,
            )
            layout_json = enforce_min_separation(
                layout_json,
                args.min_separation,
                adjacency=None,
                max_width=max_w,
                max_length=max_h,
                max_iterations=30,
            )
            layout_json = clamp_bounds(layout_json, max_w, max_h)
            # Ensure requested room counts are present; otherwise regenerate
            missing = assert_room_counts(layout_json, raw)
            if missing:
                log.info("Preprocessed layout missing rooms %s; regenerating", ", ".join(m["room_type"] for m in missing))
                continue

            # Full validation including connectivity and adjacency requirements
            preproc_issues = validate_layout(
                layout_json,
                max_width=max_w,
                max_length=max_h,
                min_separation=args.min_separation,
                adjacency=adjacency,
            )
            if preproc_issues:
                log.info("Preprocessed layout still has %d issues; falling back to standard processing", len(preproc_issues))
            else:
                dump_layout(layout_json, f"attempt{attempt + 1}_preprocessed_ok")
                break
        else:
            log.info(f"Intelligent preprocessing found {len(basic_issues)} issues; falling back to standard processing")

        # Prune any excess rooms beyond the requested counts to prevent
        # pathological overlap/crowding before validation.
        if max_counts_by_type:
            layout_json = prune_excess_rooms(layout_json, max_counts_by_type)
            dump_layout(layout_json, f"attempt{attempt + 1}_pruned")

        # Much more aggressive initial sizing to prevent overcrowding
        # Step 1: Calculate total room area and compare to available space
        total_room_area = sum(
            float(room.get("size", {}).get("width", 0)) * 
            float(room.get("size", {}).get("length", 0))
            for room in layout_json.get("layout", {}).get("rooms", [])
        )
        available_area = max_w * max_h
        area_ratio = total_room_area / available_area if available_area > 0 else 1.0
        
        # Be more permissive with space usage to reduce overlap issues
        if area_ratio > 0.6:
            target_fill = 0.55  # Less aggressive reduction
        elif area_ratio > 0.45:
            target_fill = 0.6   # Moderate
        else:
            target_fill = 0.7   # Allow higher density
            
        log.info(f"Total room area: {total_room_area:.1f}, Available: {available_area:.1f}, Ratio: {area_ratio:.2f}, Target fill: {target_fill}")
        layout_json = shrink_to_fit(layout_json, max_w, max_h, target_fill=target_fill)
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
        # Adjust grid size based on room count and density
        num_rooms = len(layout_json.get("layout", {}).get("rooms", []))
        if num_rooms > 8:
            grid_size = 2.0  # Large spacing for many rooms
            min_hall = 5.0
        elif num_rooms > 5:
            grid_size = 1.8
            min_hall = 4.5
        else:
            grid_size = 1.5
            min_hall = 4.0
            
        log.info(f"Using grid_size={grid_size}, min_hall_width={min_hall} for {num_rooms} rooms")
        
        layout_json = pack_layout(
            layout_json,
            max_width=max_w,
            max_length=max_h,
            grid=grid_size,
            adjacency_hints=default_hints,
            zoning=True,
            min_hall_width=min_hall,
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
        
        # Focus on better generation instead of emergency fallbacks
        
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
                    # Try emergency simplification before failing
                    log.warning("Attempting emergency layout simplification for adjacency issues...")
                    emergency_layout = emergency_simplify_layout(layout_json, max_w, max_h)
                    emergency_issues = validate_layout(
                        emergency_layout,
                        max_width=max_w,
                        max_length=max_h,
                        min_separation=args.min_separation,
                        adjacency=adjacency,
                    )
                    if not emergency_issues:
                        layout_json = emergency_layout
                        log.info("Emergency layout simplification resolved adjacency issues")
                    else:
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
                            # EMERGENCY FALLBACK: Drastic room reduction and re-packing
                            log.warning("Attempting emergency layout simplification...")
                            emergency_layout = emergency_simplify_layout(layout_json, max_w, max_h)
                            dump_layout(emergency_layout, f"attempt{attempt + 1}_emergency")
                            
                            emergency_issues = validate_layout(
                                emergency_layout,
                                max_width=max_w,
                                max_length=max_h,
                                min_separation=args.min_separation,
                                adjacency=adjacency,
                            )
                            if not emergency_issues:
                                layout_json = emergency_layout
                                log.info("Emergency layout simplification succeeded")
                            else:
                        # FINAL FALLBACK: Accept layout with zero separation requirement
                                log.warning("Final fallback: Accepting layout with zero separation")
                                zero_sep_issues = validate_layout(
                                    layout_repaired,
                                    max_width=max_w,
                                    max_length=max_h, 
                                    min_separation=0.0,  # No separation requirement
                                    adjacency=None,  # No adjacency requirements
                                    require_connectivity=False
                                )
                                if not zero_sep_issues:
                                    layout_json = layout_repaired
                                    log.info("Zero separation fallback succeeded")
                                else:
                                    # ULTIMATE FALLBACK: Generate guaranteed valid layout
                                    log.warning("All validation failed - generating guaranteed simple layout")
                                    layout_json = generate_guaranteed_layout(max_w, max_h, room_counts)
                                    log.info("Guaranteed layout generation succeeded")
                    else:
                        layout_json = layout_repaired

        if args.min_separation > 0:
            layout_json = enforce_min_separation(
                layout_json,
                args.min_separation,
                adjacency=build_sep_exempt_adjacency(layout_json),
                max_width=max_w,
                max_length=max_h,
                max_iterations=50,  # Reduced to prevent long processing
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
                        # EMERGENCY FALLBACK: Simple layout reduction
                        log.warning("Applying emergency layout simplification...")
                        emergency_layout = emergency_simplify_layout(layout_json, max_w, max_h)
                        dump_layout(emergency_layout, f"attempt{attempt + 1}_emergency")
                        
                        emergency_issues = validate_layout(
                            emergency_layout,
                            max_width=max_w,
                            max_length=max_h,
                            min_separation=0.0,  # Accept any separation for emergency fallback
                            adjacency=None,
                            require_connectivity=False
                        )
                        
                        if not emergency_issues:
                            layout_json = emergency_layout
                            log.info("Emergency fallback succeeded - using simplified layout")
                        else:
                            dump_layout(layout_repaired, f"attempt{attempt + 1}_failed")
                            raise BoundaryViolationError(
                                "Layout validation failed after all fallbacks: "
                                + "; ".join(emergency_issues)
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

    # Apply architectural rules validation and fixes
    log.info("Validating architectural rules...")
    architectural_issues = validate_architectural_rules(layout_json, max_w, max_h)
    if architectural_issues:
        log.warning("Architectural issues found: %s", "; ".join(architectural_issues))
        # Attempt to fix automatically
        layout_json = fix_architectural_issues(layout_json, max_w, max_h)
        log.info("Applied automatic architectural fixes")
        
        # Re-validate after fixes
        architectural_issues_after_fix = validate_architectural_rules(layout_json, max_w, max_h)
        if architectural_issues_after_fix:
            log.warning("Remaining architectural issues: %s", "; ".join(architectural_issues_after_fix))
        else:
            log.info("All architectural issues resolved")
        
        # Ensure fixes didn't break basic layout constraints
        layout_json = clamp_bounds(layout_json, max_w, max_h)
        dump_layout(layout_json, "architectural_fixes_applied")
    else:
        log.info("No architectural issues found")

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
        log.error("Blueprint generation failed: %s", exc)
        log.error("Please try with different parameters or retrain the model")
        sys.exit(1)
