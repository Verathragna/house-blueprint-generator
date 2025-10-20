import os, json, logging, re
from typing import Any, Dict, Optional, Tuple

from Generate.params import Params
from Generate.constants import (
    VERSION,
    MIN_SEPARATION_DEFAULT,
    DECODE_MAX_ATTEMPTS,
    GUIDED_TOPK_DEFAULT,
    GUIDED_BEAM_DEFAULT,
    TEMPERATURE_DEFAULT,
    BEAM_SIZE_DEFAULT,
    STRATEGY_DEFAULT,
    BACKEND_DEFAULT,
    GRID_DEFAULT,
    ZONING_DEFAULT,
    MIN_HALL_WIDTH_DEFAULT,
    CPSAT_TIME_LIMIT_S,
    CKPT_FILENAME,
)

log = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CKPT_DEFAULT = os.path.join(REPO_ROOT, "checkpoints", CKPT_FILENAME)


def emit_params_schema(path: str) -> None:
    """Write the versioned JSON Schema for Params to the given path."""
    schema = Params.model_json_schema()
    # add versioned id
    schema["$id"] = f"urn:house-blueprint-generator:params:{VERSION}"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def _resolve_checkpoint_path(requested_path: Optional[str]) -> str:
    if requested_path:
        if os.path.exists(requested_path):
            return requested_path
        raise FileNotFoundError(f"Requested checkpoint {requested_path} not found.")
    if os.path.exists(CKPT_DEFAULT):
        return CKPT_DEFAULT
    # Fallback to latest epoch_*.pt
    epoch_dir = os.path.join(REPO_ROOT, "checkpoints")
    latest_epoch: Optional[Tuple[int, str]] = None
    for name in os.listdir(epoch_dir) if os.path.isdir(epoch_dir) else []:
        m = re.match(r"epoch_(\d+)\.pt$", name)
        if not m:
            continue
        idx = int(m.group(1))
        path = os.path.join(epoch_dir, name)
        if latest_epoch is None or idx > latest_epoch[0]:
            latest_epoch = (idx, path)
    if latest_epoch is None:
        raise FileNotFoundError("Checkpoint not found. Train first (checkpoints/model_latest.pth).")
    log.info("Using most recent epoch checkpoint %s", os.path.basename(latest_epoch[1]))
    return latest_epoch[1]


def _normalize_hints(hints_dict: Optional[Dict[str, list]]) -> Dict[str, list]:
    norm: Dict[str, list] = {}
    for k, vals in (hints_dict or {}).items():
        key = (k or "").strip()
        if not key:
            continue
        key_t = key[:1].upper() + key[1:].lower()
        norm[key_t] = []
        for v in vals or []:
            val = (v or "").strip()
            if not val:
                continue
            val_t = val[:1].upper() + val[1:].lower()
            if val_t not in norm[key_t]:
                norm[key_t].append(val_t)
    return norm


def generate_layout(
    *,
    params: Params,
    raw_params: Dict[str, Any],
    backend: str = BACKEND_DEFAULT,
    strategy: str = STRATEGY_DEFAULT,
    temperature: float = TEMPERATURE_DEFAULT,
    beam_size: int = BEAM_SIZE_DEFAULT,
    min_separation: float = MIN_SEPARATION_DEFAULT,
    guided_topk: int = GUIDED_TOPK_DEFAULT,
    guided_beam: int = GUIDED_BEAM_DEFAULT,
    refine_iters: int = 0,
    refine_temp: float = 5.0,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    tokenizer=None,
    model=None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Deterministic pipeline to produce a layout and metadata.

    Returns (layout_json, metadata_dict).
    """
    from tokenizer.tokenizer import BlueprintTokenizer
    from evaluation.validators import (
        validate_layout,
        clamp_bounds,
        enforce_min_separation,
        pack_layout,
    )
    from evaluation.evaluate_sample import assert_room_counts
    from evaluation.feasibility_checker import check_layout_feasibility

    # 1) Validate feasibility upfront
    feasible, message, analysis = check_layout_feasibility(raw_params)
    if not feasible:
        raise ValueError(f"Infeasible parameters: {message}")

    dims = raw_params.get("dimensions") or {}
    max_w = float(dims.get("width", 40))
    max_h = float(dims.get("depth", dims.get("height", 40)))

    adjacency = params.adjacency.root if params.adjacency else None

    def build_adjacency_hints() -> Dict[str, list]:
        default_hints = {
            "Bedroom": ["Bathroom", "Hallway"],
            "Bathroom": ["Bedroom"],
            "Kitchen": ["Dining Room", "Living Room", "Laundry Room"],
            "Dining Room": ["Kitchen", "Living Room"],
            "Living Room": ["Dining Room", "Kitchen"],
            "Garage": ["Laundry Room"],
            "Laundry Room": ["Kitchen", "Garage"],
        }
        merged = dict(_normalize_hints(default_hints))
        if adjacency:
            for k, vals in _normalize_hints(adjacency).items():
                merged.setdefault(k, [])
                for v in vals:
                    if v not in merged[k]:
                        merged[k].append(v)
        return merged

    metadata: Dict[str, Any] = {
        "version": VERSION,
        "backend_requested": backend,
        "strategy": strategy,
        "temperature": temperature,
        "beam_size": beam_size,
        "min_separation": min_separation,
        "guided_topk": guided_topk,
        "guided_beam": guided_beam,
        "seed": seed,
    }

    layout_json: Optional[Dict[str, Any]] = None
    issues: list[str] = []
    attempts = 0

    # 2) Try CP-SAT backend first if allowed
    if backend in ("cp", "auto"):
        try:
            from solver.cpsat_solver import solve_layout_cpsat, build_intent_from_params  # type: ignore
            intent = build_intent_from_params(raw_params)
            cp_layout = solve_layout_cpsat(
                intent,
                max_width=int(round(max_w)),
                max_length=int(round(max_h)),
                min_separation=int(round(min_separation)),
                time_limit_s=CPSAT_TIME_LIMIT_S,
            )
            if cp_layout:
                adjacency_hints = build_adjacency_hints()
                cp_layout = pack_layout(
                    cp_layout,
                    max_width=max_w,
                    max_length=max_h,
                    grid=GRID_DEFAULT,
                    adjacency_hints=adjacency_hints,
                    zoning=ZONING_DEFAULT,
                    min_hall_width=MIN_HALL_WIDTH_DEFAULT,
                )
                cp_layout = clamp_bounds(cp_layout, max_w, max_h)
                issues_cp = validate_layout(
                    cp_layout,
                    max_width=max_w,
                    max_length=max_h,
                    min_separation=min_separation,
                    adjacency=adjacency,
                )
                missing_cp = assert_room_counts(cp_layout, raw_params)
                if not issues_cp and not missing_cp:
                    layout_json = cp_layout
                    metadata["backend_used"] = "cp"
                else:
                    log.info("CP-SAT produced issues (%s) or missing rooms (%s); falling back to model",
                             len(issues_cp), len(missing_cp))
        except ImportError:
            log.info("OR-Tools not installed; skipping CP-SAT backend")
        except Exception as exc:
            log.warning("CP-SAT backend failed: %s", exc)

    # 3) Model decoding fallback
    if layout_json is None and backend in ("model", "auto"):
        attempts = 0
        # Prepare tokenizer / model
        tk = tokenizer or BlueprintTokenizer()
        if model is None:
            import torch
            from models.layout_transformer import LayoutTransformer
            from models.decoding import decode
            ckpt_path = _resolve_checkpoint_path(checkpoint)
            ckpt_blob = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt_blob["model"] if isinstance(ckpt_blob, dict) and "model" in ckpt_blob else ckpt_blob

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
                try:
                    idxs = set(int(k.split(".")[2]) for k in layers)
                    return max(idxs) + 1
                except Exception:
                    return 4

            d_model = _infer_d_model(state_dict)
            dim_ff = _infer_dim_ff(state_dict)
            num_layers = _infer_num_layers(state_dict)
            nhead = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)

            net = LayoutTransformer(tk.get_vocab_size(), d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff)
            net.load_state_dict(state_dict, strict=False)
            net.to(device)
            model = net
        else:
            from models.decoding import decode  # type: ignore
            tk = tokenizer

        prefix = tk.encode_params(params.model_dump())
        adjacency_requirements = tk.adjacency_requirements_from_params(adjacency)

        def partial_validator(layout_dict):
            rooms = (layout_dict.get("layout") or {}).get("rooms", [])
            if not rooms:
                return []
            return validate_layout(
                layout_dict,
                max_width=max_w,
                max_length=max_h,
                min_separation=min_separation,
                adjacency=adjacency,
                require_connectivity=False,
            )

        decode_kwargs = {
            "max_len": 160,
            "strategy": strategy,
            "temperature": temperature,
            "beam_size": beam_size,
            "required_counts": {},
            "bias_tokens": {},
            "tokenizer": tk,
            "max_width": max_w,
            "max_length": max_h,
            "adjacency_requirements": adjacency_requirements,
        }
        if strategy == "guided":
            decode_kwargs.update(
                {
                    "constraint_validator": partial_validator,
                    "validator_min_rooms": 1,
                    "guided_top_k": guided_topk,
                    "guided_beam_size": guided_beam,
                }
            )

        max_attempts = max(1, int(DECODE_MAX_ATTEMPTS))
        adjacency_hints = build_adjacency_hints()
        last_issues: list[str] = []
        for attempt in range(max_attempts):
            attempts = attempt + 1
            layout_tokens = decode(model, prefix, **decode_kwargs)
            cand = tk.decode_layout_tokens(layout_tokens)
            # pack -> separate -> clamp -> validate
            cand = pack_layout(
                cand,
                max_width=max_w,
                max_length=max_h,
                grid=GRID_DEFAULT,
                adjacency_hints=adjacency_hints,
                zoning=ZONING_DEFAULT,
                min_hall_width=MIN_HALL_WIDTH_DEFAULT,
            )
            cand = clamp_bounds(cand, max_w, max_h)
            if min_separation > 0:
                cand = enforce_min_separation(
                    cand,
                    min_separation,
                    adjacency=adjacency,
                    max_width=max_w,
                    max_length=max_h,
                    max_iterations=100,
                )
                cand = clamp_bounds(cand, max_w, max_h)
            last_issues = validate_layout(
                cand,
                max_width=max_w,
                max_length=max_h,
                min_separation=min_separation,
                adjacency=adjacency,
            )
            missing = assert_room_counts(cand, raw_params)
            if not last_issues and not missing:
                layout_json = cand
                break
        if layout_json is None:
            # Return the last candidate along with issues in metadata
            layout_json = cand
            issues = last_issues
            metadata["note"] = "returned best-effort layout with issues"
        metadata["backend_used"] = "model"

    if layout_json is None:
        raise RuntimeError("Failed to generate a layout with requested pipeline")

    metadata["attempts"] = attempts
    metadata["issues"] = issues
    metadata["issues_count"] = len(issues)
    metadata["dims"] = {"width": max_w, "depth": max_h}

    return layout_json, metadata
