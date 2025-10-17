import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from evaluation.evaluate_sample import assert_room_counts  # noqa: E402
from evaluation.validators import validate_layout  # noqa: E402
from models.decoding import decode  # noqa: E402
from models.layout_transformer import LayoutTransformer  # noqa: E402
from tokenizer.tokenizer import BlueprintTokenizer  # noqa: E402


def load_params_dataset(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Params dataset not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            params = record.get("params") or {}
            rows.append(params)
    if not rows:
        raise RuntimeError(f"No params found in {path}")
    return rows


def issues_to_penalty(
    layout_json: Dict,
    params: Dict,
    max_width: float,
    max_length: float,
    min_separation: float,
    adjacency: Optional[Dict[str, List[str]]],
) -> Tuple[float, List[str]]:
    issues = validate_layout(
        layout_json,
        max_width=max_width,
        max_length=max_length,
        min_separation=min_separation,
        adjacency=adjacency,
    )
    missing = assert_room_counts(layout_json, params)
    for miss in missing:
        issues.append(
            f"Missing {miss['room_type']}: expected {miss['expected']} found {miss['found']}"
        )
    penalty = 0.0
    for issue in issues:
        if "overlap" in issue.lower():
            penalty += 2.0
        elif "bounds" in issue.lower():
            penalty += 1.5
        elif "missing" in issue.lower():
            penalty += 2.0
        elif "separation" in issue.lower():
            penalty += 1.0
        else:
            penalty += 0.5
    return penalty, issues


def room_requirements_from_params(params: Dict, tokenizer: BlueprintTokenizer) -> Tuple[Dict[int, int], Dict[int, float]]:
    room_counts: Dict[int, int] = {}
    bias_tokens: Dict[int, float] = {}

    def add_count(token_name: str, count: int, bias: float | None = None):
        tid = tokenizer.token_to_id.get(token_name)
        if tid is not None:
            room_counts[tid] = room_counts.get(tid, 0) + max(0, int(count))
            if bias is not None:
                bias_tokens[tid] = bias_tokens.get(tid, 0.0) + bias

    add_count("BEDROOM", params.get("bedrooms", 0))
    bath = params.get("bathrooms") or {}
    add_count("BATHROOM", bath.get("full", 0) + bath.get("half", 0))
    add_count("KITCHEN", params.get("kitchen", 1))
    add_count("LIVING", params.get("livingRooms", 1))
    add_count("DINING", params.get("diningRooms", 1))
    add_count("LAUNDRY", params.get("laundryRooms", 1))
    if params.get("bonusRoom"):
        add_count("BONUS", 1, bias=0.5)
    if params.get("garage"):
        add_count("GARAGE", 1, bias=1.0)
    return room_counts, bias_tokens


def sequence_log_prob(model: LayoutTransformer, seq: List[int], prefix_len: int, device: torch.device) -> torch.Tensor:
    if len(seq) <= 1:
        return torch.tensor(0.0, device=device)
    input_ids = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([seq[1:]], dtype=torch.long, device=device)
    logits = model(input_ids)
    log_probs = torch.log_softmax(logits, dim=-1)
    total_log_prob = torch.tensor(0.0, device=device)
    for t in range(log_probs.size(1)):
        if t < prefix_len:
            continue
        token_id = int(target_ids[0, t])
        total_log_prob = total_log_prob + log_probs[0, t, token_id]
    return total_log_prob


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the layout model with feedback-driven RL.")
    parser.add_argument("--dataset", default="dataset/train.jsonl", help="JSONL with params to sample from.")
    parser.add_argument("--checkpoint-in", default="checkpoints/model_latest.pth", help="Initial model checkpoint.")
    parser.add_argument("--checkpoint-out", default="checkpoints/model_rl_latest.pth", help="Path to save fine-tuned checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of sampled generations.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of episodes per optimisation step.")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--strategy", default="sample", choices=["sample", "greedy", "guided"], help="Decoding strategy for exploration.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--min-separation", type=float, default=1.0)
    parser.add_argument("--issues-log", default="logs/rl_feedback.log", help="Where to append feedback events.")
    parser.add_argument("--guided-topk", type=int, default=6)
    parser.add_argument("--guided-beam", type=int, default=6)
    parser.add_argument("--entropy-weight", type=float, default=0.0, help="Entropy regularisation weight.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    params_pool = load_params_dataset(dataset_path)

    tokenizer = BlueprintTokenizer()
    vocab_size = tokenizer.get_vocab_size()
    device = torch.device(args.device)

    model = LayoutTransformer(vocab_size=vocab_size)
    checkpoint = torch.load(args.checkpoint_in, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    running_baseline = 0.0
    baseline_momentum = 0.9

    os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
    issues_log_path = Path(args.issues_log)
    issues_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_feedback(entry: Dict):
        with issues_log_path.open("a", encoding="utf-8") as fh:
            json.dump(entry, fh)
            fh.write("\n")

    episodes = args.episodes
    batch_size = max(1, args.batch_size)
    batches = math.ceil(episodes / batch_size)

    for batch_idx in range(batches):
        optimiser.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        for _ in range(batch_size):
            params = random.choice(params_pool)
            dims = params.get("dimensions") or {}
            max_w = float(dims.get("width", 40))
            max_h = float(dims.get("depth", dims.get("height", 40)))

            adjacency_map = params.get("adjacency") or {}
            room_counts, bias_tokens = room_requirements_from_params(params, tokenizer)
            adjacency_requirements = tokenizer.adjacency_requirements_from_params(adjacency_map)

            prefix = tokenizer.encode_params(params)
            decode_kwargs = {
                "max_len": 160,
                "strategy": args.strategy,
                "temperature": args.temperature,
                "beam_size": args.beam,
                "required_counts": room_counts,
                "bias_tokens": bias_tokens,
                "tokenizer": tokenizer,
                "max_width": max_w,
                "max_length": max_h,
                "adjacency_requirements": adjacency_requirements,
            }
            if args.strategy == "guided":
                def partial_validator(layout_dict):
                    rooms = (layout_dict.get("layout") or {}).get("rooms", [])
                    if not rooms:
                        return []
                    return validate_layout(
                        layout_dict,
                        max_width=max_w,
                        max_length=max_h,
                        min_separation=args.min_separation,
                        adjacency=adjacency_map,
                    )

                decode_kwargs.update(
                    {
                        "constraint_validator": partial_validator,
                        "validator_min_rooms": 1,
                        "guided_top_k": args.guided_topk,
                        "guided_beam_size": args.guided_beam,
                    }
                )

            layout_tokens = decode(model, prefix, **decode_kwargs)
            model.train()
            seq = prefix + layout_tokens
            layout_json = tokenizer.decode_layout_tokens(layout_tokens)

            penalty, issues = issues_to_penalty(
                layout_json,
                params,
                max_w,
                max_h,
                args.min_separation,
                adjacency_map,
            )
            reward = 1.0 - penalty

            log_feedback(
                {
                    "params": params,
                    "issues": issues,
                    "reward": reward,
                    "penalty": penalty,
                }
            )

            log_prob = sequence_log_prob(model, seq, len(prefix), device)
            entropy = torch.tensor(0.0, device=device)
            if args.entropy_weight > 0:
                input_ids = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
                logits = model(input_ids)
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)) / probs.size(1)

            running_baseline = baseline_momentum * running_baseline + (1 - baseline_momentum) * reward
            advantage = reward - running_baseline
            loss = -advantage * log_prob
            if args.entropy_weight > 0:
                loss = loss - args.entropy_weight * entropy

            batch_loss = batch_loss + loss

        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

    torch.save({"model": model.state_dict()}, args.checkpoint_out)
    print(f"Saved fine-tuned checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
