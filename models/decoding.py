"""Decoding utilities for blueprint token sequences.

This module centralises different strategies used during autoregressive
generation.  In addition to simple greedy decoding, it supports
temperature-based sampling and beam search.  The helper :func:`decode`
provides a unified entry point selecting between the strategies.
"""

import torch
from tokenizer.tokenizer import SEP_ID, EOS_ID, PAD_ID, BOS_ID


def _prepare_token_maps(tokenizer):
    """Precompute numeric values for width/length/position tokens."""
    w_tokens = {}
    l_tokens = {}
    x_tokens = {}
    y_tokens = {}
    for tok, tid in tokenizer.token_to_id.items():
        if tok.startswith("W") and tok[1:].isdigit():
            w_tokens[tid] = int(tok[1:])
        elif tok.startswith("L") and tok[1:].isdigit():
            l_tokens[tid] = int(tok[1:])
        elif tok.startswith("X") and tok[1:].isdigit():
            x_tokens[tid] = int(tok[1:])
        elif tok.startswith("Y") and tok[1:].isdigit():
            y_tokens[tid] = int(tok[1:])
    return w_tokens, l_tokens, x_tokens, y_tokens


def _apply_bounds_mask(
    logits,
    pending_w,
    pending_l,
    w_tokens,
    l_tokens,
    x_tokens,
    y_tokens,
    max_w=None,
    max_l=None,
):
    """Mask tokens that would place rooms outside the canvas bounds."""

    if max_w is not None:
        for tid, w in w_tokens.items():
            if w > max_w:
                logits[..., tid] = -float("inf")
        if pending_w is not None:
            limit = max_w - pending_w
            for tid, x in x_tokens.items():
                if x > limit:
                    logits[..., tid] = -float("inf")
    if max_l is not None:
        for tid, l in l_tokens.items():
            if l > max_l:
                logits[..., tid] = -float("inf")
        if pending_l is not None:
            limit = max_l - pending_l
            for tid, y in y_tokens.items():
                if y > limit:
                    logits[..., tid] = -float("inf")
    return logits


def _apply_room_bias(logits, counts, required_counts, bias=5.0, extra_bias=None):
    """Apply biases or constraints for room token counts.

    ``counts`` tracks how many times a room token has been generated so far.
    ``required_counts`` maps token ids to the desired total count. Once the
    desired count for a token is met, its logit is set to ``-inf`` to prevent
    further occurrences. Tokens still required receive a positive ``bias`` to
    encourage their selection. ``extra_bias`` can be used to nudge specific
    tokens regardless of count constraints.
    """

    if required_counts:
        for tid, req in required_counts.items():
            seen = counts.get(tid, 0)
            if seen >= req:
                logits[..., tid] = -float("inf")
            else:
                logits[..., tid] += bias
    if extra_bias:
        for tid, b in extra_bias.items():
            logits[..., tid] += b
    return logits


def _trim_sequence(seq):
    """Return portion after SEP_ID if present."""
    if SEP_ID in seq:
        i = seq.index(SEP_ID) + 1
        return seq[i:]
    return seq


class _AdjacencyState:
    """Track adjacency requirements during decoding to bias neighbour selection."""

    def __init__(self, tokenizer, requirements, strength: float = 2.0):
        self.tokenizer = tokenizer
        self.requirements = requirements or {}
        self.strength = strength
        self.pending_counts: dict[int, int] = {}
        self.current_room: int | None = None
        self.in_layout = False

    def clone(self):
        cloned = _AdjacencyState(self.tokenizer, self.requirements, self.strength)
        cloned.pending_counts = dict(self.pending_counts)
        cloned.current_room = self.current_room
        cloned.in_layout = self.in_layout
        return cloned

    def observe(self, token_id: int) -> None:
        if token_id == SEP_ID:
            self.in_layout = True
            self.pending_counts.clear()
            self.current_room = None
            return
        if not self.in_layout:
            return
        if token_id in (PAD_ID, BOS_ID):
            return
        if self.tokenizer.is_room_token_id(token_id):
            if token_id in self.pending_counts:
                self.pending_counts[token_id] -= 1
                if self.pending_counts[token_id] <= 0:
                    self.pending_counts.pop(token_id, None)
            self.current_room = token_id
            return
        tok = self.tokenizer.id_to_token.get(token_id, "")
        if tok.startswith("Y"):
            if self.current_room is not None:
                for neighbour in self.requirements.get(self.current_room, ()):
                    self.pending_counts[neighbour] = self.pending_counts.get(neighbour, 0) + 1
            self.current_room = None

    def current_bias(self) -> dict[int, float]:
        return {
            tid: count * self.strength
            for tid, count in self.pending_counts.items()
            if count > 0
        }


def greedy_decode(
    model,
    prefix_ids,
    max_len: int = 160,
    required_counts=None,
    bias_tokens=None,
    tokenizer=None,
    max_width: float | None = None,
    max_length: float | None = None,
    adjacency_requirements=None,
    adjacency_bias_strength: float = 2.0,
):
    """Generate tokens using greedy argmax decoding."""
    model.eval()
    seq = list(prefix_ids)
    device = next(model.parameters()).device
    counts = {tid: 0 for tid in (required_counts or {})}
    pending_w = pending_l = None
    adjacency_state = None
    if tokenizer:
        w_tokens, l_tokens, x_tokens, y_tokens = _prepare_token_maps(tokenizer)
        if adjacency_requirements:
            adjacency_state = _AdjacencyState(tokenizer, adjacency_requirements, adjacency_bias_strength)
            for tid in seq:
                adjacency_state.observe(tid)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]
            step_bias = dict(bias_tokens or {})
            if adjacency_state:
                for tid, bonus in adjacency_state.current_bias().items():
                    step_bias[tid] = step_bias.get(tid, 0.0) + bonus
            logits = _apply_room_bias(logits, counts, required_counts, extra_bias=step_bias)
            if tokenizer:
                logits = _apply_bounds_mask(
                    logits,
                    pending_w,
                    pending_l,
                    w_tokens,
                    l_tokens,
                    x_tokens,
                    y_tokens,
                    max_width,
                    max_length,
                )
            nxt = int(logits.argmax(dim=-1))
            seq.append(nxt)
            if nxt in counts:
                counts[nxt] += 1
            if tokenizer:
                if adjacency_state:
                    adjacency_state.observe(nxt)
                tok = tokenizer.id_to_token.get(nxt, "")
                if tok.startswith("W") and tok[1:].isdigit():
                    pending_w = int(tok[1:])
                elif tok.startswith("L") and tok[1:].isdigit():
                    pending_l = int(tok[1:])
                elif tok.startswith("Y") and tok[1:].isdigit():
                    pending_w = pending_l = None
                elif tok.startswith("Y"):
                    pending_w = pending_l = None
            if nxt == EOS_ID:
                break
    return _trim_sequence(seq)


def sample_decode(
    model,
    prefix_ids,
    max_len: int = 160,
    temperature: float = 1.0,
    required_counts=None,
    bias_tokens=None,
    tokenizer=None,
    max_width: float | None = None,
    max_length: float | None = None,
    adjacency_requirements=None,
    adjacency_bias_strength: float = 2.0,
):
    """Generate tokens using temperature sampling.

    Args:
        temperature: Sampling temperature. Must be > 0.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    seq = list(prefix_ids)
    device = next(model.parameters()).device
    counts = {tid: 0 for tid in (required_counts or {})}
    pending_w = pending_l = None
    adjacency_state = None
    if tokenizer:
        w_tokens, l_tokens, x_tokens, y_tokens = _prepare_token_maps(tokenizer)
        if adjacency_requirements:
            adjacency_state = _AdjacencyState(tokenizer, adjacency_requirements, adjacency_bias_strength)
            for tid in seq:
                adjacency_state.observe(tid)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :] / temperature
            step_bias = dict(bias_tokens or {})
            if adjacency_state:
                for tid, bonus in adjacency_state.current_bias().items():
                    step_bias[tid] = step_bias.get(tid, 0.0) + bonus
            logits = _apply_room_bias(logits, counts, required_counts, extra_bias=step_bias)
            if tokenizer:
                logits = _apply_bounds_mask(
                    logits,
                    pending_w,
                    pending_l,
                    w_tokens,
                    l_tokens,
                    x_tokens,
                    y_tokens,
                    max_width,
                    max_length,
                )
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1))
            seq.append(nxt)
            if nxt in counts:
                counts[nxt] += 1
            if tokenizer:
                if adjacency_state:
                    adjacency_state.observe(nxt)
                tok = tokenizer.id_to_token.get(nxt, "")
                if tok.startswith("W") and tok[1:].isdigit():
                    pending_w = int(tok[1:])
                elif tok.startswith("L") and tok[1:].isdigit():
                    pending_l = int(tok[1:])
                elif tok.startswith("Y") and tok[1:].isdigit():
                    pending_w = pending_l = None
                elif tok.startswith("Y"):
                    pending_w = pending_l = None
            if nxt == EOS_ID:
                break
    return _trim_sequence(seq)

def beam_search_decode(
    model,
    prefix_ids,
    max_len: int = 160,
    beam_size: int = 5,
    required_counts=None,
    bias_tokens=None,
    tokenizer=None,
    max_width: float | None = None,
    max_length: float | None = None,
    adjacency_requirements=None,
    adjacency_bias_strength: float = 2.0,
):
    """Generate tokens using beam search.

    Args:
        beam_size: Number of beams to keep during search. Must be >= 1.
    """
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    model.eval()
    if tokenizer:
        w_tokens, l_tokens, x_tokens, y_tokens = _prepare_token_maps(tokenizer)
        base_adj_state = None
        if adjacency_requirements:
            base_adj_state = _AdjacencyState(tokenizer, adjacency_requirements, adjacency_bias_strength)
            for tid in prefix_ids:
                base_adj_state.observe(tid)
    else:
        w_tokens = l_tokens = x_tokens = y_tokens = None  # type: ignore
        base_adj_state = None
    sequences = [
        {
            "seq": list(prefix_ids),
            "score": 0.0,
            "counts": {tid: 0 for tid in (required_counts or {})},
            "pending_w": None,
            "pending_l": None,
            "adj_state": base_adj_state.clone() if base_adj_state else None,
        }
    ]
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for cand in sequences:
                seq = cand["seq"]
                score = cand["score"]
                cnts = cand["counts"]
                pw = cand["pending_w"]
                pl = cand["pending_l"]
                adj_state = cand["adj_state"]
                x = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model(x)[:, -1, :]
                step_bias = dict(bias_tokens or {})
                if adj_state:
                    for tid, bonus in adj_state.current_bias().items():
                        step_bias[tid] = step_bias.get(tid, 0.0) + bonus
                logits = _apply_room_bias(logits, cnts, required_counts, extra_bias=step_bias)
                if tokenizer:
                    logits = _apply_bounds_mask(
                        logits, pw, pl, w_tokens, l_tokens, x_tokens, y_tokens, max_width, max_length
                    )
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for log_prob, idx in zip(topk_log_probs[0], topk_ids[0]):
                    tid = int(idx)
                    new_cnts = dict(cnts)
                    if tid in new_cnts:
                        new_cnts[tid] += 1
                    new_pw, new_pl = pw, pl
                    new_adj_state = adj_state.clone() if adj_state else None
                    if tokenizer:
                        tok = tokenizer.id_to_token.get(tid, "")
                        if new_adj_state:
                            new_adj_state.observe(tid)
                        if tok.startswith("W"):
                            new_pw = int(tok[1:])
                        elif tok.startswith("L"):
                            new_pl = int(tok[1:])
                        elif tok.startswith("Y"):
                            new_pw = new_pl = None
                    candidate = {
                        "seq": seq + [tid],
                        "score": score + float(log_prob),
                        "counts": new_cnts,
                        "pending_w": new_pw,
                        "pending_l": new_pl,
                        "adj_state": new_adj_state,
                    }
                    all_candidates.append(candidate)
            if not all_candidates:
                break
            sequences = sorted(all_candidates, key=lambda t: t["score"], reverse=True)[:beam_size]
            if any(cand["seq"][-1] == EOS_ID for cand in sequences):
                break
    best_seq = sequences[0]["seq"] if sequences else list(prefix_ids)
    return _trim_sequence(best_seq)


def guided_decode(
    model,
    prefix_ids,
    max_len: int = 160,
    tokenizer=None,
    required_counts=None,
    bias_tokens=None,
    max_width: float | None = None,
    max_length: float | None = None,
    adjacency_requirements=None,
    adjacency_bias_strength: float = 2.0,
    top_k: int = 8,
    beam_size: int = 8,
    constraint_validator=None,
    validator_min_rooms: int = 1,
):
    """Decode while pruning candidates that violate constraints mid-stream."""
    if tokenizer is None:
        raise ValueError("guided decoding requires a tokenizer")
    if constraint_validator is None:
        raise ValueError("guided decoding requires a constraint_validator callable")
    if top_k < 1 or beam_size < 1:
        raise ValueError("top_k and beam_size must be positive")

    model.eval()
    w_tokens, l_tokens, x_tokens, y_tokens = _prepare_token_maps(tokenizer)
    adjacency_state = None
    if adjacency_requirements:
        adjacency_state = _AdjacencyState(tokenizer, adjacency_requirements, adjacency_bias_strength)
        for tid in prefix_ids:
            adjacency_state.observe(tid)
    base_counts = {tid: 0 for tid in (required_counts or {})}

    sequences = [
        {
            "seq": list(prefix_ids),
            "score": 0.0,
            "counts": dict(base_counts),
            "pending_width": None,
            "pending_length": None,
            "adj_state": adjacency_state.clone() if adjacency_state else None,
        }
    ]
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(max_len):
            expanded = []
            for cand in sequences:
                seq = cand["seq"]
                counts = cand["counts"]
                pending_w = cand["pending_width"]
                pending_l = cand["pending_length"]
                adj_state = cand["adj_state"]

                x = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model(x)[:, -1, :]
                step_bias = dict(bias_tokens or {})
                if adj_state:
                    for tid, bonus in adj_state.current_bias().items():
                        step_bias[tid] = step_bias.get(tid, 0.0) + bonus
                logits = _apply_room_bias(logits, counts, required_counts, extra_bias=step_bias)
                logits = _apply_bounds_mask(
                    logits,
                    pending_w,
                    pending_l,
                    w_tokens,
                    l_tokens,
                    x_tokens,
                    y_tokens,
                    max_width,
                    max_length,
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, top_k)
                for log_prob, idx in zip(topk_log_probs[0], topk_ids[0]):
                    tid = int(idx)
                    new_seq = seq + [tid]
                    new_counts = dict(counts)
                    if tid in new_counts:
                        new_counts[tid] += 1
                    new_pw, new_pl = pending_w, pending_l
                    new_adj_state = adj_state.clone() if adj_state else None
                    tok = tokenizer.id_to_token.get(tid, "")
                    if new_adj_state:
                        new_adj_state.observe(tid)
                    if tok.startswith("W"):
                        try:
                            new_pw = int(tok[1:])
                        except ValueError:
                            new_pw = pending_w
                    elif tok.startswith("L"):
                        try:
                            new_pl = int(tok[1:])
                        except ValueError:
                            new_pl = pending_l
                    elif tok.startswith("Y"):
                        new_pw = new_pl = None

                    partial_seq = _trim_sequence(new_seq)
                    layout_partial = tokenizer.decode_layout_tokens_partial(partial_seq)
                    rooms = (layout_partial.get("layout") or {}).get("rooms", [])
                    if len(rooms) >= validator_min_rooms:
                        issues = constraint_validator(layout_partial) or []
                        if issues:
                            continue

                    expanded.append(
                        {
                            "seq": new_seq,
                            "score": cand["score"] + float(log_prob),
                            "counts": new_counts,
                            "pending_width": new_pw,
                            "pending_length": new_pl,
                            "adj_state": new_adj_state,
                        }
                    )
            if not expanded:
                break
            sequences = sorted(expanded, key=lambda c: c["score"], reverse=True)[:beam_size]
            best_complete = [c for c in sequences if c["seq"][-1] == EOS_ID]
            if best_complete:
                for candidate in best_complete:
                    trimmed = _trim_sequence(candidate["seq"])
                    layout_final = tokenizer.decode_layout_tokens(trimmed)
                    issues = constraint_validator(layout_final) or []
                    if not issues:
                        return trimmed

    best_seq = sequences[0]["seq"] if sequences else list(prefix_ids)
    return _trim_sequence(best_seq)

def decode(
    model,
    prefix_ids,
    max_len: int = 160,
    strategy: str = "greedy",
    temperature: float = 1.0,
    beam_size: int = 5,
    required_counts=None,
    bias_tokens=None,
    tokenizer=None,
    max_width: float | None = None,
    max_length: float | None = None,
    adjacency_requirements=None,
    adjacency_bias_strength: float = 2.0,
    constraint_validator=None,
    validator_min_rooms: int = 1,
    guided_top_k: int = 8,
    guided_beam_size: int = 8,
):
    """Unified decoding interface.

    Args:
        strategy: One of ``"greedy"``, ``"sample"``, ``"beam"`` or ``"guided"``.
        temperature: Temperature for sampling when ``strategy=='sample'``.
        beam_size: Beam width used when ``strategy=='beam'``.
    """
    if strategy == "greedy":
        return greedy_decode(
            model,
            prefix_ids,
            max_len,
            required_counts,
            bias_tokens,
            tokenizer,
            max_width,
            max_length,
            adjacency_requirements,
            adjacency_bias_strength,
        )
    if strategy == "sample":
        return sample_decode(
            model,
            prefix_ids,
            max_len,
            temperature,
            required_counts,
            bias_tokens,
            tokenizer,
            max_width,
            max_length,
            adjacency_requirements,
            adjacency_bias_strength,
        )
    if strategy == "beam":
        return beam_search_decode(
            model,
            prefix_ids,
            max_len,
            beam_size,
            required_counts,
            bias_tokens,
            tokenizer,
            max_width,
            max_length,
            adjacency_requirements,
            adjacency_bias_strength,
        )
    if strategy == "guided":
        return guided_decode(
            model,
            prefix_ids,
            max_len,
            tokenizer,
            required_counts,
            bias_tokens,
            max_width,
            max_length,
            adjacency_requirements,
            adjacency_bias_strength,
            guided_top_k,
            guided_beam_size,
            constraint_validator,
            validator_min_rooms,
        )
    raise ValueError(f"Unknown decoding strategy: {strategy}")
