"""Decoding utilities for blueprint token sequences.

This module centralises different strategies used during autoregressive
generation.  In addition to simple greedy decoding, it supports
temperature-based sampling and beam search.  The helper :func:`decode`
provides a unified entry point selecting between the strategies.
"""

import torch
from tokenizer.tokenizer import SEP_ID, EOS_ID


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


def greedy_decode(
    model,
    prefix_ids,
    max_len: int = 160,
    required_counts=None,
    bias_tokens=None,
    tokenizer=None,
    max_width: float | None = None,
    max_length: float | None = None,
):
    """Generate tokens using greedy argmax decoding."""
    model.eval()
    seq = list(prefix_ids)
    device = next(model.parameters()).device
    counts = {tid: 0 for tid in (required_counts or {})}
    pending_w = pending_l = None
    if tokenizer:
        w_tokens, l_tokens, x_tokens, y_tokens = _prepare_token_maps(tokenizer)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]
            logits = _apply_room_bias(logits, counts, required_counts, extra_bias=bias_tokens)
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
    if tokenizer:
        w_tokens, l_tokens, x_tokens, y_tokens = _prepare_token_maps(tokenizer)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :] / temperature
            logits = _apply_room_bias(logits, counts, required_counts, extra_bias=bias_tokens)
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
    sequences = [
        (list(prefix_ids), 0.0, {tid: 0 for tid in (required_counts or {})}, None, None)
    ]  # (seq, score, counts, pending_w, pending_l)
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for seq, score, cnts, pw, pl in sequences:
                x = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model(x)[:, -1, :]
                logits = _apply_room_bias(logits, cnts, required_counts, extra_bias=bias_tokens)
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
                    if tokenizer:
                        tok = tokenizer.id_to_token.get(tid, "")
                        if tok.startswith("W"):
                            new_pw = int(tok[1:])
                        elif tok.startswith("L"):
                            new_pl = int(tok[1:])
                        elif tok.startswith("Y"):
                            new_pw = new_pl = None
                    candidate = (seq + [tid], score + float(log_prob), new_cnts, new_pw, new_pl)
                    all_candidates.append(candidate)
            sequences = sorted(all_candidates, key=lambda t: t[1], reverse=True)[:beam_size]
            if any(seq[-1] == EOS_ID for seq, _, _, _, _ in sequences):
                break
    best_seq = sequences[0][0]
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
):
    """Unified decoding interface.

    Args:
        strategy: One of ``"greedy"``, ``"sample"`` or ``"beam"``.
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
        )
    raise ValueError(f"Unknown decoding strategy: {strategy}")
