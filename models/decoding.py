"""Decoding utilities for blueprint token sequences.

This module centralises different strategies used during autoregressive
generation.  In addition to simple greedy decoding, it supports
temperature-based sampling and beam search.  The helper :func:`decode`
provides a unified entry point selecting between the strategies.
"""

import torch
from tokenizer.tokenizer import SEP_ID, EOS_ID


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


def greedy_decode(model, prefix_ids, max_len: int = 160, required_counts=None, bias_tokens=None):
    """Generate tokens using greedy argmax decoding."""
    model.eval()
    seq = list(prefix_ids)
    device = next(model.parameters()).device
    counts = {tid: 0 for tid in (required_counts or {})}
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]
            logits = _apply_room_bias(logits, counts, required_counts, extra_bias=bias_tokens)
            nxt = int(logits.argmax(dim=-1))
            seq.append(nxt)
            if nxt in counts:
                counts[nxt] += 1
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
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :] / temperature
            logits = _apply_room_bias(logits, counts, required_counts, extra_bias=bias_tokens)
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1))
            seq.append(nxt)
            if nxt in counts:
                counts[nxt] += 1
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
):
    """Generate tokens using beam search.

    Args:
        beam_size: Number of beams to keep during search. Must be >= 1.
    """
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    model.eval()
    sequences = [(list(prefix_ids), 0.0, {tid: 0 for tid in (required_counts or {})})]  # (seq, score, counts)
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for seq, score, cnts in sequences:
                x = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model(x)[:, -1, :]
                logits = _apply_room_bias(logits, cnts, required_counts, extra_bias=bias_tokens)
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for log_prob, idx in zip(topk_log_probs[0], topk_ids[0]):
                    new_cnts = dict(cnts)
                    tid = int(idx)
                    if tid in new_cnts:
                        new_cnts[tid] += 1
                    candidate = (seq + [tid], score + float(log_prob), new_cnts)
                    all_candidates.append(candidate)
            sequences = sorted(all_candidates, key=lambda t: t[1], reverse=True)[:beam_size]
            if any(seq[-1] == EOS_ID for seq, _, _ in sequences):
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
):
    """Unified decoding interface.

    Args:
        strategy: One of ``"greedy"``, ``"sample"`` or ``"beam"``.
        temperature: Temperature for sampling when ``strategy=='sample'``.
        beam_size: Beam width used when ``strategy=='beam'``.
    """
    if strategy == "greedy":
        return greedy_decode(model, prefix_ids, max_len, required_counts, bias_tokens)
    if strategy == "sample":
        return sample_decode(model, prefix_ids, max_len, temperature, required_counts, bias_tokens)
    if strategy == "beam":
        return beam_search_decode(model, prefix_ids, max_len, beam_size, required_counts, bias_tokens)
    raise ValueError(f"Unknown decoding strategy: {strategy}")
