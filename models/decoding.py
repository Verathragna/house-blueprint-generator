import torch
from tokenizer.tokenizer import SEP_ID, EOS_ID


def _trim_sequence(seq):
    """Return portion after SEP_ID if present."""
    if SEP_ID in seq:
        i = seq.index(SEP_ID) + 1
        return seq[i:]
    return seq


def greedy_decode(model, prefix_ids, max_len: int = 160):
    """Generate tokens using greedy argmax decoding."""
    model.eval()
    seq = list(prefix_ids)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)[:, -1, :]
            nxt = int(logits.argmax(dim=-1))
            seq.append(nxt)
            if nxt == EOS_ID:
                break
    return _trim_sequence(seq)


def sample_decode(
    model,
    prefix_ids,
    max_len: int = 160,
    temperature: float = 1.0,
):
    """Generate tokens using temperature sampling."""
    model.eval()
    seq = list(prefix_ids)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1))
            seq.append(nxt)
            if nxt == EOS_ID:
                break
    return _trim_sequence(seq)


def beam_search_decode(
    model,
    prefix_ids,
    max_len: int = 160,
    beam_size: int = 5,
):
    """Generate tokens using beam search."""
    model.eval()
    sequences = [(list(prefix_ids), 0.0)]  # (seq, score)
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                x = torch.tensor([seq], dtype=torch.long)
                logits = model(x)[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for log_prob, idx in zip(topk_log_probs[0], topk_ids[0]):
                    candidate = (seq + [int(idx)], score + float(log_prob))
                    all_candidates.append(candidate)
            sequences = sorted(all_candidates, key=lambda t: t[1], reverse=True)[:beam_size]
            if any(seq[-1] == EOS_ID for seq, _ in sequences):
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
):
    """Unified decoding interface."""
    if strategy == "greedy":
        return greedy_decode(model, prefix_ids, max_len)
    if strategy == "sample":
        return sample_decode(model, prefix_ids, max_len, temperature)
    if strategy == "beam":
        return beam_search_decode(model, prefix_ids, max_len, beam_size)
    raise ValueError(f"Unknown decoding strategy: {strategy}")
