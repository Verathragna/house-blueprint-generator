import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LayoutTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, nhead=8, num_layers=4, dim_ff=512, dropout=0.1, causal: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, vocab_size)
        self.causal = causal
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Upper-triangular mask with -inf above diagonal to prevent attending to future tokens
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x, key_padding_mask=None):
        h = self.embed(x)
        h = self.pos(h)
        attn_mask = None
        if self.causal:
            S = x.size(1)
            attn_mask = self._causal_mask(S, h.device)
        h = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        return self.proj(h)
