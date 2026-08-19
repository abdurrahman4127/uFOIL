import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerFormatterBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TransformerFormatter(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerFormatterBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) -> (batch, seq_len, vocab_size) logits."""
        x = self.embed(token_ids)
        x = self.pos_encode(x)

        for block in self.blocks:
            x = block(x)

        return self.out_proj(x)


def tokenize(text: str, char_to_idx: dict[str, int], unk_idx: int = 0) -> list[int]:
    return [char_to_idx.get(c, unk_idx) for c in text]


def build_vocab(texts: list[str]) -> dict[str, int]:
    """char-level vocab, index 0 reserved for unknown/padding chars."""
    chars = sorted(set("".join(texts)))
    vocab = {"<unk>": 0}
    for i, c in enumerate(chars, start=1):
        vocab[c] = i
    return vocab


# idx_to_char[0] is the literal string "<unk>" (5 chars), so joining
# it straight in silently stretches the output whenever the model
# predicts unknown - swap it for a single placeholder char instead.
def decode_logits(logits: torch.Tensor, idx_to_char: dict[int, str]) -> str:
    ids = logits.argmax(dim=-1).squeeze(0).tolist()
    chars = []
    for i in ids:
        c = idx_to_char.get(i, "")
        chars.append("?" if c == "<unk>" else c)
    return "".join(chars)


def structure_output(
    formatter: TransformerFormatter,
    raw_text: str,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
) -> str:
    ids = tokenize(raw_text, char_to_idx)
    token_ids = torch.tensor([ids], dtype=torch.long)

    formatter.eval()
    with torch.no_grad():
        logits = formatter(token_ids)

    return decode_logits(logits, idx_to_char)