import torch
import torch.nn as nn


class CharBLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        vocab_size: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # x2 since bidirectional concats forward + backward hidden states
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, seq_len, vocab_size)
        out, _ = self.lstm(x)
        return self.classifier(out)


def decode_greedy(logits: torch.Tensor, idx_to_char: dict[int, str]) -> list[str]:
    pred_ids = logits.argmax(dim=-1)  # (batch, seq_len)

    decoded = []
    for seq in pred_ids:
        chars = [idx_to_char.get(int(i), "") for i in seq]
        decoded.append("".join(chars))

    return decoded


def sequence_probability(logits: torch.Tensor) -> torch.Tensor:
    # P(Y|X) as the product of per-timestep max softmax probs
    # logits: (batch, seq_len, vocab_size) -> (batch,)
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return max_probs.prod(dim=-1)
