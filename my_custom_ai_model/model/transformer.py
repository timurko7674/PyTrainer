import torch.nn as nn
import torch


class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_size=256, n_heads=4, n_layers=4, seq_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb[:, :x.size(1)]
        x = self.transformer(x)
        return self.fc(x)