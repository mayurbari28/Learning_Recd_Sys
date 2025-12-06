import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, num_items, hidden=64, heads=2, layers=2, max_len=50):
        super().__init__()
        self.item_emb = nn.Embedding(num_items+1, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(hidden, num_items)

    def forward(self, seq):
        positions = torch.arange(seq.size(1)).to(seq.device)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.transformer(x)
        out = self.fc(x[:, -1])
        return out
