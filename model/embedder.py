import torch
import torch.nn as nn
import torch.nn.functional as F
from config import STREAM_DIM

class VSL_Embedder(nn.Module):
    def __init__(self, stream_dim=STREAM_DIM, in_coords=3, hidden_dim=256, embed_dim=512):
        super().__init__()
        self.pos_dim = stream_dim
        self.vel_dim = stream_dim

        self.pos_encoder = nn.Sequential(
            nn.Linear(self.pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.vel_encoder = nn.Sequential(
            nn.Linear(self.vel_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, padding=2, groups=hidden_dim * 2),
                nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            for _ in range(3)
        ])
        self.conv_norms = nn.ModuleList([nn.LayerNorm(hidden_dim * 2) for _ in range(3)])

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x, mask=None, normalize=True):
        if mask is None:
            mask = (x.abs().sum(dim=-1) > 1e-6).float()

        pos = x[:, :, :self.pos_dim]
        vel = x[:, :, self.pos_dim:self.pos_dim + self.vel_dim]

        h = torch.cat([self.pos_encoder(pos), self.vel_encoder(vel)], dim=-1)
        h = h * mask.unsqueeze(-1)

        for conv, norm in zip(self.conv_blocks, self.conv_norms):
            res = h
            h_c = conv(h.transpose(1, 2)).transpose(1, 2)
            h = norm(res + h_c) * mask.unsqueeze(-1)

        # Chạy thẳng Bi-LSTM trên GPU và lọc padding bằng mask (tránh deadlock DataParallel)
        lstm_out, _ = self.lstm(h)
        lstm_out = lstm_out * mask.unsqueeze(-1)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=1)
        features = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)

        out = self.head(features)
        if normalize:
            return F.normalize(out, p=2, dim=1)
        return out
if __name__ == '__main__':
    pass