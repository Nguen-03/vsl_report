import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=512, out_features=3315, s=20.0, m=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.set_margin(m)

    def set_margin(self, m):
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        if input.ndim != 2 or input.size(1) != self.in_features:
            raise ValueError(
                f"Expected embeddings with shape (batch, {self.in_features}), "
                f"got {tuple(input.shape)}"
            )
        if label.ndim != 1 or label.size(0) != input.size(0):
            raise ValueError("Labels must have shape (batch,)")
        if torch.any((label < 0) | (label >= self.out_features)):
            raise ValueError("Labels must be in [0, out_features)")

        cosine = F.linear(F.normalize(input, dim=1), F.normalize(self.weight, dim=1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if self.m == 0.0:
            return self.s * cosine

        sine = torch.sqrt(1.0 - cosine.square())
        phi = cosine * self.cos_m - sine * self.sin_m
       
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.to(device=input.device, dtype=torch.long).unsqueeze(1), 1.0)
        return self.s * (one_hot * phi + (1.0 - one_hot) * cosine)
