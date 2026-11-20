import torch
import torch.nn as nn
class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        """Wrap an encoder that maps one 60-frame, selected-joint video to an embedding."""
        super().__init__()
        self.encoder = encoder

    def forward_once(self, sequence):
        return self.encoder(sequence)

    def forward(self, anchor, positive=None, negative=None):
        if positive is None and negative is None:
            return self.forward_once(anchor)
        if positive is None or negative is None:
            raise ValueError("Provide both positive and negative sequences for triplet input")
        return (
            self.forward_once(anchor),
            self.forward_once(positive),
            self.forward_once(negative),
        )