import logging 

import torch.nn as nn
import torch.F as F
import torch

logger = logging.getLogger(__name__)

class GatedAttentionPooling(nn.Module):
    """
    Gated attention mechanism for MIL
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim) # feature transformation
        self.U = nn.Linear(input_dim, hidden_dim) # gating mechanism
        self.w = nn.Linear(hidden_dim, 1) # cal logits

    def forward(self, h):
        tanh_val = torch.tanh(self.V(h))
        sigm_val = torch.sigmoid(self.U(h))

        gated_out = tanh_val * sigm_val 

        scores = self.w(gated_out)
        weights = F.softmax(scores, dim=0)
        aggregate_feature = torch.sum(weights * h, dim=0, keepdim=True)

        return aggregate_feature, weights

