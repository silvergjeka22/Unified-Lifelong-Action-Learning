import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights


class MobileNetV3SmallLSTMStudent(nn.Module):
    """
    MobileNetV3-Small backbone + single-layer LSTM classifier.
    Lightweight student for Knowledge Distillation.
    Input : (B, T, C, H, W) video clips
    Output: (logits, hidden_state)
    """

    BACKBONE_DIM = 576

    def __init__(self, num_classes: int = 10, hidden_size: int = 128, dropout_p: float = 0.3):
        super().__init__()
        mobile        = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobile.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm     = nn.LSTM(self.BACKBONE_DIM, hidden_size, batch_first=True)
        self.norm     = nn.LayerNorm(hidden_size)
        self.dropout  = nn.Dropout(dropout_p)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        feat   = self.backbone(x.view(B * T, C, H, W))
        pooled = self.pool(feat).view(B, T, self.BACKBONE_DIM)
        out, _ = self.lstm(pooled)
        h_last = out[:, -1, :]
        logits = self.fc(self.dropout(self.norm(h_last)))
        return logits, h_last
