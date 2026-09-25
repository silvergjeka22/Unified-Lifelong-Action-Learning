import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from src.config.config import SELECTED_CLASSES, HIDDEN_SIZE

NUM_CLASSES = len(SELECTED_CLASSES)


class MobileNetV3SmallLSTMStudent(nn.Module):
    """MobileNetV3-Small + LSTM: the small student, with the same 256-d LSTM as the teacher.

    forward(x) -> logits, features(x) -> 256-d LSTM state
    """

    BACKBONE_DIM = 576

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE, dropout_p=0.3):
        super().__init__()
        mobile = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobile.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm     = nn.LSTM(self.BACKBONE_DIM, hidden_size, batch_first=True)
        self.dropout  = nn.Dropout(dropout_p)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def features(self, x):
        B, T, C, H, W = x.shape
        feats  = self.pool(self.backbone(x.view(B * T, C, H, W))).view(B, T, self.BACKBONE_DIM)
        out, _ = self.lstm(feats)
        return out[:, -1, :]

    def forward(self, x):
        return self.fc(self.dropout(self.features(x)))
