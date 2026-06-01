import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50LSTMTeacher(nn.Module):
    """
    ResNet-50 backbone (layer4 unfrozen) + single-layer LSTM classifier.
    Input : (B, T, C, H, W) video clips
    Output: (logits, hidden_state)
    """

    def __init__(self, hidden_size: int, num_classes: int = 10, dropout_p: float = 0.4):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for p in resnet.parameters():
            p.requires_grad = False
        for p in resnet.layer4.parameters():
            p.requires_grad = True
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm     = nn.LSTM(2048, hidden_size, batch_first=True)
        self.dropout  = nn.Dropout(dropout_p)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        feat   = self.backbone(x.view(B * T, C, H, W)).view(B * T, -1)
        pooled = feat.view(B, T, 2048)
        out, _ = self.lstm(pooled)
        h_last = out[:, -1, :]
        return self.fc(self.dropout(h_last)), h_last
