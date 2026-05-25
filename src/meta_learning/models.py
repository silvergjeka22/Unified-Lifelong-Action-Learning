import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, MobileNet_V3_Small_Weights


class ResNet50LSTMTeacher(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 10, dropout_p: float = 0.4):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        for p in resnet.parameters():
            p.requires_grad = False

        for p in resnet.layer4.parameters():
            p.requires_grad = True

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        b, t, c, h, w = x.shape
        feat = self.backbone(x.view(b * t, c, h, w)).view(b * t, -1)
        pooled = feat.view(b, t, 2048)
        out, _ = self.lstm(pooled)
        h_last = out[:, -1, :]
        logits = self.fc(self.dropout(h_last))
        return logits, h_last


class MobileNetV3SmallLSTMStudent(nn.Module):
    BACKBONE_DIM = 576

    def __init__(self, num_classes: int = 10, hidden_size: int = 128, dropout_p: float = 0.3):
        super().__init__()
        mobile = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobile.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(self.BACKBONE_DIM, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        b, t, c, h, w = x.shape
        feat = self.backbone(x.view(b * t, c, h, w))
        pooled = self.pool(feat).view(b, t, self.BACKBONE_DIM)
        out, _ = self.lstm(pooled)
        h_last = out[:, -1, :]
        logits = self.fc(self.dropout(self.norm(h_last)))
        return logits, h_last


class EmbeddingHead(nn.Module):
    def __init__(self, student_hidden: int, teacher_hidden: int, num_classes: int = 10, dropout_p: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(student_hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(student_hidden, num_classes)
        self.projector = nn.Linear(student_hidden, teacher_hidden)

    def forward(self, s_emb: torch.Tensor, training: bool = True):
        normed = self.norm(s_emb)
        x = self.dropout(normed) if training else normed
        logits = self.classifier(x)
        proj = self.projector(normed)
        return logits, proj