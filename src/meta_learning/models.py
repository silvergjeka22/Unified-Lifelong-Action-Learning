import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, MobileNet_V3_Small_Weights


# =========================================================
# TEACHER MODEL
# =========================================================
class ResNet50LSTMTeacher(nn.Module):
    """
    ResNet-50 backbone (layer4 unfrozen) + LSTM temporal aggregator.
    Returns (logits, lstm_hidden_state).
    """
    def __init__(self, hidden_size: int, num_classes: int = 10, dropout_p: float = 0.4):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for p in resnet.parameters():
            p.requires_grad = False
        for p in resnet.layer4.parameters():
            p.requires_grad = True

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm     = nn.LSTM(2048, hidden_size, batch_first=True)
        self.dropout  = nn.Dropout(dropout_p)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat   = self.backbone(x.view(B * T, C, H, W))
        pooled = self.avgpool(feat).view(B, T, 2048)
        out, _ = self.lstm(pooled)
        h_last = out[:, -1, :]
        return self.fc(self.dropout(h_last)), h_last


# =========================================================
# STUDENT MODEL
# =========================================================
class MobileNetV3SmallLSTMStudent(nn.Module):
    """
    MobileNetV3-Small backbone + LSTM temporal aggregator.
    Returns (logits, lstm_hidden_state).
    """
    BACKBONE_DIM = 576

    def __init__(self, num_classes: int = 10, hidden_size: int = 256, dropout_p: float = 0.3):
        super().__init__()
        mobile = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobile.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm     = nn.LSTM(self.BACKBONE_DIM, hidden_size, batch_first=True)
        self.dropout  = nn.Dropout(dropout_p)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat   = self.backbone(x.view(B * T, C, H, W))
        pooled = self.pool(feat).view(B, T, self.BACKBONE_DIM)
        out, _ = self.lstm(pooled)
        h_last = out[:, -1, :]
        return self.fc(self.dropout(h_last)), h_last


# =========================================================
# EMBEDDING HEAD
# =========================================================
class EmbeddingHead(nn.Module):
    """
    Lightweight classification + projection head trained on top of
    pre-extracted student embeddings.

    Outputs:
        logits  — class scores (student_hidden → num_classes)
        proj    — projected embedding (student_hidden → teacher_hidden)
                  used for MSE distillation against teacher embeddings
    """
    def __init__(
        self,
        student_hidden: int,
        teacher_hidden: int,
        num_classes: int = 10,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.dropout    = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(student_hidden, num_classes)
        self.projector  = nn.Linear(student_hidden, teacher_hidden)

    def forward(self, s_emb, training=True):
        x = self.dropout(s_emb) if training else s_emb
        return self.classifier(x), self.projector(s_emb)