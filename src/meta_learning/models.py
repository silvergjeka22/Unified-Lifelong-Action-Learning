import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, MobileNet_V3_Small_Weights

BACKBONE_DIM = 576


def resnet50_lstm_teacher(x, hidden_size, num_classes=10, dropout_p=0.4):
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for p in resnet.parameters():
        p.requires_grad = False
    for p in resnet.layer4.parameters():
        p.requires_grad = True

    backbone = nn.Sequential(*list(resnet.children())[:-2])
    avgpool  = nn.AdaptiveAvgPool2d((1, 1))
    lstm     = nn.LSTM(2048, hidden_size, batch_first=True)
    dropout  = nn.Dropout(dropout_p)
    fc       = nn.Linear(hidden_size, num_classes)

    B, T, C, H, W = x.shape
    feat   = backbone(x.view(B * T, C, H, W))
    pooled = avgpool(feat).view(B, T, 2048)
    out, _ = lstm(pooled)
    h_last = out[:, -1, :]
    return fc(dropout(h_last)), h_last


def mobilenetv3_small_lstm_student(x, num_classes=10, hidden_size=256, dropout_p=0.3):
    mobile   = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone = mobile.features
    pool     = nn.AdaptiveAvgPool2d((1, 1))
    lstm     = nn.LSTM(BACKBONE_DIM, hidden_size, batch_first=True)
    dropout  = nn.Dropout(dropout_p)
    fc       = nn.Linear(hidden_size, num_classes)

    B, T, C, H, W = x.shape
    feat   = backbone(x.view(B * T, C, H, W))
    pooled = pool(feat).view(B, T, BACKBONE_DIM)
    out, _ = lstm(pooled)
    h_last = out[:, -1, :]
    return fc(dropout(h_last)), h_last


def embedding_head(s_emb, student_hidden, teacher_hidden, num_classes=10, dropout_p=0.3, training=True):
    dropout    = nn.Dropout(dropout_p)
    classifier = nn.Linear(student_hidden, num_classes)
    projector  = nn.Linear(student_hidden, teacher_hidden)

    x = dropout(s_emb) if training else s_emb
    return classifier(x), projector(s_emb)