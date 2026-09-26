import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, VGG19_BN_Weights, ViT_B_16_Weights,
)

from src.config.config import SELECTED_CLASSES, DROPOUT_P, HIDDEN_SIZE

NUM_CLASSES = len(SELECTED_CLASSES)


class ScratchCNNLSTM(nn.Module):
    """Small CNN + LSTM trained from scratch. The lower-bound baseline."""

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.lstm  = nn.LSTM(128 * 28 * 28, hidden_size, batch_first=True)
        self.fc    = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(B, T, -1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ResNet18LSTM(nn.Module):
    """ResNet18 (layer4 unfrozen) + LSTM."""

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for p in resnet.parameters():
            p.requires_grad = False
        for p in resnet.layer4.parameters():
            p.requires_grad = True
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm     = nn.LSTM(512, hidden_size, batch_first=True)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats  = self.backbone(x.view(B * T, C, H, W)).view(B, T, 512)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])


class ResNet50LSTM(nn.Module):
    """ResNet50 (layer4 trainable) + LSTM: the teacher.

    forward(x) -> logits, features(x) -> 256-d LSTM state
    """

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE, dropout_p=DROPOUT_P):
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

    def features(self, x):
        B, T, C, H, W = x.shape
        feats  = self.backbone(x.view(B * T, C, H, W)).view(B, T, 2048)
        out, _ = self.lstm(feats)
        return out[:, -1, :]

    def forward(self, x):
        return self.fc(self.dropout(self.features(x)))


class DenseNet121LSTM(nn.Module):
    """DenseNet121 (last block unfrozen) + LSTM."""

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE):
        super().__init__()
        dnet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        for p in dnet.features.parameters():
            p.requires_grad = False
        for p in dnet.features.denseblock4.parameters():
            p.requires_grad = True
        for p in dnet.features.norm5.parameters():
            p.requires_grad = True
        self.features_net = dnet.features
        self.lstm = nn.LSTM(1024, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats  = self.features_net(x.view(B * T, C, H, W)).mean(dim=[2, 3]).view(B, T, 1024)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])


class VGG19BNLSTM(nn.Module):
    """VGG19-BN (last conv block unfrozen) + LSTM."""

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE):
        super().__init__()
        vgg = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        for p in vgg.features.parameters():
            p.requires_grad = False
        for p in vgg.features[34:].parameters():
            p.requires_grad = True
        self.features_net = vgg.features
        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats  = self.features_net(x.view(B * T, C, H, W)).mean(dim=[2, 3]).view(B, T, 512)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])


class ViTLSTM(nn.Module):
    """ViT-B/16 (last 2 of 12 encoder blocks unfrozen) + 2-layer LSTM."""

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE):
        super().__init__()
        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()                  # keep the 768-d class token
        for p in vit.parameters():
            p.requires_grad = False
        for block in vit.encoder.layers[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        self.backbone = vit
        self.lstm     = nn.LSTM(768, hidden_size, num_layers=2, batch_first=True)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats  = self.backbone(x.view(B * T, C, H, W)).view(B, T, 768)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])


def expand_classifier(model, new_num_classes):
    """Grow model.fc to new_num_classes, keeping the old-class rows. Returns the model."""
    old = model.fc
    new = nn.Linear(old.in_features, new_num_classes)
    with torch.no_grad():
        new.weight[:old.out_features] = old.weight
        new.bias[:old.out_features]   = old.bias
    model.fc = new
    return model


def unfreeze_all(model):
    """Make every parameter trainable. Returns the model."""
    for p in model.parameters():
        p.requires_grad = True
    return model
