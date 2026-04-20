# Run from notebook: %run /content/src/models/pretrained.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    DenseNet121_Weights,
    VGG19_BN_Weights,
)
from config.config import SELECTED_CLASSES
from config.config import DROPOUT_P

num_classes = len(SELECTED_CLASSES)


#  ResNet18 + LSTM
class ResNet18LSTM(nn.Module):
    """
    ResNet18 (frozen except layer4) + LSTM classifier.
    Input : [B, T, C, H, W]
    Output: [B, num_classes]
    """
    def __init__(self, hidden_size=256, num_classes=num_classes):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
        self.lstm   = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc     = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x        = x.view(B * T, C, H, W)
        features = self.resnet(x)               # [B*T, 512, 1, 1]
        features = features.view(B, T, -1)      # [B, T, 512]
        out, _   = self.lstm(features)
        return self.fc(out[:, -1, :])


#  ResNet50 + LSTM
class ResNet50LSTM(nn.Module):
    """
    ResNet50 (frozen except layer4) + LSTM + Dropout classifier.

    Args:
        hidden_size : LSTM hidden state size         (default: 256)
        num_classes : output classes                 (default: num_classes)
        dropout_p   : dropout probability before fc  (default: DROPOUT_P)
                      0.4 for base/task1, lower to 0.3 for task2+

    Input : [B, T, C, H, W]
    Output: [B, num_classes]
    """
    def __init__(self, hidden_size=256, num_classes=num_classes, dropout_p=None):
        super().__init__()

        if dropout_p is None:
            dropout_p = DROPOUT_P

        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # freeze all, then unfreeze layer4 only
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet  = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
        self.lstm    = nn.LSTM(input_size=2048, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        features = self.resnet(x.view(B * T, C, H, W))  # [B*T, 2048, 1, 1]
        features = features.view(B, T, -1)               # [B, T, 2048]
        out, _   = self.lstm(features)                   # [B, T, hidden]
        return self.fc(self.dropout(out[:, -1, :]))       # last hidden state only


def expand_classifier(model, new_num_classes):
    """
    Grow model.fc to new_num_classes, preserving weights from old classes.
    Dropout layer is carried over unchanged — update cfg.DROPOUT_P before
    calling this if you want a lower rate for the new task.
    """
    old_fc = model.fc
    new_fc = nn.Linear(old_fc.in_features, new_num_classes)

    # copy existing weights and biases; new rows keep random init
    with torch.no_grad():
        new_fc.weight[:old_fc.out_features] = old_fc.weight
        new_fc.bias[:old_fc.out_features]   = old_fc.bias

    model.fc = new_fc
    return model


def update_dropout(model, new_p):
    """
    Update the dropout probability in-place without rebuilding the model.
    Call this before fine-tuning each new task.

    Usage:
        update_dropout(model, new_p=0.3)   # before Task 2 / 3 / 4
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = new_p
    print(f"Dropout updated to p={new_p}")
    return model


def unfreeze_all(model):
    """Make every parameter trainable (naive upper-bound baseline)."""
    for param in model.parameters():
        param.requires_grad = True
    return model


#  DenseNet121 + LSTM
class DenseNet121LSTM(nn.Module):
    """
    DenseNet121 (frozen except denseblock4 + norm5) + LSTM classifier.
    Input : [B, T, C, H, W]
    Output: [B, num_classes]
    """
    def __init__(self, hidden_size=256, num_classes=num_classes):
        super().__init__()
        dnet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        for p in dnet.features.parameters():
            p.requires_grad = False
        for p in dnet.features.denseblock4.parameters():
            p.requires_grad = True
        for p in dnet.features.norm5.parameters():
            p.requires_grad = True

        self.features    = dnet.features
        self.feature_dim = 1024

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x     = x.view(B * T, C, H, W)
        feats = self.features(x)            # [B*T, 1024, H', W']
        feats = feats.mean(dim=[2, 3])      # GAP -> [B*T, 1024]
        feats = feats.view(B, T, -1)        # [B, T, 1024]
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])


#  VGG19-BN + LSTM
class VGG19BNLSTM(nn.Module):
    """
    VGG19 with BN (frozen except last conv block) + LSTM classifier.
    Input : [B, T, C, H, W]
    Output: [B, num_classes]
    """
    def __init__(self, hidden_size=256, num_classes=num_classes):
        super().__init__()
        vgg = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)

        for param in vgg.features.parameters():
            param.requires_grad = False
        for param in vgg.features[34:].parameters():
            param.requires_grad = True

        self.features = vgg.features
        self.lstm     = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x        = x.view(B * T, C, H, W)
        features = self.features(x)             # [B*T, 512, 7, 7]
        features = features.mean([2, 3])        # GAP -> [B*T, 512]
        features = features.view(B, T, -1)      # [B, T, 512]
        out, _   = self.lstm(features)
        return self.fc(out[:, -1, :])
