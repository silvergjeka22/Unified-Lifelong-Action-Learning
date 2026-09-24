import os
import sys
import gc
import subprocess
from getpass import getpass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------------
# Feature Extraction Utilities
# -------------------------------------------------------------------------
def extract_clip_video_features(loader, clip_model, device):
    """Passes video clips through the frozen CLIP image encoder and calculates
    temporally-averaged 512-dim normalized video feature vectors."""
    all_features, all_labels = [], []

    with torch.no_grad():
        for videos, labels in loader:
            if videos.ndim == 5:
                if videos.shape[1] == 3:  # Convert (B, C, T, H, W) -> (B, T, C, H, W)
                    videos = videos.permute(0, 2, 1, 3, 4)
                
                B, T, C, H, W = videos.shape
                frames = videos.reshape(B * T, C, H, W).to(device)
                
                frame_feats = clip_model.encode_image(frames)          # (B*T, 512)
                video_feats = frame_feats.reshape(B, T, -1).mean(dim=1) # (B, 512)
            else:
                video_feats = clip_model.encode_image(videos.to(device))

            video_feats = F.normalize(video_feats, p=2, dim=-1)
            all_features.append(video_feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

def get_text_prototypes(class_names, clip_model, device):
    """Extracts normalized 512-dim CLIP text prompt embeddings for target classes."""
    import clip
    prompts = [f"a video of a person performing {cls.replace('_', ' ')}" for cls in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
    return text_feats.cpu().numpy()

# -------------------------------------------------------------------------
# CLIP Evaluation Helper (Method 1)
# -------------------------------------------------------------------------
def evaluate_hybrid_prototypes(X_train, y_train, X_eval, y_eval, text_proto, alpha=0.4):
    """Blends zero-shot CLIP text prototypes with few-shot visual centroids."""
    num_classes = len(text_proto)
    visual_centroids = []
    
    for c in range(num_classes):
        cls_mask = (y_train == c)
        if np.sum(cls_mask) > 0:
            c_mean = X_train[cls_mask].mean(axis=0)
            c_mean = c_mean / np.linalg.norm(c_mean)
        else:
            c_mean = np.zeros(512)
        visual_centroids.append(c_mean)
        
    visual_centroids = np.array(visual_centroids)
    
    hybrid_proto = alpha * text_proto + (1.0 - alpha) * visual_centroids
    hybrid_proto = hybrid_proto / np.linalg.norm(hybrid_proto, axis=1, keepdims=True)
    
    cosine_sims = X_eval @ hybrid_proto.T
    preds = np.argmax(cosine_sims, axis=1)
    return np.mean(preds == y_eval) * 100.0

# -------------------------------------------------------------------------
# CLIP-MLP Adapter Architecture
# -------------------------------------------------------------------------
class CLIPMLPAdapter(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, num_classes=101, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)