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
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Features
def extract_clip_video_features(loader, clip_model, device):
    """Encode every frame with the frozen CLIP image encoder and average them into one 512-d feature per clip."""
    all_features, all_labels = [], []

    with torch.no_grad():
        for videos, labels in loader:
            if videos.ndim == 5:
                if videos.shape[1] == 3:  # (B, C, T, H, W) -> (B, T, C, H, W)
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
    """CLIP text feature of every class name (normalised)."""
    import clip
    prompts = [f"a video of a person performing {cls.replace('_', ' ')}" for cls in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
    return text_feats.cpu().numpy()

# Hybrid prototypes
def evaluate_hybrid_prototypes(X_train, y_train, X_eval, y_eval, text_proto, alpha=0.4):
    """Classify by cosine to prototypes that mix the text feature and the mean clip feature. Returns accuracy in %."""
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

# MLP adapter
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


def display_fewshot_results(df_results, resnet_loss_curves, mlp_loss_curves, k_shots_list):
    """
    Displays a styled Pandas summary table and plots training loss curves 
    for ResNet50+LSTM and CLIP+MLP Adapter across k-shots.
    """
    # --- 1. Formatted Results Table ---
    df_display = df_results.copy()
    df_display.columns = ['k (Shot Size)', 'ResNet50 + LSTM Acc (%)', 'CLIP + MLP Adapter Acc (%)']

    styled_df = df_display.style\
        .format({'ResNet50 + LSTM Acc (%)': '{:.2f}%', 'CLIP + MLP Adapter Acc (%)': '{:.2f}%'})\
        .background_gradient(cmap='Blues', subset=['ResNet50 + LSTM Acc (%)', 'CLIP + MLP Adapter Acc (%)'])\
        .set_properties(**{'text-align': 'center', 'font-size': '13px', 'padding': '8px'})\
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#2b2b2b'), ('color', 'white'), ('text-align', 'center')]},
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '16px'), ('font-weight', 'bold')]}
        ])\
        .set_caption("Few-Shot Evaluation Performance (k-shots)")

    display(styled_df)

    # --- 2. Learning Curves Visualization ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    active_k_shots = [k for k in k_shots_list if k > 0]
    colors = sns.color_palette("viridis", len(active_k_shots))

    # ResNet50 + LSTM Loss Plot
    ax1 = axes[0]
    for idx, k in enumerate(active_k_shots):
        if k in resnet_loss_curves and len(resnet_loss_curves[k]) > 0:
            ax1.plot(range(1, len(resnet_loss_curves[k]) + 1), resnet_loss_curves[k], marker='o', label=f'k = {k}', color=colors[idx])
    ax1.set_title("ResNet50 + LSTM Training Loss", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.legend(title="k-shots")

    # CLIP + MLP Adapter Loss Plot
    ax2 = axes[1]
    for idx, k in enumerate(active_k_shots):
        if k in mlp_loss_curves and len(mlp_loss_curves[k]) > 0:
            ax2.plot(range(1, len(mlp_loss_curves[k]) + 1), mlp_loss_curves[k], label=f'k = {k}', color=colors[idx], linewidth=1.8)
    ax2.set_title("CLIP + MLP Adapter Training Loss", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax2.legend(title="k-shots")

    plt.tight_layout()
    plt.show()