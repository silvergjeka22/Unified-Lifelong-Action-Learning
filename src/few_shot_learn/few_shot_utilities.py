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

def display_fewshot_results(df_results, resnet_histories, mlp_histories, k_shots_list):
    """
    Displays summary table, loss curves (Train/Val), epoch accuracy curves (Train/Val), 
    and final test accuracy over k-shots. Supports both dictionary and list history structures.
    """
    sns.set_theme(style="whitegrid")
    active_k_shots = [k for k in k_shots_list if k > 0]
    colors = sns.color_palette("viridis", len(active_k_shots))

    # Helper function to unpack metrics whether history is a dict or a direct list
    def unpack_history(hist):
        if isinstance(hist, dict):
            train_loss = hist.get('train_losses', hist.get('train_loss', []))
            val_loss = hist.get('val_losses', hist.get('val_loss', []))
            train_acc = hist.get('train_accs', hist.get('train_acc', []))
            val_acc = hist.get('val_accs', hist.get('val_acc', []))
            return train_loss, val_loss, train_acc, val_acc
        elif isinstance(hist, list):
            return hist, [], [], []
        return [], [], [], []

    # -------------------------------------------------------------------
    # 1. Summary Performance Table
    # -------------------------------------------------------------------
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
        .set_caption("Few-Shot Evaluation Performance Summary")

    display(styled_df)

    # -------------------------------------------------------------------
    # 2. ResNet50 + LSTM: Loss & Accuracy per Epoch
    # -------------------------------------------------------------------
    fig1, (ax_r_loss, ax_r_acc) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, k in enumerate(active_k_shots):
        t_loss, v_loss, t_acc, v_acc = unpack_history(resnet_histories.get(k, {}))
        epochs = range(1, len(t_loss) + 1)
        if len(epochs) > 0:
            # Loss
            ax_r_loss.plot(epochs, t_loss, label=f'Train k={k}', color=colors[idx], linestyle='-')
            if len(v_loss) > 0:
                ax_r_loss.plot(epochs, v_loss, linestyle='--', color=colors[idx], alpha=0.7)
            # Accuracy
            if len(t_acc) > 0:
                ax_r_acc.plot(epochs, [a * 100 if a <= 1.0 else a for a in t_acc], label=f'Train k={k}', color=colors[idx], linestyle='-')
            if len(v_acc) > 0:
                ax_r_acc.plot(epochs, [a * 100 if a <= 1.0 else a for a in v_acc], linestyle='--', color=colors[idx], alpha=0.7)

    ax_r_loss.set_title("ResNet50 + LSTM: Loss per Epoch\n(Solid=Train, Dashed=Val)", fontsize=11, fontweight='bold')
    ax_r_loss.set_xlabel("Epoch")
    ax_r_loss.set_ylabel("Cross Entropy Loss")
    ax_r_loss.legend(title="k-shots", fontsize=8)

    ax_r_acc.set_title("ResNet50 + LSTM: Accuracy per Epoch\n(Solid=Train, Dashed=Val)", fontsize=11, fontweight='bold')
    ax_r_acc.set_xlabel("Epoch")
    ax_r_acc.set_ylabel("Accuracy (%)")
    ax_r_acc.legend(title="k-shots", fontsize=8)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------
    # 3. CLIP + MLP Adapter: Loss & Accuracy per Epoch
    # -------------------------------------------------------------------
    fig2, (ax_m_loss, ax_m_acc) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, k in enumerate(active_k_shots):
        t_loss, v_loss, t_acc, v_acc = unpack_history(mlp_histories.get(k, {}))
        epochs = range(1, len(t_loss) + 1)
        if len(epochs) > 0:
            # Loss
            ax_m_loss.plot(epochs, t_loss, label=f'Train k={k}', color=colors[idx], linestyle='-')
            if len(v_loss) > 0:
                ax_m_loss.plot(epochs, v_loss, linestyle='--', color=colors[idx], alpha=0.7)
            # Accuracy
            if len(t_acc) > 0:
                ax_m_acc.plot(epochs, [a * 100 if a <= 1.0 else a for a in t_acc], label=f'Train k={k}', color=colors[idx], linestyle='-')
            if len(v_acc) > 0:
                ax_m_acc.plot(epochs, [a * 100 if a <= 1.0 else a for a in v_acc], linestyle='--', color=colors[idx], alpha=0.7)

    ax_m_loss.set_title("CLIP + MLP Adapter: Loss per Epoch\n(Solid=Train, Dashed=Val)", fontsize=11, fontweight='bold')
    ax_m_loss.set_xlabel("Epoch")
    ax_m_loss.set_ylabel("Cross Entropy Loss")
    ax_m_loss.legend(title="k-shots", fontsize=8)

    ax_m_acc.set_title("CLIP + MLP Adapter: Accuracy per Epoch\n(Solid=Train, Dashed=Val)", fontsize=11, fontweight='bold')
    ax_m_acc.set_xlabel("Epoch")
    ax_m_acc.set_ylabel("Accuracy (%)")
    ax_m_acc.legend(title="k-shots", fontsize=8)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------
    # 4. Final Test Accuracy vs k-shots (Linear & Log2 Scale)
    # -------------------------------------------------------------------
    fig3, (ax_k_lin, ax_k_log) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax_k_lin.plot(df_results['k'], df_results['ResNet50+LSTM Test Acc (%)'], marker='o', linewidth=2, label='ResNet50 + LSTM', color='#1f77b4')
    ax_k_lin.plot(df_results['k'], df_results['CLIP+MLP Adapter Test Acc (%)'], marker='s', linewidth=2, label='CLIP + MLP Adapter', color='#ff7f0e')
    ax_k_lin.set_title("Final Test Accuracy vs. k-shots (Linear Scale)", fontsize=12, fontweight='bold')
    ax_k_lin.set_xlabel("k (clips per class)")
    ax_k_lin.set_ylabel("Test Accuracy (%)")
    ax_k_lin.set_xticks(k_shots_list)
    ax_k_lin.legend()

    # Log2 scale for k > 0
    df_nonzero = df_results[df_results['k'] > 0]
    ax_k_log.plot(df_nonzero['k'], df_nonzero['ResNet50+LSTM Test Acc (%)'], marker='o', linewidth=2, label='ResNet50 + LSTM', color='#1f77b4')
    ax_k_log.plot(df_nonzero['k'], df_nonzero['CLIP+MLP Adapter Test Acc (%)'], marker='s', linewidth=2, label='CLIP + MLP Adapter', color='#ff7f0e')
    ax_k_log.set_xscale('log', base=2)
    ax_k_log.set_title("Final Test Accuracy vs. k-shots (Log2 Scale)", fontsize=12, fontweight='bold')
    ax_k_log.set_xlabel("k (clips per class - log2)")
    ax_k_log.set_ylabel("Test Accuracy (%)")
    ax_k_log.set_xticks(active_k_shots)
    ax_k_log.set_xticklabels([str(k) for k in active_k_shots])
    ax_k_log.legend()

    plt.tight_layout()
    plt.show()