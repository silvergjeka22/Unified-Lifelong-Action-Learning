import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

class ScratchCNNLSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=256):
        super(ScratchCNNLSTM, self).__init__()

        # ---- CNN backbone (from your original ScratchModel) ----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Assuming input frames are 224x224 → downsampled by 2x2 pooling three times
        # 224 → 112 → 56 → 28 → final feature map size: [128, 28, 28]
        self.feature_size = 128 * 28 * 28

        # ---- LSTM for temporal modeling ----
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # ---- Final classification layer ----
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.size()

        # ---- Step 1: Merge batch and time for CNN ----
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]

        # ---- Step 2: CNN feature extraction ----
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten per-frame features
        x = x.view(B, T, -1)  # [B, T, feature_size]

        # ---- Step 3: LSTM over temporal dimension ----
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_dim]

        # Use only the last frame's output for classification
        final_features = lstm_out[:, -1, :]  # [B, hidden_dim]

        # ---- Step 4: Final classification ----
        logits = self.fc(final_features)  # [B, num_classes]
        return logits