
# Run from notebook: %run /content/src/fine_tune/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from config.config import SELECTED_CLASSES

from torchvision.models import mobilenet_v2
from torchvision import models


class StudentModel_KD_cosine(nn.Module):
    def __init__(self, num_classes=10, hidden_size=256, dropout_p=0.5):
        super().__init__()

        # CNN backbone (frame encoder)
        self.backbone = mobilenet_v2(weights='DEFAULT').features

        self.align = nn.Conv2d(1280, 2048, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # temporal model
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, num_classes)


    def encode_frame(self, frame):
        x = self.backbone(frame)
        x = self.align(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape

        feats = []

        for t in range(T):
            frame = x[:, t]          # [B,3,224,224]
            f = self.encode_frame(frame)
            feats.append(f)

        feats = torch.stack(feats, dim=1)  # [B,T,2048]

        out, _ = self.lstm(feats)

        x = out[:, -1, :]
        logits = self.fc(self.dropout(x))

        return logits, x



class TeacherModel_KD_cosine(nn.Module):
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
        out, _  = self.lstm(features)                   # [B, T, hidden]
        feat = out[:, -1, :]          # (B, hidden)
        logits = self.fc(self.dropout(feat))     # last hidden state only
        return logits
    
    def hidden_representation(self, x):
        with torch.no_grad():
            B, T, C, H, W = x.shape
            features = self.resnet(x.view(B * T, C, H, W))  # [B*T, 2048, 1, 1]
            features = features.view(B, T, -1)               # [B, T, 2048]
            out, _  = self.lstm(features)                   # [B, T, hidden]
            feat = out[:, -1, :]          # (B, hidden)
        return feat




def evaluate_model_kd_cosine(model, dataloader, device):
    """
    Evaluate model on a dataloader.
    For KD with cosine embedding.

    Args:
        model        : PyTorch model
        dataloader   : DataLoader
        device       : 'cuda' or 'cpu'

    Returns:
        (accuracy, avg_loss)
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss    = 0.0
    total_samples = 0
    criterion     = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss   = criterion(output, target)
            pred   = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            total_loss    += loss.item() * data.size(0)
            total_samples += data.size(0)

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


def train_one_epoch_kd_cosine(student_model, teacher_model, train_loader, criterion, optimizer, hidden_rep_loss_weight, ce_loss_weight, device):
    """
    Handles the training loop for a single epoch.
    For KD with cosine embedding.
    """
    student_model.train()
    teacher_model.eval()  # Teacher in eval mode, no gradients

    for param in teacher_model.parameters():
        param.requires_grad = False  # Ensure teacher is frozen

    running_loss, correct, total = 0.0, 0, 0

    cosine_loss = nn.CosineEmbeddingLoss()

    for clips, labels in tqdm(train_loader, desc="  Training", leave=False):

        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16): 
            with torch.no_grad():
                teacher_hidden_rep = teacher_model.hidden_representation(clips)

            outputs_student, student_hidden_rep = student_model(clips)

            hidden_rep_loss = cosine_loss(student_hidden_rep, teacher_hidden_rep, target=torch.ones(clips.size(0)).to(device))

            output_loss = criterion(outputs_student, labels)
            loss = hidden_rep_loss * hidden_rep_loss_weight + output_loss * ce_loss_weight
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs_student.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total, running_loss / total




def train_model_kd_cosine(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=5e-5,
    device='cuda',
    criterion=None, 
    optimizer=None,
    save_path=None,
    hidden_rep_loss_weight=0.25,
    ce_loss_weight=0.75,
):
    # Setup
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(student_model.parameters(), lr=lr)

    history = {
        'train_losses': [], 'val_losses': [],
        'train_accs': [], 'val_accs': [],
        'best_val_acc': 0.0
    }

    student_model.to(device)
    teacher_model.to(device)

    for epoch in range(num_epochs):
        # 1. Train using our helper
        train_acc, train_loss = train_one_epoch_kd_cosine(student_model, teacher_model, train_loader, criterion, optimizer, hidden_rep_loss_weight, ce_loss_weight, device)
        torch.cuda.empty_cache()  # Clear cache after each epoch to manage VRAM
        
        # 2. Validate using your existing evaluate_model
        val_acc, val_loss = evaluate_model_kd_cosine(student_model, val_loader, device)
        torch.cuda.empty_cache()  # Clear cache after validation

        # 3. Update History
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}")

        # 4. Save best model
        if save_path and val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            torch.save(student_model.state_dict(), save_path)
            print(f"  --> Model saved to {save_path}")

    return history


def test_model_kd_cosine(model, test_loader, device='cuda'):
    """
    Run inference on the test set without task offsets.

    Args:
        model        : PyTorch model
        test_loader  : DataLoader for the test set
        device       : 'cuda' or 'cpu'

    Returns:
        (test_accuracy, all_preds, all_labels)
    """
    model.eval()
    all_preds, all_labels = [], []

    print(f"Running Inference on Test Set...")
    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc="  Testing", leave=False):
            clips, labels = clips.to(device), labels.to(device)
            
            outputs, _ = model(clips)
            preds   = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return test_acc, all_preds, all_labels
