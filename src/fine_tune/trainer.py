# Run from notebook: %run /content/src/fine_tune/trainer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataloader.

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
            output = model(data)
            loss   = criterion(output, target)
            pred   = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            total_loss    += loss.item() * data.size(0)
            total_samples += data.size(0)

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss

# Function to train a model for one single epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for clips, labels in tqdm(train_loader, desc="  Training", leave=False):

        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total, running_loss / total

def train_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=10, 
    lr=1e-4, 
    device='cuda',
    criterion=None, 
    optimizer=None,
    save_path=None
):
    # Setup
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_losses': [], 'val_losses': [],
        'train_accs': [], 'val_accs': [],
        'best_val_acc': 0.0
    }

    model.to(device)

    for epoch in range(num_epochs):
        # 1. Train using our helper
        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 2. Validate using your existing evaluate_model
        val_acc, val_loss = evaluate_model(model, val_loader, device)

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
            torch.save(model.state_dict(), save_path)
            print(f"  --> Model saved to {save_path}")

    return history


# INFERENCE
def test_model(model, test_loader, device='cuda'):
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
            
            outputs = model(clips)
            preds   = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return test_acc, all_preds, all_labels


# METRICS  (print)
def print_detailed_metrics(
    all_labels,
    all_preds,
    num_classes=None,
    classes_list=None,
    split_old=None,
    split_new=None,
):
    """
    Print weighted + per-class metrics with old/new class breakdown and
    top/worst 5 classes by F1.

    When split_old / split_new are provided the function also reports
    metrics separately for the "old" (previously seen) and "new" (just
    learned) class ranges — useful for measuring catastrophic forgetting.

    Args:
        all_labels   : list / array of ground-truth labels
        all_preds    : list / array of predicted labels
        num_classes  : total number of classes (default: len(SELECTED_CLASSES))
        classes_list : list of class name strings (default: SELECTED_CLASSES)
        split_old    : (start, end) index tuple for old classes, e.g. (0, 40)
                       if None, the whole range is treated as a single group
        split_new    : (start, end) index tuple for new classes, e.g. (40, 50)
                       if None, skipped
    """
    if classes_list is None:
        classes_list = SELECTED_CLASSES
    if num_classes is None:
        num_classes = len(classes_list)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    # Overall weighted metrics
    overall_acc       = accuracy_score(all_labels, all_preds)
    overall_f1        = f1_score(all_labels,  all_preds, average='weighted', zero_division=0)
    overall_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_recall    = recall_score(all_labels,   all_preds, average='weighted', zero_division=0)

    print(f"\nMETRICS FOR {num_classes} CLASSES")
    print("OVERALL (Weighted Average)")
    print(f"  Accuracy  : {overall_acc:.4f}")
    print(f"  Precision : {overall_precision:.4f}")
    print(f"  Recall    : {overall_recall:.4f}")
    print(f"  F1-Score  : {overall_f1:.4f}")

    # Per-class report (full)
    print("\nPER-CLASS REPORT ")
    print(classification_report(
        all_labels, all_preds,
        labels=np.arange(num_classes),
        target_names=classes_list[:num_classes],
        zero_division=0
    ))

    # Old / New class breakdown (CL-specific, optional)
    def _group_metrics(mask, label_range, group_name):
        if mask.sum() == 0:
            print(f"  (no samples for {group_name})")
            return
        g_labels = all_labels[mask]
        g_preds  = all_preds[mask]
        labs     = np.arange(label_range[0], label_range[1])
        names    = classes_list[label_range[0]:label_range[1]]
        print(f"{group_name} [{label_range[0]}–{label_range[1]-1}]"
              f"  ({label_range[1]-label_range[0]} classes)")
        print(f"  Accuracy  : {accuracy_score(g_labels, g_preds):.4f}")
        print(f"  Precision : {precision_score(g_labels, g_preds, average='weighted', labels=labs, zero_division=0):.4f}")
        print(f"  Recall    : {recall_score(g_labels, g_preds, average='weighted', labels=labs, zero_division=0):.4f}")
        print(f"  F1-Score  : {f1_score(g_labels, g_preds, average='weighted', labels=labs, zero_division=0):.4f}")
        print()
        print(classification_report(g_labels, g_preds,
                                    labels=labs,
                                    target_names=names,
                                    zero_division=0))

    if split_old is not None:
        old_mask = (all_labels >= split_old[0]) & (all_labels < split_old[1])
        _group_metrics(old_mask, split_old, "OLD CLASSES")

    if split_new is not None:
        new_mask = (all_labels >= split_new[0]) & (all_labels < split_new[1])
        _group_metrics(new_mask, split_new, "NEW CLASSES")

    # Top / worst 5 by F1
    f1_per = f1_score(all_labels, all_preds, average=None,
                      labels=np.arange(num_classes), zero_division=0)
    best5  = np.argsort(f1_per)[-5:][::-1]
    worst5 = np.argsort(f1_per)[:5]

    print("TOP 5 BEST CLASSES (F1)")
    for rank, idx in enumerate(best5, 1):
        print(f"  {rank}. {classes_list[idx]:25s}: {f1_per[idx]:.4f}")

    print("TOP 5 WORST CLASSES (F1)")
    for rank, idx in enumerate(worst5, 1):
        print(f"  {rank}. {classes_list[idx]:25s}: {f1_per[idx]:.4f}")


# CONTINUAL LEARNING — MULTI-TASK EVALUATION
def evaluate_all_tasks(
    model,
    device,
    base_loader,
    task_loaders=None,
    combined_loaders=None,
):
    task_loaders     = task_loaders or []
    combined_loaders = combined_loaders or []

    results = {}

    # Base
    acc, loss = evaluate_model(model, base_loader, device)
    results["base"] = {"accuracy": acc, "loss": loss}
    print(f"  base             -> Acc={acc:.4f}  Loss={loss:.4f}")

    # Individual tasks
    for i, loader in enumerate(task_loaders):
        split_name = f"task{i + 1}_only"
        acc, loss  = evaluate_model(model, loader, device)

        results[split_name] = {"accuracy": acc, "loss": loss}
        print(f"  {split_name:<16} -> Acc={acc:.4f}  Loss={loss:.4f}")

    # Combined datasets
    for i, loader in enumerate(combined_loaders):
        split_name = f"combined_t{i + 1}"
        acc, loss  = evaluate_model(model, loader, device)

        results[split_name] = {"accuracy": acc, "loss": loss}
        print(f"  {split_name:<16} -> Acc={acc:.4f}  Loss={loss:.4f}")

    return results
