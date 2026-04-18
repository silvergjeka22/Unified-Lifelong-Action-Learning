# Run from notebook: %run /content/src/fine_tune/trainer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from config.config import SELECTED_CLASSES


def fine_tune_model(model, train_loader, val_loader, criterion, optimizer,
                    num_epochs=10, device='cuda', save_flag=False, save_path='best_model.pth'):
    """
    Train and validate a model.

    Returns a dict with train/val losses and accuracies per epoch.

    Args:
        model        : PyTorch model
        train_loader : DataLoader for training
        val_loader   : DataLoader for validation
        criterion    : loss function
        optimizer    : optimizer
        num_epochs   : number of epochs (default 10)
        device       : 'cuda' or 'cpu'
        save_flag    : if True, saves the best model checkpoint
        save_path    : path to save the best model
    """
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss    = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):

        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                loss    = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                preds    = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_loss = running_loss / total
        val_acc  = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if save_flag and val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> Best val loss: {val_loss:.4f}")

    if save_flag and best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"Best model saved: {save_path}")

    return {
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'train_accs':   train_accs,
        'val_accs':     val_accs,
        'best_val_loss': best_val_loss,
    }


def test_model_with_predictions(model, test_loader, device='cuda'):
    """
    Run inference on the test set.

    Returns (test_accuracy, all_preds, all_labels).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc, all_preds, all_labels


def print_detailed_metrics(all_labels, all_preds,
                           num_classes=None, classes_list=None):
    """
    Print weighted metrics and per-class report with top/worst 5 classes.

    Args:
        all_labels   : list of ground-truth labels
        all_preds    : list of predicted labels
        num_classes  : number of classes (default: len(SELECTED_CLASSES))
        classes_list : list of class names (default: SELECTED_CLASSES)
    """
    if classes_list is None:
        classes_list = SELECTED_CLASSES
    if num_classes is None:
        num_classes = len(classes_list)

    overall_acc       = accuracy_score(all_labels, all_preds)
    overall_f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_recall    = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("\nMETRICS (Weighted Average)")
    print(f"  Accuracy  : {overall_acc:.4f}")
    print(f"  Precision : {overall_precision:.4f}")
    print(f"  Recall    : {overall_recall:.4f}")
    print(f"  F1-Score  : {overall_f1:.4f}")

    print("\nPER-CLASS METRICS")
    print(classification_report(
        all_labels, all_preds,
        labels=np.arange(num_classes),
        target_names=classes_list[:num_classes],
        zero_division=0
    ))

    f1_per_class  = f1_score(all_labels, all_preds, average=None,
                             labels=np.arange(num_classes), zero_division=0)
    best_classes  = np.argsort(f1_per_class)[-5:][::-1]
    worst_classes = np.argsort(f1_per_class)[:5]

    print("\nTOP 5 BEST CLASSES (F1-Score):")
    for idx in best_classes:
        print(f"  {classes_list[idx]:25s}: {f1_per_class[idx]:.4f}")

    print("\nTOP 5 WORST CLASSES (F1-Score):")
    for idx in worst_classes:
        print(f"  {classes_list[idx]:25s}: {f1_per_class[idx]:.4f}")
