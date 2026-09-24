import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.training.train import evaluate_model
from src.cl.rehearsal import distillation_loss


def extract_features_and_labels(model, loader, device):
    """Passes dataset through model to extract intermediate features and class labels."""
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            feats = model.features(inputs)
            all_features.append(feats.detach().cpu())
            all_labels.append(targets.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_features, all_labels


def compute_kmeans_prototypes(features, labels, num_classes, n_clusters=1, seed=42):
    """Computes feature prototypes for each class using KMeans clustering."""
    prototypes = {}
    for c in range(num_classes):
        cls_mask = (labels == c)
        cls_feats = features[cls_mask]
        if len(cls_feats) == 0:
            continue
        k = min(n_clusters, len(cls_feats))
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        kmeans.fit(cls_feats)
        prototypes[c] = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    return prototypes


def prototype_alignment_loss(student_feats, targets, teacher_prototypes, device):
    """Computes alignment loss between student mini-batch features and target class KMeans prototypes."""
    loss = torch.tensor(0.0, device=device)
    valid_count = 0

    for i, target_cls in enumerate(targets.tolist()):
        if target_cls in teacher_prototypes:
            t_proto = teacher_prototypes[target_cls].to(device)  # shape: (k, feature_dim)
            s_feat = student_feats[i].unsqueeze(0)                # shape: (1, feature_dim)

            # MSE distance to nearest class prototype center
            dist = F.mse_loss(s_feat.expand_as(t_proto), t_proto, reduction='none').mean(dim=-1)
            loss = loss + torch.min(dist)
            valid_count += 1

    if valid_count > 0:
        return loss / valid_count
    return torch.tensor(0.0, device=device)


def train_student(student, teacher, train_loader, val_loader, device, mode="kd",
                  epochs=10, lr=1e-3, T=2.0, ce_weight=0.75, distill_weight=0.25,
                  proto_weight=0.5, num_classes=None, n_clusters=1, seed=42,
                  freeze_backbone=True):
    """
    Distil a trained teacher into the student on real clips. Returns (best_student_model, history)
    where best_student_model is the deep-copied model instance that achieved the highest validation accuracy.

        "ce"          cross-entropy only, no teacher
        "kd"          CE + KL on softened logits
        "cosine"      CE + cosine distance between student and teacher features
        "mse"         CE + MSE between a learnable regressor of the student features and the
                      teacher features - the tutorial's RegressorMSE.
        "proto_align" CE + KD + KMeans Prototype Alignment between student features and 
                      teacher class prototypes.

    freeze_backbone: train only the LSTM + head (backbone frozen). Lighter and faster, and
    it isolates the small head so the teacher's contribution (KD vs CE) is clearly visible.
    """
    student.to(device)
    if freeze_backbone:
        for p in student.backbone.parameters():
            p.requires_grad_(False)
    if teacher is not None:
        teacher.to(device).eval()

    regressor = None
    params = [p for p in student.parameters() if p.requires_grad]
    if mode == "mse":
        regressor = nn.Linear(student.fc.in_features, teacher.fc.in_features).to(device)
        params = params + list(regressor.parameters())
    
    optimizer = torch.optim.Adam(params, lr=lr)
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}
    
    # Initialize best accuracy tracking and copy initial model as fallback
    best_acc = 0.0
    best_state = copy.deepcopy(student.state_dict())
    best_model = copy.deepcopy(student)

    # Pre-compute teacher class prototypes if using prototype alignment
    teacher_prototypes = None
    if mode == "proto_align":
        print("[proto_align] Extracting teacher feature prototypes using KMeans...")
        teacher_feats, teacher_labels = extract_features_and_labels(teacher, train_loader, device)
        if num_classes is None:
            num_classes = getattr(student.fc, "out_features", int(np.max(teacher_labels) + 1))
        teacher_prototypes = compute_kmeans_prototypes(
            teacher_feats, teacher_labels, num_classes, n_clusters=n_clusters, seed=seed
        )

    for epoch in range(epochs):
        student.train()
        if freeze_backbone:
            student.backbone.eval()          # keep frozen BatchNorm on its running stats
        run_loss, correct, total = 0.0, 0, 0

        for clips, y in train_loader:
            clips, y = clips.to(device), y.to(device)
            optimizer.zero_grad()
            s_feat = student.features(clips)                       # one student forward per batch
            logits = student.fc(student.dropout(s_feat))
            loss = ce_weight * F.cross_entropy(logits, y)

            if mode != "ce":
                with torch.no_grad():
                    if mode in ("kd", "proto_align"):
                        t_logits = teacher(clips)
                    if mode in ("cosine", "mse"):
                        t_feat = teacher.features(clips)

                if mode == "kd":
                    loss = loss + distill_weight * distillation_loss(logits, t_logits, T)
                elif mode == "cosine":
                    target = torch.ones(clips.size(0), device=device)
                    loss = loss + distill_weight * F.cosine_embedding_loss(s_feat, t_feat, target)
                elif mode == "mse":
                    loss = loss + distill_weight * F.mse_loss(regressor(s_feat), t_feat)
                elif mode == "proto_align":
                    loss_kd = distillation_loss(logits, t_logits, T)
                    loss_proto = prototype_alignment_loss(s_feat, y, teacher_prototypes, device)
                    loss = loss + (distill_weight * loss_kd) + (proto_weight * loss_proto)

            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        train_acc, train_loss = correct / total, run_loss / total
        val_acc, val_loss     = evaluate_model(student, val_loader, device)
        history["train_accs"].append(train_acc)
        history["train_losses"].append(train_loss)
        history["val_accs"].append(val_acc)
        history["val_losses"].append(val_loss)
        print(f"[{mode:>11}] epoch {epoch+1:02d}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Check for best validation accuracy and save model copy
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(student.state_dict())
            best_model = copy.deepcopy(student)

    # Ensure best state dict is loaded onto the returned best model instance
    best_model.load_state_dict(best_state)
    print(f"  best [{mode}] val {best_acc:.4f}")
    
    return best_model, history