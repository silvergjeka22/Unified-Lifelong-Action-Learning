import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.train import evaluate_model
from src.cl.rehearsal import distillation_loss


def train_student(student, teacher, train_loader, val_loader, device, mode="kd",
                  epochs=10, lr=1e-3, T=2.0, ce_weight=0.75, distill_weight=0.25,
                  freeze_backbone=False):
    """
    Distil a trained teacher into the student on real clips. Returns (student, history)
    with the best-val weights loaded.

        "ce"      cross-entropy only, no teacher
        "kd"      CE + KL on softened logits
        "cosine"  CE + cosine distance between student and teacher features
        "mse"     CE + MSE between a learnable regressor of the student features and the
                  teacher features - the tutorial's RegressorMSE. The regressor translates
                  the student space into the teacher's; it is training-only and dropped at
                  inference (the saved student is unchanged).

    These four are the methods compared in the PyTorch knowledge-distillation tutorial:
    cross-entropy, soft-target KD, cosine hidden-representation loss, and regressor MSE.

    freeze_backbone: train only the LSTM + head (backbone frozen). Lighter and faster, and
    it isolates the small head so the teacher's contribution (KD vs CE) is clearly visible -
    the "distillation power" setup. Applied identically to every mode, so it stays fair.
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
    best_acc, best_state = 0.0, copy.deepcopy(student.state_dict())

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
                    if mode == "kd":
                        t_logits = teacher(clips)
                    else:
                        t_feat = teacher.features(clips)
                if mode == "kd":
                    loss = loss + distill_weight * distillation_loss(logits, t_logits, T)
                elif mode == "cosine":
                    target = torch.ones(clips.size(0), device=device)
                    loss = loss + distill_weight * F.cosine_embedding_loss(s_feat, t_feat, target)
                elif mode == "mse":
                    loss = loss + distill_weight * F.mse_loss(regressor(s_feat), t_feat)

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
        print(f"[{mode:>6}] epoch {epoch+1:02d}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    print(f"  best [{mode}] val {best_acc:.4f}")
    return student, history
