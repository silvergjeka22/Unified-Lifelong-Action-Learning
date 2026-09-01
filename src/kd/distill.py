import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.train import evaluate_model
from src.cl.rehearsal import distillation_loss


def train_student(student, teacher, train_loader, val_loader, device, mode="kd",
                  epochs=10, lr=1e-3, T=2.0, ce_weight=0.75, distill_weight=0.25):
    """
    Distil a trained teacher into the student on real clips. Returns the best-val student.

        "ce"      cross-entropy only, no teacher
        "kd"      CE + KL on softened logits
        "cosine"  CE + cosine distance between student and teacher features
        "mse"     CE + MSE between student and teacher features
    """
    student.to(device)
    if teacher is not None:
        teacher.to(device).eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    best_acc, best_state = 0.0, copy.deepcopy(student.state_dict())

    for epoch in range(epochs):
        student.train()
        for clips, y in train_loader:
            clips, y = clips.to(device), y.to(device)
            optimizer.zero_grad()
            logits = student(clips)
            loss = ce_weight * F.cross_entropy(logits, y)

            if mode != "ce":
                with torch.no_grad():
                    if mode == "kd":
                        t_out = teacher(clips)
                    else:
                        t_out = teacher.features(clips)
                if mode == "kd":
                    loss = loss + distill_weight * distillation_loss(logits, t_out, T)
                elif mode == "cosine":
                    target = torch.ones(clips.size(0), device=device)
                    loss = loss + distill_weight * F.cosine_embedding_loss(student.features(clips), t_out, target)
                elif mode == "mse":
                    loss = loss + distill_weight * F.mse_loss(student.features(clips), t_out)

            loss.backward()
            optimizer.step()

        val_acc, _ = evaluate_model(student, val_loader, device)
        print(f"[{mode:>6}] epoch {epoch+1:02d}/{epochs} | val {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    print(f"  best [{mode}] val {best_acc:.4f}")
    return student
