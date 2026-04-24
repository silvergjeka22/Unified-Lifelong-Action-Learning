import torch.nn.functional as F
import random
import torch

def distillation_loss(student_logits, teacher_logits, T=5.0):
    student = F.log_softmax(student_logits / T, dim=1)
    teacher = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(student, teacher, reduction='batchmean') * (T * T)

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.data = []

    def add_batch(self, x, y):
        for i in range(len(x)):
            self.data.append((x[i].cpu(), y[i].cpu()))

        # keep buffer size fixed
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]

    def sample(self, batch_size):
        if len(self.data) == 0:
            return None, None

        samples = random.sample(self.data, min(batch_size, len(self.data)))
        x, y = zip(*samples)
        return torch.stack(x), torch.tensor(y)
    
def train_continual(
    model,
    teacher,
    train_loader,
    val_loader,   
    replay_buffer,
    optimizer,
    device,
    num_old_classes=10,
    lambda_distill=1.0,
    epochs=5,
    kd=True
):
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # -----------------------------
        # TRAINING
        # -----------------------------
        model.train()
        total_loss = 0

        for x_new, y_new in train_loader:
            x_new = x_new.to(device)
            y_new = y_new.to(device)

            # Sample from buffer
            x_old, y_old = replay_buffer.sample(len(x_new))

            if x_old is not None:
                x_old = x_old.to(device)
                y_old = y_old.to(device)

                x = torch.cat([x_new, x_old], dim=0)
                y = torch.cat([y_new, y_old], dim=0)
            else:
                x, y = x_new, y_new

            # Forward
            logits = model(x)

            # CE LOSS
            loss_ce = ce_loss(logits, y)

            # DISTILLATION
            if kd:
                with torch.no_grad():
                    teacher_logits = teacher(x)

                student_old = logits[:, :num_old_classes]
                teacher_old = teacher_logits[:, :num_old_classes]

                loss_kd = distillation_loss(student_old, teacher_old)
                loss = loss_ce + lambda_distill * loss_kd
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                logits = model(x_val)
                loss = ce_loss(logits, y_val)

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)

        val_accuracy = correct / total if total > 0 else 0

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {total_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )