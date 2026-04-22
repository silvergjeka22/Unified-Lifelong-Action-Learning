import torch.nn.functional as F
import random

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
    replay_buffer,
    optimizer,
    device,
    num_old_classes=10,
    lambda_distill=1.0,
    epochs=5
):
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
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

            # -----------------------------
            # Forward
            # -----------------------------
            logits = model(x)

            # -----------------------------
            # CE LOSS (all classes)
            # -----------------------------
            loss_ce = ce_loss(logits, y)

            # DISTILLATION LOSS
            with torch.no_grad():
                teacher_logits = teacher(x)

            student_old = logits[:, :num_old_classes]
            teacher_old = teacher_logits[:, :num_old_classes]

            loss_kd = distillation_loss(student_old, teacher_old)

            # -----------------------------
            # TOTAL LOSS
            # -----------------------------
            loss = loss_ce + lambda_distill * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # replay_buffer.add_batch(x_new.detach(), y_new.detach())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")