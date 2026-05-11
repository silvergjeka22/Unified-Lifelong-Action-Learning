import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



class EmbeddingLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256,
                 num_layers=1, num_classes=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.drop(h_n[-1]))


    def get_embedding(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.drop(h_n[-1])          # [B, hidden_size] — before FC



def fresh_model(num_classes, cfg, device, lstm_hidden):
    m    = EmbeddingLSTM(hidden_size=lstm_hidden, num_classes=num_classes).to(device)
    ckpt = torch.load(cfg.RESNET50_PATH, map_location=device)
    lstm_only = {k: v for k, v in ckpt.items() if k.startswith("lstm.")}
    m.load_state_dict(lstm_only, strict=False)
    return m



def make_teacher(source_model):
    t = copy.deepcopy(source_model)
    t.eval()
    for p in t.parameters():
        p.requires_grad = False
    return t


def make_optimizer(model):
    return torch.optim.Adam([
        {"params": model.lstm.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(),   "lr": 1e-3},
    ], weight_decay=1e-4)



def distillation_loss(student_logits, teacher_logits, T=5.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)



def weighted_ce(logits, labels, new_class_ids, new_weight=1.0, device="cpu"):
    weights = torch.ones(len(labels), device=device)
    for i, lbl in enumerate(labels):
        if lbl.item() in new_class_ids:
            weights[i] = new_weight
    return (weights * F.cross_entropy(logits, labels, reduction='none')).mean()



@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y      = x.to(device), y.to(device)
        logits    = model(x)
        loss_sum += ce(logits, y).item()
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / max(len(loader), 1), correct / total if total else 0.0