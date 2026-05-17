import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



class EmbeddingLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256,
                 num_layers=1, num_classes=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.drop(h_n[-1]))

    def get_embedding(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.drop(h_n[-1])



def fresh_model(num_classes, cfg, device, lstm_hidden):
    model = EmbeddingLSTM(
        hidden_size=lstm_hidden,
        num_classes=num_classes
    ).to(device)
    return model


def load_compatible_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    compatible = {
        k: v for k, v in ckpt.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    model_dict.update(compatible)
    model.load_state_dict(model_dict, strict=False)
    return model


def load_old_head_weights(model, checkpoint_path, device, num_old_classes):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    if "fc.weight" in ckpt and "fc.bias" in ckpt:
        with torch.no_grad():
            rows = min(num_old_classes, ckpt["fc.weight"].size(0), model_dict["fc.weight"].size(0))
            model_dict["fc.weight"][:rows] = ckpt["fc.weight"][:rows]
            model_dict["fc.bias"][:rows] = ckpt["fc.bias"][:rows]

    model.load_state_dict(model_dict, strict=False)
    return model


def build_student_from_task0(num_classes, cfg, device, lstm_hidden, num_old_classes):
    model = fresh_model(
        num_classes=num_classes,
        cfg=cfg,
        device=device,
        lstm_hidden=lstm_hidden
    )
    model = load_compatible_checkpoint(model, cfg.RESNET50_PATH, device)
    model = load_old_head_weights(model, cfg.RESNET50_PATH, device, num_old_classes)
    return model


def make_frozen_teacher_from_checkpoint(num_classes, cfg, device, lstm_hidden, checkpoint_path):
    teacher = fresh_model(
        num_classes=num_classes,
        cfg=cfg,
        device=device,
        lstm_hidden=lstm_hidden
    )
    teacher = load_compatible_checkpoint(teacher, checkpoint_path, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def snapshot_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_rehearsal_optimizer(model, lr=1e-4):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


def make_reptile_outer_optimizer(model, lstm_lr=1e-4, fc_lr=1e-3):
    return torch.optim.Adam([
        {"params": model.lstm.parameters(), "lr": lstm_lr},
        {"params": model.fc.parameters(), "lr": fc_lr},
    ], weight_decay=1e-4)


def distillation_loss(student_logits, teacher_logits, T=5.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += ce(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / max(len(loader), 1), (correct / total if total else 0.0)


def make_student_t1():
    return build_student_from_task0(
        num_classes    = NUM_CLASSES_T1,
        cfg            = cfg,
        device         = device,
        lstm_hidden    = LSTM_HIDDEN,
        num_old_classes= N0,
    )

def make_student_t2(trained_t1_model):
    t2 = fresh_model(
        num_classes = NUM_CLASSES_T2,
        cfg         = cfg,
        device      = device,
        lstm_hidden = LSTM_HIDDEN,
    )
    sd_t1 = trained_t1_model.state_dict()
    sd_t2 = t2.state_dict()
    for k in sd_t2:
        if k.startswith("lstm"):
            sd_t2[k] = sd_t1[k].clone()
    with torch.no_grad():
        rows = NUM_CLASSES_T1
        sd_t2["fc.weight"][:rows] = sd_t1["fc.weight"][:rows].clone()
        sd_t2["fc.bias"][:rows]   = sd_t1["fc.bias"][:rows].clone()
    t2.load_state_dict(sd_t2)
    return t2.to(device)

def make_reh_opt(m):
    return make_rehearsal_optimizer(m, lr=REH_LR)