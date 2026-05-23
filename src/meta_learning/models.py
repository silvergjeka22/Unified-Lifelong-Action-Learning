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


def load_model_checkpoint(model, checkpoint_path, device):
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


def build_student(num_classes, cfg, device, lstm_hidden, num_old_classes):
    model = fresh_model(
        num_classes=num_classes,
        cfg=cfg,
        device=device,
        lstm_hidden=lstm_hidden
    )
    model = load_model_checkpoint(model, cfg.RESNET50_PATH, device)
    model = load_old_head_weights(model, cfg.RESNET50_PATH, device, num_old_classes)
    return model


def make_frozen_teacher(num_classes, cfg, device, lstm_hidden, checkpoint_path):
    teacher = fresh_model(
        num_classes=num_classes,
        cfg=cfg,
        device=device,
        lstm_hidden=lstm_hidden
    )
    teacher = load_model_checkpoint(teacher, checkpoint_path, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher

