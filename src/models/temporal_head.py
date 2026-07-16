import torch
import torch.nn as nn


class TemporalHead(nn.Module):
    """
    LSTM + classifier over cached frame features from a frozen CNN backbone.

    Input : (B, T, backbone_dim) — per-frame features, e.g. (B, 16, 2048)
    Output: (logits, proj)       — same contract as EmbeddingHead, so it drops
                                   straight into finetune_head, train_reptile,
                                   train_with_smart_replay and eval_head.

    The freeze boundary is set by `freeze_lstm`:

        freeze_lstm=False  ResNet [FROZEN] -> LSTM [TRAIN] -> head [TRAIN]   (2.4M trainable)
        freeze_lstm=True   ResNet [FROZEN] -> LSTM [FROZEN] -> head [TRAIN]  (70K trainable)

    Both modes should be initialised from the Task-0 teacher LSTM via
    load_lstm_state() so the only variable between them is whether it can move.
    """

    def __init__(
        self,
        backbone_dim: int = 2048,
        hidden_size: int = 256,
        teacher_hidden: int = 256,
        num_classes: int = 10,
        dropout_p: float = 0.25,
        freeze_lstm: bool = False,
    ):
        super().__init__()
        self.lstm       = nn.LSTM(backbone_dim, hidden_size, batch_first=True)
        self.norm       = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.projector  = nn.Linear(hidden_size, teacher_hidden)
        self.set_lstm_frozen(freeze_lstm)

    def set_lstm_frozen(self, frozen: bool):
        for p in self.lstm.parameters():
            p.requires_grad_(not frozen)
        self.lstm_frozen = frozen
        return self

    def load_lstm_state(self, lstm_state: dict):
        """Initialise from the Task-0 teacher LSTM (keeps both probes comparable)."""
        self.lstm.load_state_dict(lstm_state)
        return self

    def forward(self, x: torch.Tensor, training: bool = True):
        out, _ = self.lstm(x)            # (B, T, hidden)
        h_last = out[:, -1, :]           # (B, hidden)
        normed = self.norm(h_last)
        z      = self.dropout(normed) if training else normed
        return self.classifier(z), self.projector(normed)

    @torch.no_grad()
    def embed(self, x: torch.Tensor):
        """h_last only — used to build replay prototypes and for t-SNE."""
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def weight_align(head, n_old: int):
    """
    Weight Aligning (Zhao et al., CVPR 2020) — rescale new-class classifier rows so
    their mean norm matches the old classes'.

    Corrects task-recency bias with ZERO stored data. This is the baseline that stops
    "replay beats naive" from being a strawman: if replay only matches this, then replay
    is just fixing logit imbalance and you should know that.
    """
    with torch.no_grad():
        W = head.classifier.weight.data
        if n_old <= 0 or n_old >= W.shape[0]:
            return head
        norm_old = W[:n_old].norm(dim=1).mean()
        norm_new = W[n_old:].norm(dim=1).mean()
        if norm_new > 0:
            W[n_old:] *= (norm_old / norm_new)
    return head
