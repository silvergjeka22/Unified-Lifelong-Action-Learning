import torch
import torch.nn as nn


class TemporalHead(nn.Module):
    """
    LSTM + classifier over cached frame features from a frozen CNN backbone.

    Input : (B, T, backbone_dim) — per-frame features, e.g. (B, 16, 2048)
    Output: (logits, proj)       — the contract finetune_head, train_reptile,
                                   train_student and eval_head expect.

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


def make_temporal_head(num_classes: int, freeze_lstm: bool = False, lstm_state: dict = None,
                       device=None, backbone_dim: int = None, hidden_size: int = None,
                       teacher_hidden: int = None, dropout_p: float = None):
    """
    Build a TemporalHead using project config defaults, seeded with the Task-0 LSTM
    weights when given.

    The LSTM is trainable by default. A frozen LSTM (72,986 params) cannot reshape the
    feature space, so it cannot separate classes that overlap in the frozen 256-d
    representation, and it underfits the current task: on this data it reached only
    81.8% on the task it had just trained on, versus 98.2% with the LSTM trainable.
    """
    import src.config.config as cfg

    head = TemporalHead(
        backbone_dim   = backbone_dim   if backbone_dim   is not None else cfg.BACKBONE_DIM,
        hidden_size    = hidden_size    if hidden_size    is not None else cfg.TEACHER_HIDDEN,
        teacher_hidden = teacher_hidden if teacher_hidden is not None else cfg.TEACHER_HIDDEN,
        num_classes    = num_classes,
        dropout_p      = dropout_p      if dropout_p      is not None else cfg.HEAD_DROPOUT,
        freeze_lstm    = freeze_lstm,
    )
    if lstm_state is not None:
        head.load_lstm_state(lstm_state)
    return head.to(device) if device is not None else head


def expand_head(head, new_num_classes: int):
    """
    Grow head.classifier to cover new classes, preserving old-class rows.

    Mutates in place (rather than returning a fresh module) so that any ModelWrapper
    already wrapping this head picks up the new layer automatically — EWC's Fisher
    bookkeeping depends on that.

    TemporalHead exposes the growable layer as `.classifier`.
    """
    old = head.classifier
    if new_num_classes <= old.out_features:
        return head
    new = nn.Linear(old.in_features, new_num_classes)
    with torch.no_grad():
        new.weight[:old.out_features] = old.weight
        new.bias[:old.out_features]   = old.bias
    head.classifier = new.to(next(head.parameters()).device)
    return head


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
