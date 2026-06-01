import torch
import torch.nn as nn


class EmbeddingHead(nn.Module):
    """
    Lightweight head that sits on top of pre-extracted student embeddings.
    Used in Reptile meta-learning and KD training loops.

    Forward:
        s_emb    : (B, student_hidden) — student hidden state
        training : bool — whether to apply dropout

    Returns:
        logits : (B, num_classes)
        proj   : (B, teacher_hidden) — projection for MSE alignment with teacher
    """

    def __init__(
        self,
        student_hidden: int,
        teacher_hidden: int,
        num_classes: int = 10,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.norm       = nn.LayerNorm(student_hidden)
        self.dropout    = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(student_hidden, num_classes)
        self.projector  = nn.Linear(student_hidden, teacher_hidden)

    def forward(self, s_emb: torch.Tensor, training: bool = True):
        normed = self.norm(s_emb)
        x      = self.dropout(normed) if training else normed
        logits = self.classifier(x)
        proj   = self.projector(normed)
        return logits, proj
