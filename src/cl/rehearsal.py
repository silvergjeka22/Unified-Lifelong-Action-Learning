"""
Distillation loss for LwF (Learning without Forgetting) and KD.

The old ReplayBuffer (FIFO) and train_continual() lived here; both are superseded:
    ReplayBuffer     -> src/cl/smart_replay.py  SmartReplayBuffer (prototype-aware)
    train_continual  -> src/cl/trainer.py       train_cl_arm (one loop for every arm)
"""

import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, T=5.0):
    """
    KL(student || teacher) on temperature-softened logits.

    Multiplied by T^2 so the gradient magnitude stays comparable to the CE term as T
    changes (Hinton et al. 2015) — otherwise raising T silently shrinks the KD signal.

    Used by:
        LwF   — teacher is the model's own pre-task snapshot, on old-class logits only
        KD    — teacher is the frozen ResNet50+LSTM (Section 8)
    """
    student = F.log_softmax(student_logits / T, dim=1)
    teacher = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean") * (T * T)
