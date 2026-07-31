from src.kd.trainer import finetune_head, train_student
from src.kd.utils   import restore_head_to_student, eval_head

__all__ = ["finetune_head", "train_student", "restore_head_to_student", "eval_head"]
