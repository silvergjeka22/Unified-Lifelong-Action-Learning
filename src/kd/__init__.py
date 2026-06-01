from src.kd.trainer import finetune_head
from src.kd.utils   import (
    extract_embeddings, eval_head, verify_embeddings,
    restore_head_to_student, calculate_accuracies,
    compare_head_vs_student_on_same_batch,
)

__all__ = [
    "finetune_head",
    "extract_embeddings", "eval_head", "verify_embeddings",
    "restore_head_to_student", "calculate_accuracies",
    "compare_head_vs_student_on_same_batch",
]
