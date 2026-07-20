from src.cl.rehearsal    import distillation_loss
from src.cl.smart_replay import SmartReplayBuffer, train_with_smart_replay
from src.cl.trainer      import (
    train_cl_arm, train_task, run_cl_stream, save_arm_result, snapshot_teacher,
    eval_tasks_upto, format_row, emb_loader, eval_head_ce,
    train_base_head, load_or_train_base_head,
    fill_buffer, buffer_bytes_per_class, fisher_key, ewc_penalty,
)
from src.cl.ewc          import (
    compute_fisher, check_ewc_penalty, pad_fisher_after_expand, train_ewc, ewc_report,
)

__all__ = [
    "distillation_loss",
    "SmartReplayBuffer", "train_with_smart_replay",
    "train_cl_arm", "train_task", "run_cl_stream", "save_arm_result", "snapshot_teacher",
    "eval_tasks_upto", "format_row", "emb_loader", "eval_head_ce",
    "train_base_head", "load_or_train_base_head",
    "fill_buffer", "buffer_bytes_per_class", "fisher_key", "ewc_penalty",
    "compute_fisher", "check_ewc_penalty", "pad_fisher_after_expand",
    "train_ewc", "ewc_report",
]
