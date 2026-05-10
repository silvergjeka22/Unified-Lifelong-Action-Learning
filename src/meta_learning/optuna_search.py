import copy
import torch
import optuna

from meta_learning.models   import fresh_model, evaluate
from meta_learning.buffers  import ReplayBuffer, EpisodeBuffer
from meta_learning.training import train_reptile_full, train_rehearsal


# utilities 
def _make_study():
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=1, interval_steps=1
        ),
    )

def _print_search_summary(study, tag):
    print(f"\n  [{tag}] Best params  : {study.best_params}")
    print(f"  [{tag}] Best val acc : {study.best_value:.4f}")


# Reptile hyper-parameter search
def optuna_search_reptile(
    task_tag,
    val_loader,
    new_loaders_for_buffer,     # loaders
    new_class_ids,              # set of new class
    search_new_weight,          # search the new-class bias in the loss
    cfg, device, lstm_hidden,
    exemplar_train_loader,      # old-task
    num_old_classes,
    kd_flag=False,              # knowledge-distillation loss
    teacher=None,               # frozen teacher model
    new_class_bias=3,           # bias toward new classes
    n_trials=20,
    search_episodes=100,        # meta-episodes
    search_epochs=2,
    rep_n_way=5,
    rep_k_support=3,
    rep_k_query=5,
    rep_lambda_kd=0.3,
):
    print(
        f"\nOPTUNA REPTILE — {task_tag} "
        f"({'KD' if kd_flag else 'no-KD'}) | "
        f"{n_trials} trials | episodes={search_episodes} × epochs={search_epochs}"
    )

    # build episode buffer
    shared_buf = EpisodeBuffer()
    shared_buf.add_from_loader(exemplar_train_loader, mark_new=False)
    for ldr in new_loaders_for_buffer:
        shared_buf.add_from_loader(ldr, mark_new=True)

    # the number of available classes in the buffer
    n_way_eff = min(rep_n_way, len(shared_buf.available_classes()))

    def objective(trial):
        # hyper-parameters
        inner_lr   = trial.suggest_float("inner_lr",   0.001, 0.02, log=True)
        epsilon    = trial.suggest_float("epsilon",    0.1,   0.5,  step=0.05)
        new_weight = (
            trial.suggest_float("new_weight", 1.0, 6.0, step=0.5)
            if search_new_weight else 1.0
        )

        num_classes = num_old_classes + len(new_class_ids or [])
        m   = fresh_model(num_classes=num_classes, cfg=cfg,
                          device=device, lstm_hidden=lstm_hidden)
        buf = copy.deepcopy(shared_buf)   # isolate each trial's buffer

        _, val_acc = train_reptile_full(
            model=m,
            episode_buffer=buf,
            val_loader=val_loader,
            device=device,
            teacher=teacher if kd_flag else None,
            num_old_classes=num_old_classes,
            n_way=n_way_eff,
            k_support=rep_k_support,
            k_query=rep_k_query,
            inner_lr=inner_lr,
            inner_steps=5,
            epsilon=epsilon,
            episodes=search_episodes,
            epochs=search_epochs,
            lambda_kd=rep_lambda_kd,
            new_class_bias=new_class_bias,
            new_class_ids=new_class_ids,
            new_weight=new_weight,
            kd=kd_flag,
            tag=f"search-{task_tag}-t{trial.number}",
            trial=trial,
        )

        # free GPU memory
        del m, buf
        torch.cuda.empty_cache()
        return val_acc

    study = _make_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    _print_search_summary(study, task_tag)
    return study.best_params


# Rehearsal hyper-parameter search
def optuna_search_rehearsal(
    task_tag,
    train_loader,               # new-task training data
    val_loader,                 # combined validation
    exemplar_train_loader,      # T0 exemplars
    extra_replay_loader,        # replay data 
    teacher,                    # frozen teacher for KD
    cfg, device, lstm_hidden,
    num_classes,                # total output classes
    num_old_classes,            # number of old classes
    limit=3,                    # max exemplars
    kd=True,                    # knowledge-distillation loss
    n_trials=20,
    search_epochs=3,
    new_repeat=10,              # times new task is repeated per epoch
):
    print(f"\nOPTUNA REHEARSAL — {task_tag} | {n_trials} trials × {search_epochs} epochs")

    def objective(trial):
        # hyper-parameters being searched
        lambda_distill = trial.suggest_float("lambda_distill", 0.05, 1.0,  step=0.05)
        lstm_lr        = trial.suggest_float("lstm_lr",        1e-5, 1e-3, log=True)
        fc_lr          = trial.suggest_float("fc_lr",          1e-4, 1e-2, log=True)

        m = fresh_model(num_classes=num_classes, cfg=cfg,
                        device=device, lstm_hidden=lstm_hidden)
        opt = torch.optim.Adam(
            [
                {"params": m.lstm.parameters(), "lr": lstm_lr},
                {"params": m.fc.parameters(),   "lr": fc_lr},
            ],
            weight_decay=1e-4,
        )

        # build replay buffer
        buf = ReplayBuffer(max_size=1000)
        buf.add_from_loader(exemplar_train_loader, max_per_class=limit)
        if extra_replay_loader is not None:
            buf.add_from_loader(extra_replay_loader, max_per_class=limit)

        train_rehearsal(
            m, teacher, train_loader, val_loader,
            buf, opt, device,
            num_old_classes=num_old_classes,
            lambda_distill=lambda_distill,
            epochs=search_epochs,
            new_repeat=new_repeat,
            kd=kd,
        )

        _, val_acc = evaluate(m, val_loader, device)

        # free GPU
        del m, opt, buf
        torch.cuda.empty_cache()

        # intermediate values so the pruner can cut bad trials early
        for ep in range(search_epochs):
            trial.report(val_acc, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_acc

    study = _make_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    _print_search_summary(study, task_tag)
    return study.best_params