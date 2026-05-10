import copy
import torch
import optuna

from meta_learning.models   import fresh_model, evaluate
from meta_learning.buffers  import ReplayBuffer, EpisodeBuffer
from meta_learning.training import (
    train_reptile_full,
    train_rehearsal,
)
def _make_study():
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
    )


def _print_search_summary(study, tag):
    print(f"\n  [{tag}] Best params : {study.best_params}")
    print(f"  [{tag}] Best val acc : {study.best_value:.4f}")
    n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"  [{tag}] Trials — complete: {n_complete} | pruned: {n_pruned}")


def optuna_search_reptile(task_tag, val_loader, new_loaders_for_buffer,
                          new_class_ids, search_new_weight,
                          cfg, device, lstm_hidden,
                          exemplar_train_loader, num_old_classes,
                          kd_flag=False, teacher=None,
                          new_class_bias=3, n_trials=20,
                          search_episodes=100, search_epochs=2,
                          rep_n_way=5, rep_k_support=3, rep_k_query=5,
                          rep_lambda_kd=0.3):
    print(f"\n{'='*60}\nOPTUNA REPTILE — {task_tag}  "
          f"({'KD' if kd_flag else 'no-KD'}) | "
          f"{n_trials} trials | proxy={search_episodes}ep×{search_epochs}epochs")
    print(f"  Pruner: MedianPruner | Sampler: TPE (Bayesian)\n{'='*60}")

    shared_buf = EpisodeBuffer()
    shared_buf.add_from_loader(exemplar_train_loader, mark_new=False)
    for ldr in new_loaders_for_buffer:
        shared_buf.add_from_loader(ldr, mark_new=True)
    n_way_eff = min(rep_n_way, len(shared_buf.available_classes()))

    def objective(trial):
        lr  = trial.suggest_float("inner_lr",   0.001, 0.02, log=True)
        eps = trial.suggest_float("epsilon",     0.1,   0.5,  step=0.05)
        w   = (trial.suggest_float("new_weight", 1.0,   6.0,  step=0.5)
               if search_new_weight else 1.0)
        m   = fresh_model(num_classes=num_old_classes + len(new_class_ids or []),
                          cfg=cfg, device=device, lstm_hidden=lstm_hidden)
        buf = copy.deepcopy(shared_buf)
        try:
            _, val_acc = train_reptile_full(
                model=m, episode_buffer=buf,
                val_loader=val_loader, device=device,
                teacher=teacher if kd_flag else None,
                num_old_classes=num_old_classes, n_way=n_way_eff,
                k_support=rep_k_support, k_query=rep_k_query,
                inner_lr=lr, inner_steps=5, epsilon=eps,
                episodes=search_episodes, epochs=search_epochs,
                lambda_kd=rep_lambda_kd, new_class_bias=new_class_bias,
                new_class_ids=new_class_ids, new_weight=w,
                kd=kd_flag, tag=f"search-{task_tag}-t{trial.number}", trial=trial)
        except optuna.TrialPruned:
            raise
        finally:
            del m, buf; torch.cuda.empty_cache()
        return val_acc

    study = _make_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    _print_search_summary(study, task_tag)
    return study.best_params


def optuna_search_rehearsal(task_tag, train_loader, val_loader,
                             exemplar_train_loader, extra_replay_loader,
                             teacher, cfg, device, lstm_hidden,
                             num_classes, num_old_classes,
                             limit=3, kd=True,
                             n_trials=20, search_epochs=3, new_repeat=10):
    print(f"\n{'='*60}\nOPTUNA REHEARSAL — {task_tag} | {n_trials} trials × {search_epochs} epochs")
    print(f"  Pruner: MedianPruner | Sampler: TPE (Bayesian)\n{'='*60}")

    def objective(trial):
        lambda_distill = trial.suggest_float("lambda_distill", 0.05, 1.0,  step=0.05)
        lstm_lr        = trial.suggest_float("lstm_lr",        1e-5, 1e-3, log=True)
        fc_lr          = trial.suggest_float("fc_lr",          1e-4, 1e-2, log=True)
        m   = fresh_model(num_classes=num_classes, cfg=cfg,
                          device=device, lstm_hidden=lstm_hidden)
        opt = torch.optim.Adam([{"params": m.lstm.parameters(), "lr": lstm_lr},
                                 {"params": m.fc.parameters(),   "lr": fc_lr}],
                               weight_decay=1e-4)
        buf = ReplayBuffer(max_size=1000)
        buf.add_from_loader(exemplar_train_loader, max_per_class=limit)
        if extra_replay_loader is not None:
            buf.add_from_loader(extra_replay_loader, max_per_class=limit)
        try:
            train_rehearsal(m, teacher, train_loader, val_loader,
                            buf, opt, device,
                            num_old_classes=num_old_classes,
                            lambda_distill=lambda_distill,
                            epochs=search_epochs, new_repeat=new_repeat, kd=kd)
            _, val_acc = evaluate(m, val_loader, device)
        finally:
            del m, opt, buf; torch.cuda.empty_cache()
        for ep in range(search_epochs):
            trial.report(val_acc, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return val_acc

    study = _make_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    _print_search_summary(study, task_tag)
    return study.best_params