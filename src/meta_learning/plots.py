import json
import os
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#1c1b19"
GRID_COLOR = "#393836"
TEXT_COLOR = "#cdccca"
TAB_COLORS = px.colors.qualitative.Plotly

METHOD_COLORS = {
    "Rehearsal no-KD": "#e8af34",
    "Rehearsal KD":    "#b07a00",
    "Reptile no-KD":   "#4f98a3",
    "Reptile KD":      "#227f8b",
}
METHOD_COLORS_LIST = ["#4f98a3", "#227f8b", "#e8af34", "#b07a00"]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _base_layout(title):
    return dict(
        title={"text": title, "font": {"size": 16, "color": TEXT_COLOR}},
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)"
        ),
    )

def _extract_embeddings(model, loader, device):
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            h = model.get_embedding(x.to(device))
            embs.append(h.cpu())
            labs.append(y)
    return torch.cat(embs).numpy(), torch.cat(labs).numpy()

def _project_2d(embeddings):
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    else:
        reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)

def _save(fig, path):
    fig.write_image(path)
    meta_path = path + ".meta.json"
    name = path.replace(".png", "").replace("_", " ").title()
    with open(meta_path, "w") as f:
        json.dump({"caption": name, "description": f"Auto-generated plot: {name}"}, f)
    print(f"  saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# EXISTING PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_accuracy(results_dict, task_label, save_path=None):
    save_path = save_path or f"accuracy_{task_label}.png"
    names = list(results_dict.keys())
    x     = list(range(len(names)))

    splits  = ["Old-task retention", "New-task plasticity", "Combined (CL)"]
    keys    = ["old_acc",            "new_acc",             "all_acc"]
    colors  = ["#4f98a3",            "#e8af34",             "#6daa45"]
    offsets = [-0.27,                0.0,                   0.27]

    fig = go.Figure()
    for label, key, color, off in zip(splits, keys, colors, offsets):
        vals = [results_dict[n][key] for n in names]
        fig.add_bar(
            x=[i + off for i in x], y=vals,
            name=label, marker_color=color, width=0.22,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_COLOR),
        )

    fig.update_layout(
        **_base_layout(f"{task_label} — Accuracy: Retention vs Plasticity vs CL"),
        barmode="group",
        xaxis=dict(tickvals=x, ticktext=names, tickfont=dict(size=12), showgrid=False),
        yaxis=dict(range=[0, 1.12], tickformat=".0%", tickfont=dict(size=12),
                   gridcolor=GRID_COLOR, title_text="Accuracy"),
    )
    fig.update_xaxes(title_text="Method")
    fig.update_traces(cliponaxis=False)
    _save(fig, save_path)
    return fig

def plot_umap(results_dict, task_label, save_path=None):
    save_path = save_path or f"umap_{task_label}.png"
    names = [n for n in results_dict if results_dict[n]["embeddings"] is not None
             and len(results_dict[n]["embeddings"]) > 0]
    n = len(names)
    if n == 0:
        print("  [plot_umap] no embeddings found — pass val_loader to record()")
        return None

    fig = make_subplots(rows=1, cols=n, subplot_titles=names,
                        horizontal_spacing=0.06)

    for col, name in enumerate(names, start=1):
        d    = results_dict[name]
        proj = _project_2d(d["embeddings"])
        labs = d["emb_labels"]
        cn   = d["class_names"] or [str(i) for i in range(int(labs.max()) + 1)]
        n_cls = len(cn)

        for ci in range(n_cls):
            mask = labs == ci
            if not mask.any():
                continue
            fig.add_scatter(
                x=proj[mask, 0], y=proj[mask, 1],
                mode="markers",
                marker=dict(size=5, color=TAB_COLORS[ci % len(TAB_COLORS)], opacity=0.8),
                name=cn[ci],
                legendgroup=cn[ci],
                showlegend=(col == 1),
                row=1, col=col,
            )

    proj_method = "UMAP" if UMAP_AVAILABLE else "PCA"
    fig.update_layout(
        **_base_layout(f"{task_label} — Embedding Space ({proj_method}) per Method"),
        legend=dict(orientation="v", x=1.01, y=0.5, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title_text=f"{proj_method}-1", showgrid=False, zeroline=False,
                     tickfont=dict(size=10))
    fig.update_yaxes(title_text=f"{proj_method}-2", showgrid=False, zeroline=False,
                     tickfont=dict(size=10))
    _save(fig, save_path)
    return fig

def plot_confusion(results_dict, task_label, save_path=None):
    save_path = save_path or f"confusion_{task_label}.png"
    names = list(results_dict.keys())
    n     = len(names)

    fig = make_subplots(rows=1, cols=n, subplot_titles=names,
                        horizontal_spacing=0.10)

    for col, name in enumerate(names, start=1):
        d  = results_dict[name]
        cn = d["class_names"] or [str(i) for i in range(int(d["labels"].max()) + 1)]
        cm = confusion_matrix(d["labels"], d["preds"], labels=list(range(len(cn))))
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(cm.sum(axis=1, keepdims=True) == 0, 0,
                               cm / cm.sum(axis=1, keepdims=True))

        fig.add_heatmap(
            z=cm_norm,
            x=cn, y=cn,
            colorscale="Blues",
            zmin=0, zmax=1,
            showscale=(col == n),
            colorbar=dict(title="Rate", tickfont=dict(size=10)) if col == n else None,
            row=1, col=col,
        )

    fig.update_layout(
        **_base_layout(f"{task_label} — Confusion Matrices (normalised per row)"),
        legend=dict(visible=False),
    )
    fig.update_xaxes(title_text="Predicted", tickfont=dict(size=8), tickangle=35)
    fig.update_yaxes(title_text="True",      tickfont=dict(size=8), autorange="reversed")
    _save(fig, save_path)
    return fig

def plot_weight_delta(results_dict, task_label, save_path=None):
    save_path = save_path or f"weight_delta_{task_label}.png"

    layer_names = []
    for d in results_dict.values():
        if d["weight_delta"]:
            layer_names = list(d["weight_delta"].keys())
            break

    if not layer_names:
        print("  [plot_weight_delta] no weight_delta found — pass W0 to record()")
        return None

    def _short(k):
        return k.replace("weight_ih", "ih").replace("weight_hh", "hh") \
                .replace("bias_ih", "b-ih").replace("bias_hh", "b-hh") \
                .replace("_l0", "-L0").replace("_l1", "-L1")

    short_names = [_short(k) for k in layer_names]
    names       = list(results_dict.keys())

    fig = go.Figure()
    for i, (name, color) in enumerate(zip(names, METHOD_COLORS_LIST)):
        deltas = [results_dict[name]["weight_delta"].get(k, 0.0) for k in layer_names]
        fig.add_bar(
            y=short_names, x=deltas,
            name=name, orientation="h",
            marker_color=color, opacity=0.85,
        )

    fig.update_layout(
        **_base_layout(f"{task_label} — Weight Change per Layer (||ΔW||₂)"),
        barmode="group",
        yaxis=dict(tickfont=dict(size=11), showgrid=False, title_text="Layer"),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=11),
                   title_text="L2 norm of ΔW"),
    )
    fig.update_traces(cliponaxis=False)
    _save(fig, save_path)
    return fig

def plot_retention_plasticity(results_t1, results_t2, save_path=None):
    save_path = save_path or "retention_plasticity.png"

    task_data    = {"T1": results_t1, "T2": results_t2}
    task_markers = {"T1": "circle",   "T2": "square"}
    task_labels  = {"T1": "● T1",     "T2": "■ T2"}

    fig = go.Figure()

    for task_label, results in task_data.items():
        for name, d in results.items():
            color = METHOD_COLORS.get(name, "#cdccca")
            fig.add_scatter(
                x=[d["new_acc"]],
                y=[d["old_acc"]],
                mode="markers+text",
                marker=dict(
                    size=18, color=color,
                    symbol=task_markers[task_label],
                    line=dict(width=1.5, color=TEXT_COLOR),
                ),
                text=[f"  {name} ({task_label})"],
                textposition="middle right",
                textfont=dict(size=10, color=TEXT_COLOR),
                name=f"{name} — {task_label}",
                showlegend=False,
            )

    fig.add_annotation(
        x=0.97, y=0.97, text="★ ideal", showarrow=False,
        font=dict(size=13, color="#6daa45"), xref="paper", yref="paper",
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color=GRID_COLOR, dash="dot", width=1.5),
    )

    for name, color in METHOD_COLORS.items():
        fig.add_scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color, symbol="circle"),
            name=name, showlegend=True,
        )
    for task_label, symbol in task_markers.items():
        fig.add_scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color="#cdccca", symbol=symbol),
            name=task_labels[task_label], showlegend=True,
        )

    fig.update_layout(
        **_base_layout("Retention vs Plasticity — Rehearsal vs Reptile (T1 & T2)"),
        legend=dict(orientation="v", x=1.02, y=1.0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title_text="New-task plasticity",
                     range=[0, 1.08], tickformat=".0%",
                     gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(title_text="Old-task retention",
                     range=[0, 1.08], tickformat=".0%",
                     gridcolor=GRID_COLOR, zeroline=False)
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NEW PLOT 1 — Continual-learning accuracy curve over tasks
# ──────────────────────────────────────────────────────────────────────────────

def plot_cl_curve(history, save_path="cl_curve.png"):
    fig = go.Figure()

    for method, entries in history.items():
        if not entries:
            continue
        entries_sorted = sorted(entries, key=lambda e: e["task_id"])
        xs = [e["task_id"] for e in entries_sorted]
        ys = [e["all_acc"]  for e in entries_sorted]

        fig.add_scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=method,
            line=dict(color=METHOD_COLORS.get(method, None), width=2.5),
            marker=dict(size=9),
        )

    fig.update_layout(
        **_base_layout("Average accuracy over tasks (CL curve)"),
    )
    fig.update_xaxes(
        title_text="Task index (k)",
        gridcolor=GRID_COLOR,
        tickmode="linear", dtick=1,
    )
    fig.update_yaxes(
        title_text="Combined accuracy (all tasks seen so far)",
        tickformat=".0%",
        gridcolor=GRID_COLOR,
        range=[0, 1.05],
    )
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NEW PLOT 2 — Forgetting per task
# ──────────────────────────────────────────────────────────────────────────────

def plot_forgetting(forgetting_dict, save_path="forgetting.png"):
    methods = list(forgetting_dict.keys())
    if not methods:
        print("  [plot_forgetting] empty forgetting_dict")
        return None

    task_ids = sorted(next(iter(forgetting_dict.values())).keys())
    x        = np.arange(len(task_ids), dtype=float)
    width    = 0.8 / max(len(methods), 1)

    fig = go.Figure()
    for i, method in enumerate(methods):
        vals   = [forgetting_dict[method].get(t, 0.0) for t in task_ids]
        offset = (i - (len(methods) - 1) / 2.0) * width
        fig.add_bar(
            x=x + offset, y=vals,
            name=method,
            width=width * 0.92,
            marker_color=METHOD_COLORS.get(method, None),
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_COLOR),
        )

    all_vals = [v for m in methods for v in forgetting_dict[m].values()]
    y_max    = max(all_vals) * 1.3 if all_vals else 0.3

    fig.update_layout(
        **_base_layout("Forgetting per task — Rehearsal vs Reptile"),
        barmode="group",
    )
    fig.update_xaxes(
        title_text="Task ID",
        tickvals=x,
        ticktext=[f"T{t}" for t in task_ids],
        showgrid=False,
    )
    fig.update_yaxes(
        title_text="Forgetting  (best acc − final acc)",
        tickformat=".0%",
        gridcolor=GRID_COLOR,
        range=[0, y_max],
    )
    fig.update_traces(cliponaxis=False)
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NEW PLOT 3 — Run-to-run variability (box plot)
# ──────────────────────────────────────────────────────────────────────────────

def plot_run_variance(runs, save_path="run_variance.png"):
    if not runs:
        print("  [plot_run_variance] empty runs dict")
        return None

    fig = go.Figure()
    for method, accs in runs.items():
        if not accs:
            continue
        color = METHOD_COLORS.get(method, None)
        fig.add_box(
            y=accs,
            name=method,
            marker_color=color,
            line_color=color,
            boxmean=True,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            marker=dict(size=7, opacity=0.7),
        )

    fig.update_layout(
        **_base_layout("Run-to-run variability (final combined accuracy)"),
    )
    fig.update_xaxes(showgrid=False, title_text="Method")
    fig.update_yaxes(
        title_text="Final combined accuracy (all_acc)",
        tickformat=".0%",
        gridcolor=GRID_COLOR,
        range=[0, 1.05],
    )
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NEW PLOT 4 — Hyperparameter sensitivity (Optuna trials)
# ──────────────────────────────────────────────────────────────────────────────

def plot_hparam_sensitivity(trials, x_param, method_name, save_path=None):
    save_path = save_path or f"hparam_{method_name.replace(' ', '_')}_{x_param}.png"

    xs, ys, pruned_xs, pruned_ys = [], [], [], []
    for t in trials:
        if x_param not in t.params:
            continue
        val = t.params[x_param]
        if t.state.name == "COMPLETE" and t.value is not None:
            xs.append(val)
            ys.append(t.value)
        elif t.state.name == "PRUNED":
            pruned_xs.append(val)
            pruned_ys.append(0.0)

    if not xs:
        print(f"  [plot_hparam_sensitivity] no completed trials with param '{x_param}'")
        return None

    color = METHOD_COLORS.get(method_name, "#cdccca")

    fig = go.Figure()

    fig.add_scatter(
        x=xs, y=ys,
        mode="markers",
        name="Completed",
        marker=dict(color=color, size=9, line=dict(width=1, color=TEXT_COLOR)),
    )

    if pruned_xs:
        fig.add_scatter(
            x=pruned_xs, y=pruned_ys,
            mode="markers",
            name="Pruned",
            marker=dict(symbol="x", color="#888", size=8),
        )

    if xs:
        best_idx = int(np.argmax(ys))
        fig.add_scatter(
            x=[xs[best_idx]], y=[ys[best_idx]],
            mode="markers",
            name="Best",
            marker=dict(symbol="star", size=16, color="#6daa45",
                        line=dict(width=1, color=TEXT_COLOR)),
        )

    fig.update_layout(
        **_base_layout(f"{method_name} — {x_param} vs val acc"),
    )
    fig.update_xaxes(title_text=x_param, gridcolor=GRID_COLOR)
    fig.update_yaxes(
        title_text="Validation accuracy",
        tickformat=".0%",
        gridcolor=GRID_COLOR,
        range=[0, 1.05],
    )
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NEW PLOT 5 — Optuna trial history (val acc over trial index)
# ──────────────────────────────────────────────────────────────────────────────

def plot_optuna_history(studies, save_path="optuna_history.png"):
    fig = go.Figure()

    for method, study in studies.items():
        trials     = [t for t in study.trials if t.state.name == "COMPLETE"
                      and t.value is not None]
        trial_nums = [t.number for t in trials]
        values     = [t.value  for t in trials]

        if not values:
            continue

        running_best = []
        cur_best = -1.0
        for v in values:
            cur_best = max(cur_best, v)
            running_best.append(cur_best)

        color = METHOD_COLORS.get(method, None)

        fig.add_scatter(
            x=trial_nums, y=values,
            mode="markers",
            name=f"{method} (trials)",
            marker=dict(color=color, size=6, opacity=0.45),
            showlegend=True,
        )

        fig.add_scatter(
            x=trial_nums, y=running_best,
            mode="lines",
            name=f"{method} (best so far)",
            line=dict(color=color, width=2.5, dash="solid"),
            showlegend=True,
        )

    fig.update_layout(
        **_base_layout("Optuna search history — val acc per trial"),
    )
    fig.update_xaxes(title_text="Trial index", gridcolor=GRID_COLOR)
    fig.update_yaxes(
        title_text="Validation accuracy",
        tickformat=".0%",
        gridcolor=GRID_COLOR,
        range=[0, 1.05],
    )
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NEW PLOT 6 — Per-class accuracy heatmap (Rehearsal vs Reptile)
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_class_accuracy(results_dict, task_label, save_path=None):
    save_path = save_path or f"per_class_acc_{task_label}.png"
    names = list(results_dict.keys())

    first = next(iter(results_dict.values()))
    n_cls = int(first["labels"].max()) + 1
    cn    = first["class_names"] or [str(i) for i in range(n_cls)]

    matrix = []
    for name in names:
        d = results_dict[name]
        row = []
        for ci in range(n_cls):
            mask  = d["labels"] == ci
            if mask.sum() == 0:
                row.append(float("nan"))
            else:
                row.append((d["preds"][mask] == ci).mean())
        matrix.append(row)

    fig = go.Figure()
    fig.add_heatmap(
        z=matrix,
        x=cn,
        y=names,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="Accuracy", tickformat=".0%",
                      tickfont=dict(size=10)),
        text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=9),
    )

    fig.update_layout(
        **_base_layout(f"{task_label} — Per-class accuracy (Rehearsal vs Reptile)"),
        legend=dict(visible=False),
    )
    fig.update_xaxes(title_text="Class", tickangle=40, tickfont=dict(size=9),
                     showgrid=False)
    fig.update_yaxes(title_text="Method", tickfont=dict(size=11), showgrid=False)
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NEW PLOT 7 — Buffer composition (Replay vs Episode)
# ──────────────────────────────────────────────────────────────────────────────

def plot_buffer_composition(buffer_stats, class_names=None, save_path="buffer_comp.png"):
    """
    buffer_stats: dict with keys 'rehearsal' and 'reptile', each a list of dicts:
        rehearsal entry: { "task_tag": str, "total": int, "per_class": {cls_id: count, ...} }
        reptile entry:   { "task_tag": str, "total": int,
                           "per_class": {cls_id: count, ...},
                           "old_counts": {cls_id: count, ...},
                           "new_counts": {cls_id: count, ...} }
    """
    entries = []
    if buffer_stats.get("rehearsal"):
        entries.append(("Rehearsal", buffer_stats["rehearsal"][0]["per_class"]))
    if buffer_stats.get("reptile"):
        entries.append(("Reptile", buffer_stats["reptile"][0]["per_class"]))

    if not entries:
        print("  [plot_buffer_composition] no buffer stats")
        return None

    cls_ids = sorted(set(c for _, d in entries for c in d.keys()))
    if class_names is None:
        x = [str(c) for c in cls_ids]
    else:
        x = [class_names[c] for c in cls_ids]

    fig = go.Figure()
    for name, counts in entries:
        if name == "Rehearsal":
            color = METHOD_COLORS.get("Rehearsal KD", "#e8af34")
        else:
            color = METHOD_COLORS.get("Reptile KD", "#4f98a3")
        y = [counts.get(c, 0) for c in cls_ids]
        fig.add_bar(
            x=x,
            y=y,
            name=name,
            marker_color=color,
        )

    fig.update_layout(
        **_base_layout("Buffer composition — samples per class"),
        barmode="group",
    )
    fig.update_xaxes(title_text="Class", tickangle=40, tickfont=dict(size=9),
                     showgrid=False)
    fig.update_yaxes(title_text="# exemplars", gridcolor=GRID_COLOR,
                     tickfont=dict(size=11))
    _save(fig, save_path)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────

def plot_all(results_dict, task_label, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Generating plots for {task_label} ...")

    plot_accuracy(
        results_dict, task_label,
        save_path=f"{out_dir}/accuracy_{task_label}.png",
    )
    plot_umap(
        results_dict, task_label,
        save_path=f"{out_dir}/umap_{task_label}.png",
    )
    plot_confusion(
        results_dict, task_label,
        save_path=f"{out_dir}/confusion_{task_label}.png",
    )
    plot_weight_delta(
        results_dict, task_label,
        save_path=f"{out_dir}/weight_delta_{task_label}.png",
    )
    plot_per_class_accuracy(
        results_dict, task_label,
        save_path=f"{out_dir}/per_class_acc_{task_label}.png",
    )

    print(f"  All {task_label} plots saved to {out_dir}/")

def plot_all_tasks(
    results_t1,
    results_t2,
    history=None,
    forgetting_dict=None,
    runs=None,
    studies=None,
    buffer_stats=None,
    class_names=None,
    out_dir=".",
):
    os.makedirs(out_dir, exist_ok=True)

    plot_all(results_t1, task_label="T1", out_dir=out_dir)
    plot_all(results_t2, task_label="T2", out_dir=out_dir)

    plot_retention_plasticity(
        results_t1, results_t2,
        save_path=f"{out_dir}/retention_plasticity.png",
    )

    if history is not None:
        plot_cl_curve(
            history,
            save_path=f"{out_dir}/cl_curve.png",
        )

    if forgetting_dict is not None:
        plot_forgetting(
            forgetting_dict,
            save_path=f"{out_dir}/forgetting.png",
        )

    if runs is not None:
        plot_run_variance(
            runs,
            save_path=f"{out_dir}/run_variance.png",
        )

    if studies is not None:
        plot_optuna_history(
            studies,
            save_path=f"{out_dir}/optuna_history.png",
        )

    if buffer_stats is not None:
        plot_buffer_composition(
            buffer_stats,
            class_names=class_names,
            save_path=f"{out_dir}/buffer_comp.png",
        )

    print(f"\n  All plots saved to {out_dir}/")