import json
import os
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import torch.nn as nn



try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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

# identity model: needed because plot_umap expects model.get_embedding(x)
class IdentityEmbed(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    def get_embedding(self, x):
        return x

# loader that converts [B, T, 2048] -> [B, 2048] by temporal mean
def make_clip_embedding_loader(ds, batch_size=256, shuffle=False, num_workers=2):
    def collate(batch):
        xs, ys = [], []
        for t, y in batch:
            xs.append(t.mean(dim=0))            # [2048], average over frames
            ys.append(y)
        x = torch.stack(xs)                     # [B, 2048]
        y = torch.tensor(ys, dtype=torch.long)
        return x, y
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate)

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

def _project_2d(embeddings):
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    else:
        reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)

def plot_accuracy(results_dict, task_label):
    clean = {
        n: d for n, d in results_dict.items()
        if all(d.get(k) is not None for k in ("old_acc", "new_acc", "all_acc"))
    }

    if not clean:
        print(f"  [plot_accuracy] nothing to plot for '{task_label}' — all entries have None metrics.")
        return go.Figure()

    skipped = [n for n in results_dict if n not in clean]
    if skipped:
        print(f"  [plot_accuracy] skipped (None metrics): {skipped}")

    names = list(clean.keys())
    x     = list(range(len(names)))

    splits  = ["Old-task retention", "New-task plasticity", "Combined (CL)"]
    keys    = ["old_acc",            "new_acc",             "all_acc"]
    colors  = ["#4f98a3",            "#e8af34",             "#6daa45"]
    offsets = [-0.27,                0.0,                    0.27]

    fig = go.Figure()

    for label, key, color, off in zip(splits, keys, colors, offsets):
        vals = [clean[n][key] for n in names]
        fig.add_bar(
            x            = [i + off for i in x],
            y            = vals,
            name         = label,
            marker_color = color,
            width        = 0.22,
            text         = [f"{v:.1%}" for v in vals],
            textposition = "outside",
            textfont     = dict(size=11, color=TEXT_COLOR),
        )

    fig.update_layout(
        **_base_layout(f"{task_label} — Accuracy: Retention vs Plasticity vs CL"),
        barmode = "group",
        xaxis   = dict(
            tickvals   = x,
            ticktext   = names,
            tickangle  = -25 if max(len(n) for n in names) > 18 else 0,
            tickfont   = dict(size=12),
            showgrid   = False,
            title_text = "Method",
        ),
        yaxis   = dict(
            range      = [0, 1.14],
            tickformat = ".0%",
            tickfont   = dict(size=12),
            gridcolor  = GRID_COLOR,
            title_text = "Accuracy",
        ),
    )

    fig.update_layout(legend=dict(
        orientation = "h",
        yanchor     = "bottom",
        y           = 1.02,
        xanchor     = "right",
        x           = 1,
    ))

    fig.update_traces(cliponaxis=False)
    return fig

def plot_umap(results_dict, task_label, global_class_names=None):
    names = [n for n in results_dict
             if results_dict[n].get("embeddings") is not None
             and len(results_dict[n]["embeddings"]) > 0]
    n = len(names)
    
    if n == 0:
        return None

    all_unique_classes = set()
    for name in names:
        d = results_dict[name]
        labs = d["emb_labels"]
        cn = global_class_names or d.get("class_names") or [str(i) for i in range(int(labs.max()) + 1)]
        for idx in np.unique(labs):
            val = int(idx)
            class_str = cn[val] if val < len(cn) else str(val)
            all_unique_classes.add(class_str)
    
    sorted_classes = sorted(list(all_unique_classes))
    class_color_map = {cls: TAB_COLORS[i % len(TAB_COLORS)] for i, cls in enumerate(sorted_classes)}

    fig = make_subplots(
        rows=1, cols=n, 
        subplot_titles=[f"<b>{name}</b>" for name in names],
        horizontal_spacing=0.06
    )

    legend_tracker = set()

    for col, name in enumerate(names, start=1):
        d = results_dict[name]
        proj = _project_2d(d["embeddings"])
        labs = d["emb_labels"]
        cn = global_class_names or d.get("class_names") or [str(i) for i in range(int(labs.max()) + 1)]
        
        present_indices = np.unique(labs)

        for ci in present_indices:
            mask = (labs == ci)
            if not np.any(mask): continue
                
            val = int(ci)
            class_name = cn[val] if val < len(cn) else str(val)
            
            fig.add_trace(
                go.Scatter(
                    x=proj[mask, 0], y=proj[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=6, color=class_color_map[class_name],
                        opacity=0.7, line=dict(width=0.5, color='black')
                    ),
                    name=class_name,
                    legendgroup=class_name,
                    showlegend=(class_name not in legend_tracker),
                ),
                row=1, col=col
            )
            legend_tracker.add(class_name)

    proj_method = "UMAP" if UMAP_AVAILABLE else "PCA"
    fig.update_layout(**_base_layout(f"{task_label} — Latent Space ({proj_method})"))
    fig.update_layout(
        width=600 * n, height=700,
        legend=dict(title="<b>Classes</b>", orientation="v", x=1.02, y=0.5, xanchor="left"),
        margin=dict(t=100, b=80, l=50, r=150)
    )
    return fig

def plot_confusion(results_dict, task_label, global_class_names=None):
    names = [k for k, v in results_dict.items() if v.get("preds") is not None]
    n = len(names)
    if n == 0: 
        return None

    fig = make_subplots(
        rows=1, cols=n, 
        subplot_titles=[f"<b>{name}</b>" for name in names], 
        horizontal_spacing=0.12
    )

    for col, name in enumerate(names, start=1):
        d = results_dict[name]
        y_true, y_pred = d["labels"], d["preds"]
        present_indices = np.unique(np.concatenate([y_true, y_pred]))
        
        cn = global_class_names or d.get("class_names")
        if cn:
            tick_text = [cn[int(i)] if int(i) < len(cn) else str(i) for i in present_indices]
        else:
            tick_text = [str(i) for i in present_indices]

        cm = confusion_matrix(y_true, y_pred, labels=present_indices)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(cm_sum == 0, 0, cm / cm_sum)

        annot_text = []
        for i, row in enumerate(cm_norm):
            row_text = []
            for j, val in enumerate(row):
                if val < 0.005:
                    row_text.append("") 
                else:
                    color = "#00FF00" if i == j else "#FF0000"
                    row_text.append(f"<span style='color:{color}; font-size:14px'><b>{val:.2f}</b></span>")
            annot_text.append(row_text)

        # 3. Add Heatmap Trace
        fig.add_trace(
            go.Heatmap(
                z=cm_norm, 
                x=tick_text, 
                y=tick_text,
                text=annot_text, 
                texttemplate="%{text}",
                textfont={"family": "Arial Black", "size": 14},
                colorscale=[[0, "rgb(10,10,10)"], [1, "rgb(20, 40, 80)"]],
                zmin=0, zmax=1,
                showscale=(col == n), 
                xgap=1, ygap=1,
            ),
            row=1, col=col
        )

    fig.update_layout(**_base_layout(f"{task_label} — Confusion Analysis"))

    fig.update_layout(
        width=850 * n, 
        height=850,
        paper_bgcolor="black",
        plot_bgcolor="white", 
        font=dict(color="white")
    )

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='white', 
        mirror=True, tickfont=dict(color="white", size=11),
        gridcolor="rgba(255,255,255,0.1)"
    )
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='white', 
        mirror=True, tickfont=dict(color="white", size=11),
        autorange="reversed", scaleanchor="x", scaleratio=1
    )
    
    return fig

def plot_weight_delta(results_dict, task_label):
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
    return fig

def plot_retention_plasticity(results_t1, results_t2):
    task_data    = {"T1": results_t1, "T2": results_t2}
    task_markers = {"T1": "circle",   "T2": "square"}
    task_labels  = {"T1": "● T1",     "T2": "■ T2"}

    fig = go.Figure()

    # 1. Plot the actual data points
    for task_label, results in task_data.items():
        for name, d in results.items():
            if d.get("new_acc") is None or d.get("old_acc") is None:
                continue
                
            color = METHOD_COLORS.get(name, "#cdccca")
            
            fig.add_scatter(
                x=[d["new_acc"]],
                y=[d["old_acc"]],
                mode="markers+text",
                marker=dict(
                    size=18, color=color,
                    symbol=task_markers[task_label],
                    line=dict(width=1.5, color="white"),
                ),
                text=[f"  {name} ({task_label})"],
                textposition="middle right",
                textfont=dict(size=11, color="white"),
                name=f"{name} — {task_label}",
                showlegend=False,
            )

    fig.add_annotation(
        x=1.0, y=1.0, text="★ IDEAL (100/100)", showarrow=False,
        font=dict(size=14, color="#6daa45"), 
        xanchor="right", yanchor="bottom"
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

    layout_args = _base_layout("Retention (Old) vs. Plasticity (New) — T1 & T2 Comparison")
    
    layout_args.update(dict(
        paper_bgcolor="black",
        plot_bgcolor="black",
        legend=dict(
            orientation="v", 
            x=1.02, y=1.0,
            font=dict(size=11, color="white"), 
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="white",
            borderwidth=1
        ),
    ))

    fig.update_layout(**layout_args)

    fig.update_xaxes(
        title_text="New-task Plasticity (Learning)",
        range=[0, 1.1], tickformat=".0%",
        gridcolor=GRID_COLOR, linecolor="white", zeroline=False
    )
    fig.update_yaxes(
        title_text="Old-task Retention (Remembering)",
        range=[0, 1.1], tickformat=".0%",
        gridcolor=GRID_COLOR, linecolor="white", zeroline=False
    )

    return fig

def plot_per_class_accuracy(results_dict, task_label, global_class_names=None):
    names = [k for k, v in results_dict.items() if v.get("preds") is not None]
    if not names: return None

    n_cls = int(max([v["labels"].max() for v in results_dict.values()])) + 1
    raw_matrix = []
    for name in names:
        d = results_dict[name]
        y_true, y_pred = d["labels"], d["preds"]
        row = [(y_pred[y_true == ci] == ci).mean() if (y_true == ci).any() else np.nan for ci in range(n_cls)]
        raw_matrix.append(row)
    
    matrix_np = np.array(raw_matrix)

    valid_col_mask = ~np.all(np.isnan(matrix_np), axis=0)
    filtered_matrix = matrix_np[:, valid_col_mask]
    
    all_cn = global_class_names or results_dict[names[0]].get("class_names") or [str(i) for i in range(n_cls)]
    filtered_cn = [all_cn[i] for i, valid in enumerate(valid_col_mask) if valid]

    fig = go.Figure()
    fig.add_heatmap(
        z=filtered_matrix,
        x=filtered_cn,
        y=names,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="Accuracy", tickformat=".0%"),
        # Use filtered_matrix for annotations
        text=[[f"<b>{v:.2f}</b>" if not np.isnan(v) else "" for v in row] for row in filtered_matrix],
        texttemplate="%{text}",
        textfont=dict(size=12, family="Arial Black"),
    )

    fig.update_layout(**_base_layout(f"{task_label} — Per-Class Accuracy"))
    fig.update_layout(
        width=max(400, 120 * len(filtered_cn)), 
        height=100 * len(names) + 200,
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(t=100, b=100, l=180, r=50)
    )
    
    fig.update_xaxes(tickangle=45)
    return fig



def _buffer_class_counts(buf):
    counts = {}
    if hasattr(buf, "data") and isinstance(buf.data, list):
        for _, lbl in buf.data:
            # Safely handle both standard integers and PyTorch tensors
            lbl_val = lbl.item() if hasattr(lbl, "item") else int(lbl)
            counts[lbl_val] = counts.get(lbl_val, 0) + 1
    return counts


def plot_buffer_composition(buffer_stats, class_names=None):
    if not buffer_stats:
        print("  [plot_buffer_composition] empty buffer_stats")
        return None

    # Get the processed counts using our updated helper
    all_counts = {label: _buffer_class_counts(buf)
                  for label, buf in buffer_stats.items()}
    
    all_classes = sorted({c for counts in all_counts.values() for c in counts})
    buf_labels = list(buffer_stats.keys())
    fig = go.Figure()

    palette = [
        "#4f98a3", "#e8af34", "#6daa45", "#dd6974",
        "#a86fdf", "#fdab43", "#5591c7", "#bb653b",
    ]

    for idx, cls_id in enumerate(all_classes):
        if class_names and cls_id < len(class_names):
            cls_display_name = class_names[cls_id]
        else:
            cls_display_name = f"Class {cls_id}"
            
        vals = [all_counts[bl].get(cls_id, 0) for bl in buf_labels]
        
        fig.add_bar(
            x=buf_labels,
            y=vals,
            name=cls_display_name,
            marker_color=palette[idx % len(palette)],
            text=[str(v) if v > 0 else "" for v in vals],
            textposition="inside",
            textfont=dict(size=10, color="#1c1b19"),
        )

    totals = [sum(all_counts[bl].values()) for bl in buf_labels]
    for i, total in enumerate(totals):
        fig.add_annotation(
            x=i, y=total,
            text=f"<b>{total:,}</b>",
            showarrow=False,
            yshift=12,
            font=dict(size=12, color=TEXT_COLOR if 'TEXT_COLOR' in globals() else "black"),
        )

    max_y = max(totals) * 1.18 if totals else 10

    # Build the base layout configuration safely
    layout_args = {}
    if '_base_layout' in globals():
        layout_args = _base_layout("Replay Buffer Composition — samples per class")
    else:
        layout_args = dict(title=dict(text="Replay Buffer Composition — samples per class", x=0.5))
    
    layout_args.update(
        barmode="stack",
        margin=dict(r=160, t=80, b=50, l=50), 
        xaxis=dict(title_text="Buffer Type", tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title_text="Number of samples", 
                   gridcolor=GRID_COLOR if 'GRID_COLOR' in globals() else "#eee",
                   range=[0, max_y]),
        legend=dict(
            title_text="<b>Classes</b>",
            orientation="v",
            x=1.02,            
            xanchor="left",    
            y=1.0,
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)"  
        )
    )

    fig.update_layout(**layout_args)
    fig.update_traces(cliponaxis=False)
    
    return fig