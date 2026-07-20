"""
Embedding quality probes — "are these cached features any good?"

Run straight after Section 2's extraction, before training anything. Answers the
question that otherwise waits for Section 3:

    The ResNet was fine-tuned on 10 sports classes and then frozen. Do its features
    still carry the NEW classes (makeup, boxing), or did fine-tuning throw that away?

A probe answers it in seconds. If a 1-NN classifier on the raw frozen features already
separates Task1's classes, the representation is fine and any later failure is
forgetting, not a bad representation. If the probe is near chance, no amount of
continual-learning machinery will save you — stop and fix the features first.

Everything here mean-pools (N, T, D) -> (N, D) over time. That deliberately ignores
temporal order: it measures what the *frozen backbone* provides, independent of the
LSTM that Section 3 studies.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier


# ── Pooling ───────────────────────────────────────────────────────────────────
def to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Tensor -> ndarray, tolerating environments where .numpy() is unavailable
    (a torch/numpy ABI mismatch, or an unsupported dtype such as bfloat16).
    """
    t = t.detach().float().cpu()
    try:
        return t.numpy()
    except (RuntimeError, TypeError):
        return np.asarray(t.tolist(), dtype=np.float32)


def pool_features(feats: torch.Tensor) -> np.ndarray:
    """(N, T, D) -> (N, D) by averaging over time. Also accepts (N, D)."""
    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    return to_numpy(feats)


def _xy(cache, task_ids, split):
    xs, ys = [], []
    for t in task_ids:
        k = f"t{t}_{split}"
        if k in cache:
            f, y = cache[k]
            xs.append(pool_features(f))
            ys.append(to_numpy(y).astype(int))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


# ── Probes ────────────────────────────────────────────────────────────────────
def knn_probe(tr_x, tr_y, te_x, te_y, k=1):
    """1-NN accuracy. No training, no hyperparameters — pure geometry."""
    k = min(k, len(tr_x))
    m = KNeighborsClassifier(n_neighbors=k).fit(tr_x, tr_y)
    return float(m.score(te_x, te_y))


def linear_probe(tr_x, tr_y, te_x, te_y, max_iter=2000, seed=42):
    """
    Logistic-regression accuracy — the standard 'how much linearly separable
    information is in these features' measure.
    """
    mu, sd = tr_x.mean(0, keepdims=True), tr_x.std(0, keepdims=True) + 1e-6
    # multi_class= was deprecated in sklearn 1.5; the default is multinomial anyway.
    m = LogisticRegression(max_iter=max_iter, random_state=seed)
    m.fit((tr_x - mu) / sd, tr_y)
    return float(m.score((te_x - mu) / sd, te_y))


def probe_task(cache, task_id, k=1):
    """Probe one task in isolation (N-way within that task)."""
    tr_x, tr_y = _xy(cache, [task_id], "train")
    te_x, te_y = _xy(cache, [task_id], "test")
    if tr_x is None or te_x is None:
        return None
    n_cls = len(set(tr_y.tolist()))
    return {
        "n_classes": n_cls,
        "n_train":   len(tr_x),
        "n_test":    len(te_x),
        "chance":    1.0 / max(n_cls, 1),
        "knn":       knn_probe(tr_x, tr_y, te_x, te_y, k),
        "linear":    linear_probe(tr_x, tr_y, te_x, te_y),
    }


def probe_joint(cache, task_ids, k=1):
    """Probe every task together — the hard, meaningful number."""
    tr_x, tr_y = _xy(cache, task_ids, "train")
    te_x, te_y = _xy(cache, task_ids, "test")
    if tr_x is None or te_x is None:
        return None
    n_cls = len(set(tr_y.tolist()))
    return {
        "n_classes": n_cls,
        "n_train":   len(tr_x),
        "n_test":    len(te_x),
        "chance":    1.0 / max(n_cls, 1),
        "knn":       knn_probe(tr_x, tr_y, te_x, te_y, k),
        "linear":    linear_probe(tr_x, tr_y, te_x, te_y),
    }


def probe_report(cache, task_names, k=1, verbose=True):
    """
    Per-task probes plus the joint probe. Returns a dict keyed by task name plus
    'JOINT'.

    Read it like this:
        every task well above chance -> the frozen features carry all the classes;
                                        later failures are forgetting, not features
        a task near chance           -> the backbone genuinely lost those classes
        JOINT much lower than each   -> classes are separable alone but collide when
        task alone                      mixed (this is where confusions live)
    """
    ids = [t for t in range(len(task_names)) if f"t{t}_train" in cache]
    out = {}
    for t in ids:
        r = probe_task(cache, t, k)
        if r:
            out[task_names[t]] = r
    j = probe_joint(cache, ids, k)
    if j:
        out["JOINT"] = j

    if verbose:
        print(f"{'Group':<10}{'classes':>8}{'chance':>9}{'1-NN':>9}{'linear':>9}"
              f"{'vs chance':>11}")
        print("-" * 56)
        for name, r in out.items():
            mark = " <-- all tasks together" if name == "JOINT" else ""
            print(f"{name:<10}{r['n_classes']:>8}{r['chance']:>9.1%}"
                  f"{r['knn']:>9.1%}{r['linear']:>9.1%}"
                  f"{r['linear'] - r['chance']:>+10.1%}{mark}")
        print("-" * 56)
        j = out.get("JOINT")
        if j:
            if j["linear"] > 0.70:
                print("VERDICT: frozen features separate the classes well.")
                print("         Representation is NOT the bottleneck — later drops are forgetting.")
            elif j["linear"] > 0.40:
                print("VERDICT: features are usable but strained.")
                print("         Expect the Section 3 ceiling to sit here-ish.")
            else:
                print("VERDICT: features are weak for these classes.")
                print("         Fix this before trusting any CL result — consider a more")
                print("         diverse base-class set (uncomment classes in config.py).")
    return out


# ── Geometry ──────────────────────────────────────────────────────────────────
def class_centroids(cache, task_ids, split="train"):
    """{label: mean feature vector}."""
    x, y = _xy(cache, task_ids, split)
    return {int(c): x[y == c].mean(0) for c in np.unique(y)}


def centroid_distances(centroids: dict):
    """Pairwise L2 between class centroids. Returns (labels, matrix)."""
    labels = sorted(centroids)
    M = np.stack([centroids[l] for l in labels])
    d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=-1)
    return labels, d


def closest_pairs(centroids: dict, class_names: list, top=10):
    """
    The most-confusable class pairs by centroid distance.

    This is where the interesting story usually is: classes that sit close together
    are the ones that will fight during continual learning.
    """
    labels, d = centroid_distances(centroids)
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((d[i, j], class_names[labels[i]], class_names[labels[j]]))
    pairs.sort()
    return pairs[:top]


def print_closest_pairs(centroids: dict, class_names: list, task_classes: list,
                        task_names: list, top=10):
    """Print the most-confusable class pairs, flagging the ones that cross tasks."""
    task_of = {}
    for t, group in enumerate(task_classes):
        for c in group:
            task_of[c] = t

    print(f"{'dist':>8}  {'class A':<22} {'class B':<22} tasks")
    print("-" * 68)
    for d, a, b in closest_pairs(centroids, class_names, top):
        ta = task_of.get(a, -1)
        tb = task_of.get(b, -1)
        na = task_names[ta] if ta >= 0 else "?"
        nb = task_names[tb] if tb >= 0 else "?"
        flag = "  <- different tasks" if ta != tb else ""
        print(f"{d:8.2f}  {a:<22} {b:<22} [{na}/{nb}]{flag}")


def print_cache_summary(cache: dict, task_names: list):
    """Print clips, shape and size for every task/split in the cache."""
    print(f"{'key':<12}{'clips':>7}{'shape':>16}{'classes':>9}{'MB':>8}")
    print("-" * 52)
    total_clips = 0
    total_mb = 0.0
    for k in sorted(cache):
        f, y = cache[k]
        mb = f.element_size() * f.nelement() / 1e6
        total_clips += f.shape[0]
        total_mb += mb
        shape = f"({f.shape[1]}, {f.shape[2]})"
        print(f"{k:<12}{f.shape[0]:>7}{shape:>16}{len(set(y.tolist())):>9}{mb:>8.1f}")
    print("-" * 52)
    print(f"{'TOTAL':<12}{total_clips:>7}{'':>16}{'':>9}{total_mb:>8.1f}")


def separability(cache, task_ids, split="train", seed=42, max_n=3000):
    """
    Silhouette score in [-1, 1] — how tight and well-separated the class clusters are.
    Higher is better; near 0 means clusters overlap heavily.
    """
    x, y = _xy(cache, task_ids, split)
    if x is None or len(np.unique(y)) < 2:
        return None
    if len(x) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]
    return float(silhouette_score(x, y))


# ── Projection for plotting ───────────────────────────────────────────────────
def project_2d(cache, task_ids, split="train", method="tsne", seed=42,
               max_n=2000, perplexity=30):
    """
    Project pooled features to 2-D for visualisation. Returns (xy, labels).

    PCA to 50-d first — standard practice, and it makes t-SNE on 2048-d features
    roughly an order of magnitude faster without changing the picture.
    """
    from sklearn.manifold import TSNE

    x, y = _xy(cache, task_ids, split)
    if x is None:
        return None, None

    if len(x) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]

    if x.shape[1] > 50:
        x = PCA(n_components=50, random_state=seed).fit_transform(x)

    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(x), y

    ts = TSNE(n_components=2, random_state=seed,
              perplexity=min(perplexity, max(5, len(x) // 4 - 1)),
              init="pca", learning_rate="auto")
    return ts.fit_transform(x), y
