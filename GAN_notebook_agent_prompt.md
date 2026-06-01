# Agent Prompt: Implement GIL GAN Notebook for ULAL Project

## Task

Implement a new Jupyter notebook called `09_gil_gan.ipynb` inside
`notebooks/` that adds the **GAN-based Generative Iterative Learning (GIL)**
pipeline from the paper:

> **"Continual Learning Improves Zero-Shot Action Recognition"**  
> Gowda, Moltisanti & Sevilla-Lara — arXiv 2410.10497  
> https://arxiv.org/abs/2410.10497

This notebook must fit seamlessly into the existing **Unified Lifelong Action
Learning (ULAL)** project, reusing all existing `src/` modules and following
the exact style of the other notebooks (01–08).

---

## Paper Summary (GIL — What You Are Implementing)

The paper proposes **Generative Iterative Learning (GIL)**, which uses a
continual learning replay memory to prevent catastrophic forgetting during
zero-shot action recognition fine-tuning.

### Core idea
Instead of directly fine-tuning a video model on seen classes (which causes it
to forget pre-training knowledge), GIL builds a **Replay Memory** of class
prototypes and uses a **GAN** to synthesise features from those prototypes.
During fine-tuning, real features from new classes are mixed with synthetic
features from all previously seen classes — the model never forgets old classes
because it keeps "replaying" them via generated features.

### Three training phases (looped iteratively)

**Phase 1 — Initialisation**
- Freeze the backbone model M (ResNet50+LSTM in our project).
- Extract visual features for every instance in the pre-training / base task classes.
- Per class C, compute: μ(C) = mean of all instance features, σ(C) = std of all instance features.
- Store (μ, σ) pairs in the **Replay Memory Buffer**.
- Train the **CVAE (E)**: takes a semantic embedding `a` (class name text vector)
  as input; outputs (μ̂, σ̂) matching the stored prototype and noise.
  Loss = MSE( (μ̂, σ̂), (μ, σ) ).
- Train the **Feature Generator GAN (F + G + H)**:
  - Generator F: takes (μ, σ, z~N(0,1)) → synthesised feature x̂ ∈ R^d
  - Discriminator G: takes (x, a) → real/fake scalar
  - Projection head H: takes x → ã (reconstructed semantic embedding)
  - GAN is trained with the WGAN-GP objective + classification loss L_CLS + mutual information loss L_MI (see equations 1–2 in the paper)
  - After Phase 1, **F is frozen permanently** — never updated again.

**Phase 2 — Incremental Learning** (core contribution)
- At each iteration t, sample 10% of the fine-tuning (seen) classes as "new classes" C_t.
- Extract real features f_j^k = M(x_j) for each video x_j in C_t using the frozen backbone.
- Generate synthetic features f̂_j^k = F(μ_k, σ_k) for every class k in the buffer C_{t-1} (old knowledge replay).
- Mix real (new) + synthetic (old) features and fine-tune **only the last 2 FC layers of M** using a cross-entropy MLP classifier. Do NOT update the backbone.
- Iterate until all seen classes have been introduced (in 10% chunks).

**Phase 3 — Update**
- After each incremental step, compute and store new class prototypes (μ, σ) in the buffer.
- Fine-tune E (CVAE) on the new classes so it can generate their prototypes from semantic embeddings.
- F stays frozen.

### Testing (Zero-Shot)
- Feed unseen class semantic embeddings through E → (μ̂, σ̂) → F → synthetic features x̂.
- Fine-tune the last 2 layers of M on these synthetic unseen-class features.
- At inference: embed a test video with M, then do **1-NN search** over unseen class embeddings.

---

## How This Connects to the Existing ULAL Project

The ULAL project already has (in `src/`):

| Module | What it provides for this notebook |
|---|---|
| `src/config/config.py` | Class splits (SELECTED_CLASSES as base/pre-train, TASK_1–TASK_4 as seen fine-tune classes), dataset roots, transforms |
| `src/data/dataset.py` | `UCF101Clips` dataset — use this to load video clips for feature extraction |
| `src/models/teacher.py` | `ResNet50LSTMTeacher` — this IS the backbone model M; freeze it and use its LSTM output as the visual feature vector |
| `src/models/head.py` | `EmbeddingHead` — reuse or adapt as the MLP classifier on top of M during incremental fine-tuning |
| `src/cl/smart_replay.py` | `SmartReplayBuffer` — conceptually similar to GIL's buffer; inspect it but build a new `GILReplayBuffer` tailored to storing (μ, σ) per class |
| `src/utils/metrics.py` | `cl_report`, `evaluate_all_tasks`, Forgetting/BWT metrics — use for final evaluation |
| `src/utils/visualize.py` | Training curves and confusion matrix — reuse for plots |

**Semantic embeddings (`a`):** Use **word2vec / sentence embeddings** of the UCF101 class names. Since Stories embeddings require external data, use `sentence-transformers` (`all-MiniLM-L6-v2`) to encode class name strings into 384-dim vectors. This is consistent with what the paper calls "sen2vec" and is the easiest self-contained option on UCF101.

**Visual feature dimension:** `ResNet50LSTMTeacher` outputs a 512-dim LSTM hidden state. Use this as the feature vector `x ∈ R^512`. All GAN networks should use hidden size 4096 (as in the paper), input/output adapting to 512.

**Classes:** Use `SELECTED_CLASSES` (10 classes) as the pre-training / initialisation set and `TASK_1` + `TASK_2` as seen fine-tuning classes (introduced 10% at a time as in the paper). Reserve `TASK_3` or a held-out subset as the **unseen / zero-shot test classes**.

---

## Notebook Structure to Implement

Follow the pattern of existing notebooks exactly: Drive mount cell, src fetch cell, then numbered sections.

### Section 0 — Setup
```python
# Mount Drive, add src to path, import everything
# pip install sentence-transformers
from src.config.config import *
from src.data.dataset import UCF101Clips
from src.models.teacher import ResNet50LSTMTeacher
from src.models.head import EmbeddingHead
from src.utils.metrics import cl_report
from src.utils.visualize import plot_curves
```

### Section 1 — Semantic Embeddings
- Load `sentence-transformers` model `all-MiniLM-L6-v2`.
- Encode all class name strings (base + seen + unseen) → dict `{class_name: tensor(384)}`.
- This is the semantic embedding `a` used by E throughout.

### Section 2 — Feature Extraction (Initialisation Phase, Part 1)
- Load pretrained `ResNet50LSTMTeacher` checkpoint (from notebook 02).
- Freeze all parameters.
- Loop over base classes (`SELECTED_CLASSES`) using `UCF101Clips` + DataLoader.
- Forward pass each clip → 512-dim LSTM output = visual feature.
- Compute per-class μ and σ. Store in `GILReplayBuffer`.

### Section 3 — GAN Architecture
Implement the following four PyTorch `nn.Module` classes (new file `src/fine_tune/gil_gan.py` — also write this file):

**`CVAE` (E):**
```
Encoder: Linear(sem_dim=384) → Linear(2048) → Linear(1024) → μ_z, log_σ_z (latent_dim=512)
Decoder: z(512) → Linear(1024) → Linear(2048) → Linear(feat_dim*2=1024)  # outputs (μ̂, σ̂) concatenated
Loss: MSE( (μ̂, σ̂), (μ_real, σ_real) ) + KL divergence term
```

**`FeatureGenerator` (F):**
```
Input: concat(μ(512), σ(512), z~N(0,I)(512)) → Linear(4096) → ReLU → Linear(4096) → ReLU → Linear(feat_dim=512)
```

**`Discriminator` (G):**
```
Input: concat(x(512), a(384)) → Linear(4096) → LeakyReLU → Linear(4096) → LeakyReLU → Linear(1)
```

**`ProjectionHead` (H):**
```
Input: x(512) → Linear(4096) → ReLU → Linear(sem_dim=384)
```

### Section 4 — GAN Training (Initialisation Phase, Part 2)
Implement the WGAN-GP training loop:
- L_D = E[G(x, H(x))] − E[G(F(a,z), a)] − α * gradient_penalty  (Eq. 1 of paper, α=10)
- L_CLS: cross-entropy on F(a,z) predicted class (add a linear classifier on top of F during training only)
- L_MI: mutual information loss = −cosine_similarity(H(F(a,z)), a) (approximation; paper uses MINE)
- Full objective: min_F min_H max_G  L_D + λ1*L_CLS + λ2*L_MI  (λ1=0.01, λ2=0.1)
- Train for 30 epochs with Adam lr=1e-4, batch_size=64.
- After training: **freeze F permanently**.
- Plot discriminator loss, generator loss, classification loss curves.

### Section 5 — Incremental Learning Loop
```
seen_classes = TASK_1 + TASK_2  # introduce 10% per iteration
chunk_size = max(1, len(seen_classes) // 10)

for t, class_chunk in enumerate(chunks(seen_classes, chunk_size)):
    # Extract real features for class_chunk using frozen M backbone
    real_feats, real_labels = extract_features(M, class_chunk)

    # Generate synthetic features for all classes already in buffer
    synth_feats, synth_labels = buffer.generate_all(F, J=len(real_feats)//len(buffer))

    # Mix and create a DataLoader
    mixed = concat(real_feats, synth_feats), concat(real_labels, synth_labels)

    # Fine-tune ONLY last 2 FC layers of M (keep backbone frozen)
    fine_tune_last2(M, mixed, epochs=5, lr=1e-4)

    # UPDATE PHASE: add new prototypes to buffer, fine-tune E
    update_buffer_and_cvae(M, E, class_chunk, buffer)

    print(f"Iteration {t}: buffer size = {len(buffer)}, seen so far = {buffer.class_list}")
```
Plot accuracy on seen classes after each iteration to show anti-forgetting effect.

### Section 6 — Zero-Shot Testing
- Unseen classes = `TASK_3` (held out, never trained on).
- For each unseen class:
  1. Get semantic embedding `a` via sentence-transformers.
  2. Feed to E → (μ̂, σ̂).
  3. Feed to F (sampled J=50 times) → 50 synthetic features per class.
- Fine-tune last 2 layers of M on these synthetic unseen features (ZSL final step).
- At test time: embed test videos with M → 1-NN against unseen class centroids.
- Report **Top-1 ZSL accuracy** and **GZSL harmonic mean** (seen acc + unseen acc).

### Section 7 — Ablation & Comparison
- Compare against: (a) no replay memory, (b) random sample replay (existing `ReplayBuffer` from `src/cl/rehearsal.py`), (c) GIL prototype+noise.
- Reuse `cl_report` and `evaluate_all_tasks` from `src/utils/metrics.py`.
- Show forgetting (BWT) for each approach.
- One summary table: Method | ZSL Top-1 | GZSL H | Forgetting.

### Section 8 — t-SNE Visualisation
- Plot t-SNE of real vs. synthetic features for 5 random classes (mirrors Fig. 4 of paper).
- Show that synthetic features form tighter clusters.

---

## File to Create Alongside the Notebook

**`src/fine_tune/gil_gan.py`** — PyTorch module file containing:
- `CVAE` class
- `FeatureGenerator` class  
- `Discriminator` class
- `ProjectionHead` class
- `GILReplayBuffer` class (stores {class_name: (μ, σ, semantic_emb)}, method `generate_all(F, J)`)
- `gradient_penalty(D, real, fake, a, device)` helper
- `train_gan(F, G, H, dataloader, device, epochs, λ1, λ2, α)` training function
- `fine_tune_last2(model, mixed_loader, epochs, lr, device)` function
- `update_buffer_and_cvae(model, E, new_classes, buffer, optimizer_E, device)` function
- `zero_shot_test(model, F, E, unseen_classes, sem_embs, test_loader, device)` function

---

## Coding Standards (match existing project)

- Python 3.10, PyTorch 2.x, no TensorFlow.
- Every section has a markdown header cell explaining what it does and why.
- Print loss every 5 epochs; use `tqdm` for dataloader loops.
- Save checkpoints to `/content/drive/MyDrive/apai/gil/` (create dir if needed).
- Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- All random seeds set to 42 at top of notebook.
- No external dataset downloads needed — use the UCF101 data already set up by notebook 01.
- Notebook must be runnable top-to-bottom in one pass on Google Colab (T4 GPU).

---

## Key Hyperparameters (from paper)

| Parameter | Value |
|---|---|
| Hidden size (GAN networks) | 4096 |
| Visual feature dim | 512 (ResNet50+LSTM output) |
| Semantic embedding dim | 384 (sentence-transformers MiniLM) |
| Latent dim (CVAE) | 512 |
| Noise dim (F input) | 512 |
| GAN lr | 1e-4 (Adam) |
| WGAN-GP penalty α | 10 |
| λ1 (classification loss) | 0.01 |
| λ2 (mutual info loss) | 0.1 |
| Incremental chunk size | 10% of seen classes per iteration |
| Synthetic samples per class (J) | ≈ avg samples in new class chunk |
| Fine-tune layers | Last 2 FC layers of M only |
| Fine-tune lr | 1e-4 |
| Fine-tune epochs per iter | 5 |
| GAN training epochs | 30 |

---

## What Success Looks Like

1. `src/fine_tune/gil_gan.py` exists and all classes/functions import cleanly.
2. `notebooks/09_gil_gan.ipynb` runs cell-by-cell without errors.
3. GAN discriminator and generator losses converge (D loss stabilises near 0, G loss decreases).
4. ZSL Top-1 accuracy on TASK_3 unseen classes is higher than the naive fine-tune baseline from notebook 03.
5. Forgetting metric (BWT) is lower (less negative) than naive baseline — proving the replay memory works.
6. t-SNE plot shows synthetic features form tighter clusters than real features.
