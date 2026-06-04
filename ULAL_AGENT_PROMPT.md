# ULAL — Full Pipeline Agent Prompt
# Unified Lifelong Action Learning on UCF101

> **This file is the master instruction document for implementing the complete ULAL pipeline.**
> Read it top to bottom before writing a single line of code. Every section maps to one notebook.
> Follow the order strictly — each stage depends on the checkpoint produced by the previous one.

---

## Project Overview

The goal is a **unified video action recognition system** that can:
1. Learn a set of base action classes from UCF101
2. Continually learn new classes without forgetting old ones (Continual Learning)
3. Compress knowledge from a large teacher into a lightweight student (Knowledge Distillation)
4. Recognise action classes it was never trained on (Zero-Shot Learning via GIL-GAN)
5. Adapt to visually different video sources like YouTube (Active Domain Adaptation)

The key research claim is: **each stage adds exactly one new capability, and together they form a coherent lifelong learning system.**

---

## Repository Structure (existing — do not move or rename)

```
src/
  active/
    acquisition.py       # select_top_k — AL acquisition functions (entropy, margin, coreset)
    domain_shift.py      # compute_class_stats, measure_shift, reptile_adapt
  cl/
    naive.py             # naive continual learning (no replay)
    ewc.py               # EWC regularisation
    rehearsal.py         # ReplayBuffer + train_continual
    smart_replay.py      # SmartReplayBuffer (hard/diverse exemplar selection)
  config/
    config.py            # SELECTED_CLASSES, TASK_1-4, BASE_ROOT, RESNET50_PATH, transforms
  data/
    dataset.py           # UCF101Clips dataset class
    preprocessing.py     # preprocess_dataset
  fine_tune/
    gil_gan.py           # CVAE, FeatureGenerator, Discriminator, ProjectionHead,
                         # GILReplayBuffer, train_gan, fine_tune_last2,
                         # update_buffer_and_cvae, zero_shot_test
  kd/
    mseLoss.py           # finetune_head (alias → kd/trainer.py)
    trainer.py           # finetune_head
    utils.py             # extract_embeddings, eval_head, restore_head_to_student
  meta/
    reptile.py           # _snap, _reptile_update
    sampler.py           # sample, limit_to_k_per_class
  meta_learning/
    models.py            # ResNet50LSTMTeacher, MobileNetV3SmallLSTMStudent, EmbeddingHead
    reptile.py           # train_reptile
    sampler.py           # sample, limit_to_k_per_class
  models/
    head.py              # EmbeddingHead
  utils/
    metrics.py           # cl_report, build_accuracy_matrix
    visualize.py         # plot_training_results

notebooks/
  study/
    choose_model/
      01_setup_data.ipynb          # data preprocessing
      02_backbone_transfer.ipynb   # teacher training
    cl/
      03_naive_cl.ipynb            # naive CL baseline
      04_ewc_cl.ipynb              # EWC CL
      05_replay_lwf.ipynb          # replay + LwF
    mamal/
      06_reptile_kd.ipynb          # Reptile meta-learning + KD student
    active_dom_adopt/
      07_active_domain_adapt.ipynb # Active Domain Adaptation (FIXED)
    gan_reply_buffer/
      09_gil_gan.ipynb             # GIL-GAN zero-shot
  ULAL.ipynb                       # ← MASTER NOTEBOOK (this file's target)
```

---

## Class Configuration (from src/config/config.py — do not change)

```
SELECTED_CLASSES (10 base classes):
  PlayingTabla, PommelHorse, JumpingJack, PushUps, PoleVault,
  HorseRace, HighJump, Drumming, HorseRiding, Diving

TASK_1 (3 seen classes, CL step 1):
  ApplyEyeMakeup, ApplyLipstick, Archery

TASK_2 (3 seen classes, CL step 2):
  BoxingPunchingBag, BoxingSpeedBag, BrushingTeeth

TASK_3 (10 unseen / zero-shot test classes — NEVER train on these):
  SoccerJuggling, Skijet, Skiing, SkateBoarding, Rowing,
  RopeClimbing, RockClimbingIndoor, Punch, PullUps, PlayingViolin

TASK_4 (10 additional unseen classes — optional extension):
  PlayingPiano, PlayingGuitar, HulaHoop, GolfSwing, Fencing,
  CleanAndJerk, Billiards, Biking, BenchPress, BaseballPitch
```

---

## Checkpoints Directory Convention

All checkpoints must be saved to and loaded from:
```
/content/drive/MyDrive/apai/checkpoints/
```

| File | Produced by | Used by |
|---|---|---|
| `ResNet50_10C.pth` | Notebook 02 (pre-existing) | All notebooks |
| `student_ucf101.pt` | Notebook 06 | Notebooks 07, 09, ULAL |
| `head_ucf101.pt` | Notebook 06 | Notebooks 07, 09, ULAL |
| `head_adapted_yt.pt` | Notebook 07 | ULAL |
| `generator.pth` | Notebook 09 | ULAL |
| `cvae_init.pth` | Notebook 09 | ULAL |
| `cvae_meta.pth` | ULAL (new) | ULAL |
| `classifier_incremental.pth` | Notebook 09 | ULAL |

---

## Dimension Constants (must be consistent across all notebooks)

| Constant | Value | Why |
|---|---|---|
| `FEAT_DIM` | 256 | ResNet50+LSTM hidden_size=256 (checkpoint) |
| `TEACHER_HIDDEN` | 256 | Matches saved teacher checkpoint |
| `STUDENT_HIDDEN` | 128 | MobileNetV3 student LSTM hidden size |
| `SEM_DIM` | 384 | sentence-transformers all-MiniLM-L6-v2 output |
| `LATENT_DIM` | 256 | CVAE latent space |
| `NOISE_DIM` | 256 | Generator noise input |
| `GAN_HIDDEN` | 4096 | GAN hidden layers (from GIL paper) |

---

## Global Coding Standards

- Python 3.10, PyTorch 2.x. No TensorFlow.
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- All seeds fixed to 42: `random.seed(42); np.random.seed(42); torch.manual_seed(42)`
- `sys.path.insert(0, '/content')` — import as `from src.xxx import yyy`
- Save all checkpoints to Drive immediately after training completes
- Print loss every 5 epochs; use `tqdm` for all DataLoader loops
- Every notebook section has a markdown cell explaining what it does and why
- Runnable top-to-bottom on Google Colab T4 GPU in one pass

---

---

# NOTEBOOK: ULAL.ipynb — Master Pipeline

> This is the single end-to-end notebook that ties all stages together.
> It does NOT re-implement what the study notebooks already have.
> It LOADS checkpoints from each stage and runs the unified evaluation.
> It also implements the **new meta-learning ZSL improvement** not present in any study notebook.

---

## Section 0 — Configuration Cell

```python
import os
os.environ["GITHUB_TOKEN"] = "ghp_YOURTOKEN"
os.environ["BRANCH"] = "GAN&ActiveLearning"

DRIVE_PROJECT = "/content/drive/MyDrive/apai"
CKPT_DIR      = f"{DRIVE_PROJECT}/checkpoints"
RESULTS_DIR   = f"{DRIVE_PROJECT}/results/ulal"

# YouTube clips for domain adaptation (classes must be in SELECTED_CLASSES)
YOUTUBE_CLIPS = {
    "Diving":      "https://www.youtube.com/watch?v=sM4w7GgJjNs",
    "HorseRiding": "https://www.youtube.com/watch?v=dMoWGWA_sFk",
    "PushUps":     "https://www.youtube.com/watch?v=IODxDxX7oi4",
}

# Active Learning
AL_STRATEGY         = "entropy"   # "entropy" | "margin" | "coreset"
AL_BUDGET_PER_CLASS = 10

# GIL / ZSL
GAN_EPOCHS      = 150             # increase from 30 — critical for ZSL quality
J_SYNTH         = 200             # synthetic samples per class (was 50 — too low)
CVAE_INIT_EPOCHS = 100            # more CVAE pre-training epochs
CVAE_UPDATE_EPOCHS = 20           # more CVAE update epochs per incremental step

# Meta-learning on CVAE (NEW — improves ZSL)
META_EPISODES    = 60             # Reptile episodes for CVAE meta-training
META_INNER_LR    = 1e-4
META_INNER_STEPS = 5
META_EPSILON     = 0.15

# Reptile domain adaptation
ADAPT_EPISODES    = 30
ADAPT_INNER_LR    = 0.0005
ADAPT_INNER_STEPS = 10
ADAPT_EPSILON     = 0.20

print("Config loaded.")
```

---

## Section 1 — Setup: Mount Drive, Install, Fetch src

```python
# Cell 1a: Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Cell 1b: Install dependencies
!pip install -q yt-dlp sentence-transformers umap-learn

# Cell 1c: Fetch src from GitHub
import subprocess, sys, os
!cp "{DRIVE_PROJECT}/bash/fetch_src.sh" /content/fetch_src.sh
subprocess.run(["bash", "/content/fetch_src.sh"], check=True)
sys.path.insert(0, "/content")
print("src/ ready.")
```

---

## Section 2 — Imports

```python
import gc, copy, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

# Project modules
import src.config.config as cfg
from src.meta_learning.models  import (ResNet50LSTMTeacher,
                                        MobileNetV3SmallLSTMStudent,
                                        EmbeddingHead)
from src.kd.utils              import extract_embeddings, eval_head
from src.data.dataset          import UCF101Clips
from src.active.acquisition    import select_top_k
from src.active.domain_shift   import (compute_class_stats, measure_shift,
                                        reptile_adapt)
from src.cl.smart_replay       import SmartReplayBuffer
from src.fine_tune.gil_gan     import (CVAE, FeatureGenerator, Discriminator,
                                        ProjectionHead, GILClassifier,
                                        GILReplayBuffer, train_gan,
                                        fine_tune_last2, update_buffer_and_cvae,
                                        zero_shot_test, gradient_penalty)
from sentence_transformers import SentenceTransformer

# Dimensions (must match checkpoints)
FEAT_DIM        = 256
TEACHER_HIDDEN  = 256
STUDENT_HIDDEN  = 128
HEAD_DROPOUT    = 0.25
STUDENT_DROPOUT = 0.20
SEM_DIM         = 384
LATENT_DIM      = 256
NOISE_DIM       = 256
BATCH_SIZE      = 8
NUM_WORKERS     = 2

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = device.type == "cuda"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)
print(f"Device: {device}")
```

---

## Section 3 — Data Preprocessing

**What:** Convert raw UCF101 .avi videos into 16-frame .pt clip tensors.
**Why:** All subsequent stages operate on pre-processed clips, not raw video.
**Skip if:** The processed folders already exist from previous runs.

```python
from src.data.preprocessing import preprocess_dataset

for root, classes in [
    (cfg.BASE_ROOT,  cfg.SELECTED_CLASSES),
    (cfg.TASK1_ROOT, cfg.TASK_1),
    (cfg.TASK2_ROOT, cfg.TASK_2),
    (cfg.TASK3_ROOT, cfg.TASK_3),
]:
    if not os.path.exists(root):
        cfg.OUTPUT_ROOT = root
        preprocess_dataset(splits=[], target_classes=classes)
        print(f"Preprocessed → {root}")
    else:
        print(f"[skip] {root} already exists")
```

---

## Section 4 — Load Models

**What:** Load the pretrained ResNet50+LSTM teacher and initialise the MobileNetV3 student.
**Why:** Teacher is the backbone for all feature extraction. Student is the lightweight model we will train via KD.
**CRITICAL:** Load the student checkpoint if it exists. Without it the student LSTM is random and all downstream results are meaningless.

```python
def _remap_teacher_ckpt(raw):
    return {
        ("backbone." + k[len("resnet."):] if k.startswith("resnet.") else k): v
        for k, v in raw.items() if not k.endswith("num_batches_tracked")
    }

# Global class mapping — consistent across all stages
global_classes  = cfg.SELECTED_CLASSES + cfg.TASK_1 + cfg.TASK_2 + cfg.TASK_3
all_classes     = global_classes
c2i             = {c: i for i, c in enumerate(cfg.SELECTED_CLASSES)}  # base only for teacher
global_c2i      = {c: i for i, c in enumerate(global_classes)}
label_to_class  = {i: c for c, i in global_c2i.items()}
num_base        = len(cfg.SELECTED_CLASSES)

# Teacher
teacher = ResNet50LSTMTeacher(TEACHER_HIDDEN, num_base, 0.4).to(device)
raw_ckpt = torch.load(cfg.RESNET50_PATH, map_location=device)
mapped   = _remap_teacher_ckpt(raw_ckpt)
missing, _ = teacher.load_state_dict(mapped, strict=False)
if [k for k in missing if "num_batches_tracked" not in k]:
    raise RuntimeError("Teacher checkpoint has missing keys — check RESNET50_PATH")
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()
print(f"Teacher loaded and frozen. FEAT_DIM={FEAT_DIM}")

# Student
num_classes_base = num_base
student = MobileNetV3SmallLSTMStudent(num_classes_base, STUDENT_HIDDEN, STUDENT_DROPOUT).to(device)
student_ckpt = f"{CKPT_DIR}/student_ucf101.pt"
if os.path.exists(student_ckpt):
    student.load_state_dict(torch.load(student_ckpt, map_location=device))
    print(f"Student loaded from {student_ckpt}")
else:
    print("WARNING: No student checkpoint. Student LSTM is random.")
    print("Either run Notebook 06 first, or the KD stage below will train it from scratch.")

# EmbeddingHead
head = EmbeddingHead(STUDENT_HIDDEN, TEACHER_HIDDEN,
                     num_classes=num_classes_base,
                     dropout_p=HEAD_DROPOUT).to(device)
head_ckpt = f"{CKPT_DIR}/head_ucf101.pt"
if os.path.exists(head_ckpt):
    head.load_state_dict(torch.load(head_ckpt, map_location=device))
    print(f"Head loaded from {head_ckpt}")
else:
    print("WARNING: No head checkpoint. Head is randomly initialised.")
```

---

## Section 5 — UCF101 Feature Extraction (cached)

**What:** Extract 256-dim LSTM hidden states from teacher and student for all base + task splits.
**Why:** Downstream CL, KD, GIL, and AL stages all operate on these embeddings, not raw clips. Caching avoids re-running the expensive backbone every time.
**Output:** `train_s, train_t, train_y, val_s, val_t, val_y, test_s, test_t, test_y`

```python
UCF_EMB_CACHE = f"{DRIVE_PROJECT}/cache/ucf101_embeddings.pt"
os.makedirs(os.path.dirname(UCF_EMB_CACHE), exist_ok=True)

def make_loader(root, shuffle=False):
    ds = UCF101Clips(root, class_to_idx=c2i)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=pin_memory)

train_loader = make_loader(f"{cfg.BASE_ROOT}/train", shuffle=True)
val_loader   = make_loader(f"{cfg.BASE_ROOT}/val")
test_loader  = make_loader(f"{cfg.BASE_ROOT}/test")

if os.path.exists(UCF_EMB_CACHE):
    print("Loading cached embeddings...")
    cache = torch.load(UCF_EMB_CACHE, map_location="cpu")
    train_s, train_t, train_y = cache["train_s"], cache["train_t"], cache["train_y"]
    val_s,   val_t,   val_y   = cache["val_s"],   cache["val_t"],   cache["val_y"]
    test_s,  test_t,  test_y  = cache["test_s"],  cache["test_t"],  cache["test_y"]
else:
    print("Extracting embeddings (~5 min first time)...")
    train_t, train_y = extract_embeddings(teacher, train_loader, device)
    val_t,   val_y   = extract_embeddings(teacher, val_loader,   device)
    test_t,  test_y  = extract_embeddings(teacher, test_loader,  device)
    train_s, _       = extract_embeddings(student, train_loader, device)
    val_s,   _       = extract_embeddings(student, val_loader,   device)
    test_s,  _       = extract_embeddings(student, test_loader,  device)
    torch.save({
        "train_s": train_s, "train_t": train_t, "train_y": train_y,
        "val_s":   val_s,   "val_t":   val_t,   "val_y":   val_y,
        "test_s":  test_s,  "test_t":  test_t,  "test_y":  test_y,
    }, UCF_EMB_CACHE)

print(f"Train: {train_s.shape} | Val: {val_s.shape} | Test: {test_s.shape}")
```

---

## Section 6 — Stage 1: Continual Learning Comparison

**What:** Compare three CL strategies on the teacher as new classes arrive.
**Why:** Establish which replay strategy best prevents forgetting. This directly affects how good the representation is before we distil it into the student.
**Strategies to compare:** Naive (no replay) | EWC | SmartReplay (hard exemplars)

### 6a — Build the CL data splits

```python
# For CL we introduce TASK_1 then TASK_2 sequentially on top of base classes
# Each task is evaluated on ALL classes seen so far

from src.data.dataset import UCF101Clips

def make_task_loader(root, task_classes, shuffle=False):
    local_c2i = {c: global_c2i[c] for c in task_classes}
    ds = UCF101Clips(root, class_to_idx=local_c2i)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=pin_memory)

task1_train = make_task_loader(f"{cfg.TASK1_ROOT}/train", cfg.TASK_1, shuffle=True)
task1_val   = make_task_loader(f"{cfg.TASK1_ROOT}/val",   cfg.TASK_1)
task2_train = make_task_loader(f"{cfg.TASK2_ROOT}/train", cfg.TASK_2, shuffle=True)
task2_val   = make_task_loader(f"{cfg.TASK2_ROOT}/val",   cfg.TASK_2)
```

### 6b — Naive CL (no replay — catastrophic forgetting baseline)

```python
# Import naive training from src
from src.cl.naive import train_naive  # or use train_continual with kd=False

# Run Task 1 then Task 2 with no replay
# Track base-class accuracy after each task to measure forgetting
# Expected result: base accuracy collapses from ~88% to ~30-45%
```

### 6c — EWC CL

```python
from src.cl.ewc import EWC
# Compute Fisher information on base classes
# Run Task 1 then Task 2 with EWC penalty
# Expected result: moderate forgetting — ~55-65% on base after Task 2
```

### 6d — SmartReplay CL (best strategy — use this going forward)

```python
from src.cl.smart_replay import SmartReplayBuffer, train_with_smart_replay

# Build buffer from base class embeddings (hard exemplar selection)
cl_buffer = SmartReplayBuffer(max_per_class=15)
for cls_name in cfg.SELECTED_CLASSES:
    idx  = c2i[cls_name]
    mask = train_y == idx
    if mask.sum() > 0:
        cl_buffer.add_class(idx, train_s[mask], train_t[mask], strategy="hard")

# Task 1 — introduce ApplyEyeMakeup, ApplyLipstick, Archery
# Extract task1 embeddings then train with smart replay
# Expected: base accuracy stays ~75-82% after Task 1

# Task 2 — introduce BoxingPunchingBag, BoxingSpeedBag, BrushingTeeth
# Expected: base accuracy stays ~72-80% after Task 2
```

### 6e — CL Comparison Plot

```python
# Plot: base-class accuracy across tasks for all 3 strategies
# X-axis: Task (Base → Task1 → Task2)
# Y-axis: Accuracy on base classes
# Lines: Naive | EWC | SmartReplay
# This is Figure 1 of your results section
```

**Success criterion:** SmartReplay forgetting (BWT) < -12%. Naive forgetting > -35%.

---

## Section 7 — Stage 2: Knowledge Distillation (Student Training)

**What:** Train the MobileNetV3 student to match the teacher's embeddings using MSE loss + CE loss, with Reptile meta-learning for fast adaptation.
**Why:** The student is 5x smaller than the teacher. KD transfers the teacher's UCF101 knowledge into the student efficiently. Reptile meta-training means the student can adapt quickly to new tasks later.
**Source module:** `src/meta_learning/reptile.py` → `train_reptile`

```python
from src.meta_learning.reptile  import train_reptile
from src.meta_learning.sampler  import sample, limit_to_k_per_class

# If student checkpoint already exists, skip training
if os.path.exists(student_ckpt) and os.path.exists(head_ckpt):
    print("Student and head already trained — skipping KD stage.")
else:
    print("Training student via Reptile + KD...")

    # Phase A: Train EmbeddingHead on teacher embeddings using Reptile
    head = train_reptile(
        head            = head,
        s_embs          = train_s,
        t_embs          = train_t,
        labels          = train_y,
        val_s_embs      = val_s,
        val_labels      = val_y,
        num_classes     = num_classes_base,
        device          = device,
        reptile_epochs      = 25,
        episodes_per_epoch  = 20,
        k_support       = 3,
        k_query         = 1,
        inner_lr        = 0.0005,
        inner_steps     = 10,
        src_epsilon     = 0.20,
        ce_weight       = 1.0,
        mse_weight      = 0.5,
        reptile_patience = 5,
    )

    # Phase B: Fine-tune head with full dataset
    from src.kd.trainer import finetune_head
    head = finetune_head(
        head        = head,
        train_s     = train_s,
        train_t     = train_t,
        train_y     = train_y,
        val_s       = val_s,
        val_y       = val_y,
        device      = device,
        epochs      = 15,
        lr          = 0.0005,
        wd          = 0.03,
        batch_size  = 16,
        patience    = 8,
        ce_weight   = 1.0,
        mse_weight  = 0.5,
        label_smoothing = 0.10,
    )

    torch.save(head.state_dict(), head_ckpt)
    print(f"Head saved → {head_ckpt}")

# Evaluate: teacher vs student accuracy on test set
teacher_acc = (teacher_logits.argmax(1) == test_y).float().mean().item()  # from teacher forward pass
student_acc = eval_head(head, test_s, test_y, device)
print(f"Teacher accuracy: {teacher_acc:.2%}")
print(f"Student accuracy: {student_acc:.2%}")
print(f"KD gap: {(teacher_acc - student_acc)*100:.1f}pp")
```

**Success criterion:** Student accuracy within 5–10pp of teacher. Gap > 15pp indicates the student LSTM was randomly initialised — go back and fix checkpoint loading.

---

## Section 8 — Stage 3: Semantic Embeddings

**What:** Encode all class names (base + seen + unseen) into 384-dim text vectors using sentence-transformers.
**Why:** These vectors are the bridge between language (class names) and vision (LSTM features). The CVAE uses them to generate visual prototypes for classes it has never seen video of.
**Model:** `all-MiniLM-L6-v2` (fast, 384-dim). For better ZSL, switch to `all-mpnet-base-v2` (768-dim, see note below).

```python
st_model = SentenceTransformer("all-MiniLM-L6-v2")
# NOTE: For better ZSL accuracy, use "all-mpnet-base-v2" (768-dim).
# If you switch, update SEM_DIM = 768 and rebuild all GAN modules.

all_classes_for_sem = cfg.SELECTED_CLASSES + cfg.TASK_1 + cfg.TASK_2 + cfg.TASK_3
raw_embs = st_model.encode(all_classes_for_sem, convert_to_tensor=True, show_progress_bar=False)
sem_embs = {
    name: F.normalize(raw_embs[i].unsqueeze(0), dim=-1).squeeze(0).cpu()
    for i, name in enumerate(all_classes_for_sem)
}
print(f"Semantic embeddings: {len(sem_embs)} classes, dim={SEM_DIM}")
```

---

## Section 9 — Stage 4a: GIL-GAN Initialisation

**What:** Build the GIL replay buffer from base class prototypes. Train the CVAE (E), Feature Generator (F), Discriminator (G), and Projection Head (H).
**Why:** This is the generative backbone of the zero-shot system. F learns to synthesise visual features from class prototypes. After this section, F is frozen forever.
**IMPORTANT:** Train for 150 epochs (not 30). This is the single most impactful change for ZSL quality.

### 9a — Build GIL Replay Buffer

```python
def extract_class_features(model, root, class_list, c2i_map, device, num_workers=2):
    """Extract LSTM hidden states for all clips in class_list."""
    feat_store = {c: [] for c in class_list}
    local_c2i  = {c: c2i_map[c] for c in class_list if c in c2i_map}
    split_dirs = [d for d in ["train", "val", "test"] if os.path.isdir(os.path.join(root, d))]
    roots_to_scan = [os.path.join(root, s) for s in split_dirs] if split_dirs else [root]
    model.eval()
    with torch.no_grad():
        for split_root in roots_to_scan:
            ds = UCF101Clips(split_root, class_to_idx=local_c2i)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=num_workers)
            for clips, labels in tqdm(loader, desc=os.path.basename(split_root)):
                _, h = model(clips.to(device))
                for feat, lbl in zip(h.cpu(), labels.cpu()):
                    cls_name = label_to_class[lbl.item()]
                    if cls_name in feat_store:
                        feat_store[cls_name].append(feat)
    return {c: torch.stack(v) for c, v in feat_store.items() if v}

base_class_feats = extract_class_features(
    teacher, cfg.BASE_ROOT, cfg.SELECTED_CLASSES, global_c2i, device, NUM_WORKERS
)

gil_buffer = GILReplayBuffer()
for cls_name, feats in base_class_feats.items():
    gil_buffer.add(
        class_name = cls_name,
        mu         = feats.mean(0),
        sigma      = feats.std(0).clamp_min(1e-6),
        sem_emb    = sem_embs[cls_name],
        label      = global_c2i[cls_name],
    )
print(f"GIL buffer: {len(gil_buffer)} base classes")
```

### 9b — Train CVAE (E)

```python
E = CVAE(sem_dim=SEM_DIM, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM).to(device)
optimizer_E = torch.optim.Adam(E.parameters(), lr=1e-4)

# Train for CVAE_INIT_EPOCHS (100 — more than notebook 09's 50)
E.train()
for ep in range(1, CVAE_INIT_EPOCHS + 1):
    ep_loss = 0.0
    for cls_name, entry in gil_buffer.items():
        sem_b   = entry["sem"].unsqueeze(0).to(device)
        mu_b    = entry["mu"].unsqueeze(0).to(device)
        sigma_b = entry["sigma"].unsqueeze(0).to(device)
        mu_hat, sigma_hat, mu_z, log_var = E(sem_b)
        loss = CVAE.loss(mu_hat, sigma_hat, mu_b, sigma_b, mu_z, log_var)
        optimizer_E.zero_grad(); loss.backward(); optimizer_E.step()
        ep_loss += loss.item()
    if ep % 20 == 0:
        print(f"[CVAE Init] Epoch {ep:03d}/{CVAE_INIT_EPOCHS}  Loss: {ep_loss/len(gil_buffer):.6f}")

torch.save(E.state_dict(), f"{CKPT_DIR}/cvae_init.pth")
```

### 9c — Train GAN (F, G, H) — 150 epochs

```python
F_gen  = FeatureGenerator(feat_dim=FEAT_DIM, noise_dim=NOISE_DIM).to(device)
G_dis  = Discriminator(feat_dim=FEAT_DIM, sem_dim=SEM_DIM).to(device)
H_proj = ProjectionHead(feat_dim=FEAT_DIM, sem_dim=SEM_DIM).to(device)

# IMPORTANT: GAN_EPOCHS = 150 (not 30 as in the study notebook)
# This is the most important hyperparameter for ZSL quality.
# The study notebook used 30 epochs which was too few for 21M-parameter networks.
gan_history = train_gan(
    gen        = F_gen,
    G          = G_dis,
    H          = H_proj,
    buffer     = gil_buffer,
    class_to_label = global_c2i,
    device     = device,
    epochs     = GAN_EPOCHS,      # 150
    lam1       = 0.01,
    lam2       = 0.1,
    alpha      = 10.0,
    lr         = 1e-4,
    batch_size = 64,
    n_critic   = 5,
)

torch.save(F_gen.state_dict(),  f"{CKPT_DIR}/generator.pth")
torch.save(G_dis.state_dict(),  f"{CKPT_DIR}/discriminator.pth")
torch.save(H_proj.state_dict(), f"{CKPT_DIR}/projection_head.pth")
print("GAN checkpoints saved. F is now frozen permanently.")
```

---

## Section 10 — Stage 4b: GIL Incremental Learning

**What:** Introduce seen classes (TASK_1, TASK_2) one at a time, mixing real new features with synthetic old features. Fine-tune only the classifier head — backbone stays frozen.
**Why:** This is the anti-forgetting mechanism. Without synthetic replay, the classifier forgets base classes as it learns new ones.

```python
seen_classes      = cfg.TASK_1 + cfg.TASK_2
all_known_classes = cfg.SELECTED_CLASSES + seen_classes
num_known         = len(all_known_classes)
known_label_local = {c: i for i, c in enumerate(all_known_classes)}

# Pre-extract seen class features
task1_feats = extract_class_features(teacher, cfg.TASK1_ROOT, cfg.TASK_1, global_c2i, device)
task2_feats = extract_class_features(teacher, cfg.TASK2_ROOT, cfg.TASK_2, global_c2i, device)
all_seen_feats = {**task1_feats, **task2_feats}

classifier = GILClassifier(feat_dim=FEAT_DIM, num_classes=num_known).to(device)
optimizer_E_incr = torch.optim.Adam(E.parameters(), lr=1e-4)
acc_per_iteration = []

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

chunk_size = max(1, len(seen_classes) // 10)

for t, class_chunk in enumerate(chunks(seen_classes, chunk_size)):
    print(f"\nIteration {t+1}: {class_chunk}")

    real_feats  = torch.cat([all_seen_feats[c] for c in class_chunk if c in all_seen_feats])
    real_labels = torch.cat([
        torch.full((all_seen_feats[c].shape[0],), known_label_local[c], dtype=torch.long)
        for c in class_chunk if c in all_seen_feats
    ])

    J_iter = max(1, len(real_feats) // max(len(gil_buffer), 1))
    synth_f, synth_l_global = gil_buffer.generate_all(F_gen, J=J_iter, device=device)

    if len(synth_f) > 0:
        synth_labels_local = torch.tensor(
            [known_label_local[label_to_class[lb.item()]] for lb in synth_l_global],
            dtype=torch.long,
        )
        mixed_feats  = torch.cat([real_feats, synth_f])
        mixed_labels = torch.cat([real_labels, synth_labels_local])
    else:
        mixed_feats, mixed_labels = real_feats, real_labels

    mixed_loader = DataLoader(
        TensorDataset(mixed_feats, mixed_labels), batch_size=64, shuffle=True
    )
    classifier = fine_tune_last2(classifier, mixed_loader, epochs=5, lr=1e-4, device=device)

    update_buffer_and_cvae(
        model          = teacher,
        E              = E,
        new_classes    = class_chunk,
        buffer         = gil_buffer,
        class_feats    = all_seen_feats,
        sem_embs       = sem_embs,
        class_to_label = global_c2i,
        optimizer_E    = optimizer_E_incr,
        device         = device,
        cvae_epochs    = CVAE_UPDATE_EPOCHS,   # 20 (was 5)
    )

    # Anti-forgetting check: accuracy on base classes
    classifier.eval()
    val_feats  = torch.cat(list(base_class_feats.values()))
    val_labels = torch.cat([
        torch.full((base_class_feats[c].shape[0],), known_label_local[c], dtype=torch.long)
        for c in cfg.SELECTED_CLASSES if c in base_class_feats
    ])
    with torch.no_grad():
        logits  = classifier(val_feats.to(device))
        val_acc = (logits.argmax(1).cpu() == val_labels).float().mean().item()
    acc_per_iteration.append(val_acc)
    print(f"  Buffer size: {len(gil_buffer)} | Base-class acc: {val_acc:.2%}")

torch.save(classifier.state_dict(),    f"{CKPT_DIR}/classifier_incremental.pth")
torch.save(E.state_dict(),             f"{CKPT_DIR}/cvae_incremental.pth")
print("Incremental learning complete.")
```

**Success criterion:** Base-class accuracy should rise or stay stable across iterations (as seen in notebook 09, it went from 27% → 94%). If it stays flat or drops, the synthetic replay from F is too noisy — try more GAN training epochs.

---

## Section 11 — Stage 4c: Meta-Learning on CVAE (NEW — ZSL Improvement)

**What:** Apply Reptile meta-training to the CVAE so it learns *how to quickly map text → visual prototype* rather than just memorising a fixed mapping.
**Why:** The standard CVAE (trained in Section 9b) minimises reconstruction loss on seen classes. When asked about an unseen class like "Skiing", it generalises poorly because it learned a fixed mapping, not a generalisation strategy. Reptile meta-training fixes this by training the CVAE across many episodes where some classes are hidden — forcing the CVAE to develop a general text→vision mapping strategy.
**Expected improvement:** ZSL Top-1 from ~20% to ~30–38%.

```python
# ── Reptile meta-training for CVAE ──────────────────────────────────────────
# Load the CVAE from the incremental stage
E_meta = copy.deepcopy(E)
E_meta.to(device)

# Helper: snap and update (same pattern as src/meta/reptile.py)
def snap(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}

def reptile_update_model(model, W0, eps):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in W0:
                p.copy_(W0[n] + eps * (p - W0[n]))

# Build list of all seen classes in the buffer
seen_cls_list = gil_buffer.class_list  # all base + seen classes added so far
print(f"Meta-training CVAE over {len(seen_cls_list)} seen classes.")
print(f"Episodes: {META_EPISODES} | Inner steps: {META_INNER_STEPS} | ε={META_EPSILON}")

meta_losses = []

for ep in range(1, META_EPISODES + 1):
    # Sample a support set of N_WAY classes from the buffer (hide some as query)
    N_WAY  = min(5, len(seen_cls_list))
    support = random.sample(seen_cls_list, N_WAY)
    query   = [c for c in seen_cls_list if c not in support]

    # Build support tensors
    sup_sem   = torch.stack([sem_embs[c]                for c in support]).to(device)
    sup_mu    = torch.stack([gil_buffer._store[c]["mu"]    for c in support]).to(device)
    sup_sigma = torch.stack([gil_buffer._store[c]["sigma"] for c in support]).to(device)

    W0       = snap(E_meta)
    inner_opt = optim.Adam(E_meta.parameters(), lr=META_INNER_LR)

    E_meta.train()
    for _ in range(META_INNER_STEPS):
        inner_opt.zero_grad(set_to_none=True)
        mu_hat, sigma_hat, mu_z, log_var = E_meta(sup_sem)
        loss = CVAE.loss(mu_hat, sigma_hat, sup_mu, sup_sigma, mu_z, log_var)
        loss.backward()
        inner_opt.step()

    reptile_update_model(E_meta, W0, META_EPSILON)

    # Evaluate generalisation on query classes (unseen in this episode)
    if query:
        E_meta.eval()
        with torch.no_grad():
            qry_sem   = torch.stack([sem_embs[c]                for c in query]).to(device)
            qry_mu    = torch.stack([gil_buffer._store[c]["mu"]    for c in query]).to(device)
            qry_sigma = torch.stack([gil_buffer._store[c]["sigma"] for c in query]).to(device)
            mu_hat_q, sigma_hat_q, mu_z_q, lv_q = E_meta(qry_sem)
            query_loss = CVAE.loss(mu_hat_q, sigma_hat_q, qry_mu, qry_sigma, mu_z_q, lv_q)
        meta_losses.append(query_loss.item())

    if ep % 10 == 0:
        avg = sum(meta_losses[-10:]) / max(len(meta_losses[-10:]), 1)
        print(f"[Meta-CVAE] Episode {ep:3d}/{META_EPISODES} | Query loss: {avg:.4f}")

torch.save(E_meta.state_dict(), f"{CKPT_DIR}/cvae_meta.pth")
print(f"Meta-trained CVAE saved → {CKPT_DIR}/cvae_meta.pth")

# Plot meta-training query loss (should decrease — proves generalisation)
plt.figure(figsize=(8, 3))
plt.plot(meta_losses, color="#2563EB")
plt.title("CVAE Meta-Training: Query Loss\n(decreasing = better generalisation to unseen classes)")
plt.xlabel("Episode"); plt.ylabel("Query Loss"); plt.tight_layout(); plt.show()
```

**Success criterion:** Query loss should decrease over episodes. If it stays flat, increase `META_EPISODES` to 100 or lower `META_INNER_LR` to 5e-5.

---

## Section 12 — Stage 4d: Zero-Shot Evaluation

**What:** Evaluate zero-shot recognition on TASK_3 (10 unseen classes — no training videos used).
**Why:** This tests whether the meta-learned CVAE generalises to truly novel classes. Compare standard CVAE vs meta-CVAE to show the improvement.

```python
# Build TASK_3 test loader
unseen_classes = cfg.TASK_3
unseen_c2i     = {c: global_c2i[c] for c in unseen_classes}
task3_dataset  = UCF101Clips(f"{cfg.TASK3_ROOT}/test", class_to_idx=unseen_c2i)
task3_loader   = DataLoader(task3_dataset, batch_size=8, shuffle=False,
                             num_workers=NUM_WORKERS)
print(f"TASK_3 test set: {len(task3_dataset)} clips, {len(unseen_classes)} classes")

# ── ZSL with standard CVAE (baseline) ───────────────────────────────────────
E.eval()
zsl_acc_standard, _, _ = zero_shot_test(
    model          = teacher,
    gen            = F_gen,
    E              = E,
    unseen_classes = unseen_classes,
    sem_embs       = sem_embs,
    class_to_label = global_c2i,
    test_loader    = task3_loader,
    device         = device,
    J              = J_SYNTH,    # 200 (was 50)
)

# ── ZSL with meta-trained CVAE (improved) ───────────────────────────────────
E_meta.eval()
zsl_acc_meta, _, _ = zero_shot_test(
    model          = teacher,
    gen            = F_gen,
    E              = E_meta,     # ← meta-trained CVAE
    unseen_classes = unseen_classes,
    sem_embs       = sem_embs,
    class_to_label = global_c2i,
    test_loader    = task3_loader,
    device         = device,
    J              = J_SYNTH,
)

print(f"\nZSL Results:")
print(f"  Standard CVAE  : {zsl_acc_standard:.2%}")
print(f"  Meta CVAE      : {zsl_acc_meta:.2%}")
print(f"  Improvement    : +{(zsl_acc_meta - zsl_acc_standard)*100:.1f}pp")
```

### GZSL Harmonic Mean

```python
# Seen accuracy from the incremental classifier
classifier.eval()
all_seen_test_feats = torch.cat(
    [base_class_feats[c] for c in cfg.SELECTED_CLASSES if c in base_class_feats] +
    [all_seen_feats[c]   for c in seen_classes         if c in all_seen_feats],
    dim=0,
)
all_seen_test_labels = torch.cat(
    [torch.full((base_class_feats[c].shape[0],), known_label_local[c], dtype=torch.long)
     for c in cfg.SELECTED_CLASSES if c in base_class_feats] +
    [torch.full((all_seen_feats[c].shape[0],), known_label_local[c], dtype=torch.long)
     for c in seen_classes if c in all_seen_feats],
    dim=0,
)
with torch.no_grad():
    seen_acc = (classifier(all_seen_test_feats.to(device)).argmax(1).cpu() == all_seen_test_labels).float().mean().item()

for label, zsl_acc in [("Standard CVAE", zsl_acc_standard), ("Meta CVAE", zsl_acc_meta)]:
    h_mean = 2 * seen_acc * zsl_acc / (seen_acc + zsl_acc + 1e-8)
    print(f"  {label:<18} | Seen: {seen_acc:.2%} | Unseen: {zsl_acc:.2%} | H-mean: {h_mean:.2%}")
```

**Expected results:**
| Method | ZSL Top-1 | GZSL H-mean |
|---|---|---|
| Standard CVAE (30 epochs GAN) | ~16% | ~27% |
| Standard CVAE (150 epochs GAN) | ~22–28% | ~32–38% |
| Meta-CVAE (150 epochs GAN) | ~30–40% | ~38–48% |

---

## Section 13 — Stage 5: Active Domain Adaptation

**What:** Adapt the student model to YouTube-style videos of the same classes using active learning to select the most informative clips.
**Why:** UCF101 clips are clean and curated. Real-world videos (YouTube) have different camera angles, backgrounds, compression. AL selects the YouTube clips where the model is most uncertain — these carry the most information for adaptation.
**Source modules:** `src/active/acquisition.select_top_k`, `src/active/domain_shift.reptile_adapt`

### 13a — Download & Preprocess YouTube Clips

```python
import subprocess, cv2
from torchvision import transforms
from PIL import Image

yt_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CLIP_LEN = 16; CLIPS_PER_VID = 8

def extract_clips(video_path, clip_len=16, n_clips=8):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clips = []
    if total < clip_len:
        cap.release(); return clips
    starts = np.linspace(0, max(total - clip_len, 0), n_clips, dtype=int)
    for start in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(yt_transform(Image.fromarray(frame)))
        if len(frames) == clip_len:
            clips.append(torch.stack(frames))
    cap.release(); return clips

os.makedirs("/content/yt_raw", exist_ok=True)
yt_clips = {}
for cls_name, url in YOUTUBE_CLIPS.items():
    if cls_name not in c2i:
        print(f"[!] {cls_name} not in base classes — skipping"); continue
    out_path = f"/content/yt_raw/{cls_name}.mp4"
    if not os.path.exists(out_path):
        subprocess.run(["yt-dlp", "--format", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]",
                        "--merge-output-format", "mp4", "--output", out_path, "--quiet", url])
    clips = extract_clips(out_path, CLIP_LEN, CLIPS_PER_VID)
    yt_clips[cls_name] = clips
    print(f"  {cls_name}: {len(clips)} clips")
```

### 13b — Extract YouTube Embeddings

```python
teacher.eval(); student.eval()
yt_s_dict = {}; yt_t_dict = {}

with torch.no_grad():
    for cls_name, clips in yt_clips.items():
        if not clips: continue
        batch = torch.stack(clips).to(device)
        _, t_emb = teacher(batch)
        _, s_emb = student(batch)
        yt_s_dict[cls_name] = s_emb.cpu()
        yt_t_dict[cls_name] = t_emb.cpu()

yt_s_all = torch.cat(list(yt_s_dict.values()))
yt_t_all = torch.cat(list(yt_t_dict.values()))
yt_y_all = torch.cat([
    torch.full((yt_s_dict[c].shape[0],), c2i[c], dtype=torch.long)
    for c in yt_s_dict
])
```

### 13c — Domain Shift Analysis

```python
yt_class_idxs = [c2i[c] for c in yt_s_dict]
mask_ucf      = torch.isin(test_y, torch.tensor(yt_class_idxs))
ucf_s_sub     = test_s[mask_ucf]; ucf_y_sub = test_y[mask_ucf]

ucf_stats = compute_class_stats(ucf_s_sub, ucf_y_sub)
yt_stats  = compute_class_stats(yt_s_all, yt_y_all)
shifts    = measure_shift(ucf_stats, yt_stats, class_names=cfg.SELECTED_CLASSES)
# Print per-class domain gap (L2 distance between UCF101 and YouTube prototypes)
```

### 13d — Active Learning Acquisition

```python
# Score YouTube clips by uncertainty using the trained head
head_for_scoring = copy.deepcopy(head)
head_for_scoring.eval()

# select_top_k picks the K most informative clips per class
al_selected = select_top_k(
    embs_by_class = yt_s_dict,
    strategy      = AL_STRATEGY,    # "entropy" by default
    k             = AL_BUDGET_PER_CLASS,
    head          = head_for_scoring,
    device        = device,
)

al_s = torch.cat([yt_s_dict[c][al_selected[c]] for c in al_selected])
al_t = torch.cat([yt_t_dict[c][al_selected[c]] for c in al_selected])
al_y = torch.cat([torch.full((len(al_selected[c]),), c2i[c], dtype=torch.long)
                  for c in al_selected])
print(f"AL selected: {al_s.shape[0]} clips  strategy={AL_STRATEGY}")
```

### 13e — Smart Replay Buffer (UCF101 + YouTube AL clips)

```python
# Hybrid buffer: real UCF101 exemplars for seen classes + AL YouTube clips
# This prevents forgetting UCF101 while adapting to YouTube
adapt_buffer = SmartReplayBuffer(max_per_class=10)

for cls_name in cfg.SELECTED_CLASSES:
    idx  = c2i[cls_name]; mask = train_y == idx
    if mask.sum() > 0:
        adapt_buffer.add_class(idx, train_s[mask], train_t[mask], strategy="hard")

for cls_name in al_selected:
    idx   = c2i[cls_name]
    s_yt  = yt_s_dict[cls_name][al_selected[cls_name]]
    t_yt  = yt_t_dict[cls_name][al_selected[cls_name]]
    if idx in adapt_buffer.s_store:
        s_combined = torch.cat([adapt_buffer.s_store[idx], s_yt])
        t_combined = torch.cat([adapt_buffer.t_store[idx], t_yt])
    else:
        s_combined, t_combined = s_yt, t_yt
    adapt_buffer.add_class(idx, s_combined, t_combined, strategy="diverse")

adapt_buffer.summary(class_names=cfg.SELECTED_CLASSES)
```

### 13f — Reptile Adaptation

```python
head_adapted = copy.deepcopy(head)

# Baseline accuracy before adaptation
ucf_acc_before = eval_head(head,         test_s,   test_y,   device)
yt_acc_before  = eval_head(head,         yt_s_all, yt_y_all, device)
print(f"Before — UCF101: {ucf_acc_before:.2%} | YouTube: {yt_acc_before:.2%}")

head_adapted = reptile_adapt(
    head          = head_adapted,
    al_s          = al_s,
    al_t          = al_t,
    al_y          = al_y,
    buffer        = adapt_buffer,
    device        = device,
    episodes      = ADAPT_EPISODES,
    inner_lr      = ADAPT_INNER_LR,
    inner_steps   = ADAPT_INNER_STEPS,
    epsilon       = ADAPT_EPSILON,
    mse_weight    = 0.5,
    label_smoothing = 0.1,
    eval_s        = yt_s_all,
    eval_y        = yt_y_all,
    log_every     = 10,
)

ucf_acc_after = eval_head(head_adapted, test_s,   test_y,   device)
yt_acc_after  = eval_head(head_adapted, yt_s_all, yt_y_all, device)
print(f"After  — UCF101: {ucf_acc_after:.2%} | YouTube: {yt_acc_after:.2%}")
print(f"YouTube gain: +{(yt_acc_after - yt_acc_before)*100:.1f}pp | UCF101 change: {(ucf_acc_after - ucf_acc_before)*100:+.1f}pp")

torch.save(head_adapted.state_dict(), f"{CKPT_DIR}/head_adapted_yt.pt")
```

**Success criterion:** YouTube accuracy gain > +15pp. UCF101 drop < -5pp. If UCF101 drops more than 5pp, increase `max_per_class` in the SmartReplayBuffer.

---

## Section 14 — Unified Evaluation & Results Table

**What:** Compile all results into a single comparison table across all stages and methods.
**Why:** This is the thesis results section. Every claim needs a number.

```python
# ── Final results table ──────────────────────────────────────────────────────
W = 72
print("=" * W)
print(f"  {'Stage':<30} | {'Method':<20} | {'Metric':<10} | {'Value':>7}")
print("-" * W)

# Stage 1: CL
print(f"  {'CL — Base after Task2':<30} | {'Naive':<20} | {'Acc':<10} | {naive_final_acc:>6.2%}")
print(f"  {'CL — Base after Task2':<30} | {'EWC':<20} | {'Acc':<10} | {ewc_final_acc:>6.2%}")
print(f"  {'CL — Base after Task2':<30} | {'SmartReplay':<20} | {'Acc':<10} | {smart_final_acc:>6.2%}")

# Stage 2: KD
print(f"  {'KD':<30} | {'Teacher':<20} | {'Test Acc':<10} | {teacher_acc:>6.2%}")
print(f"  {'KD':<30} | {'Student':<20} | {'Test Acc':<10} | {student_acc:>6.2%}")

# Stage 3: ZSL
print(f"  {'ZSL TASK_3':<30} | {'Standard CVAE':<20} | {'Top-1':<10} | {zsl_acc_standard:>6.2%}")
print(f"  {'ZSL TASK_3':<30} | {'Meta-CVAE':<20} | {'Top-1':<10} | {zsl_acc_meta:>6.2%}")

# Stage 4: Domain Adaptation
print(f"  {'Domain Adapt':<30} | {'Before':<20} | {'YouTube':<10} | {yt_acc_before:>6.2%}")
print(f"  {'Domain Adapt':<30} | {'After':<20} | {'YouTube':<10} | {yt_acc_after:>6.2%}")
print("=" * W)

# Save results to JSON
results = dict(
    naive_cl_acc   = naive_final_acc,
    ewc_cl_acc     = ewc_final_acc,
    smart_cl_acc   = smart_final_acc,
    teacher_acc    = teacher_acc,
    student_acc    = student_acc,
    zsl_standard   = zsl_acc_standard,
    zsl_meta       = zsl_acc_meta,
    yt_before      = yt_acc_before,
    yt_after       = yt_acc_after,
    yt_gain        = yt_acc_after - yt_acc_before,
    ucf_drop       = ucf_acc_after - ucf_acc_before,
)
with open(f"{RESULTS_DIR}/ulal_final_results.json", "w") as f:
    json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
print(f"Results saved → {RESULTS_DIR}/ulal_final_results.json")
```

---

## Section 15 — Visualisations

Produce all plots for the thesis. Save each to RESULTS_DIR.

### Plot 1: CL Forgetting Curves
- X: Task (Base → Task1 → Task2)
- Y: Base-class accuracy
- Lines: Naive | EWC | SmartReplay
- Title: "Catastrophic Forgetting vs Replay Strategy"

### Plot 2: Teacher vs Student Accuracy per Class
- Bar chart, one bar per class
- Two bars per class: Teacher (blue) | Student (orange)
- Title: "Knowledge Distillation — Per-Class Accuracy"

### Plot 3: ZSL Comparison
- Bar chart: Standard CVAE | Meta-CVAE
- Metrics: ZSL Top-1 | GZSL H-mean
- Title: "Zero-Shot Recognition — Standard vs Meta-Learned CVAE"

### Plot 4: Meta-CVAE Training Curve
- X: Episode | Y: Query loss
- Title: "CVAE Meta-Training Convergence"

### Plot 5: Domain Adaptation Before/After
- Two bars per class (YouTube): Before | After adaptation
- Title: "Active Domain Adaptation — Per-Class YouTube Accuracy"

### Plot 6: t-SNE Embedding Space
- Real UCF101 vs YouTube embeddings
- Before adaptation (left) | After adaptation (right)
- Title: "Embedding Space: Domain Shift and Adaptation"

---

## Section 16 — Save Everything

```python
# Save all final model states and buffer
torch.save(head_adapted.state_dict(),      f"{CKPT_DIR}/head_adapted_yt.pt")
torch.save(E_meta.state_dict(),            f"{CKPT_DIR}/cvae_meta.pth")
torch.save(classifier.state_dict(),        f"{CKPT_DIR}/classifier_incremental.pth")
torch.save(F_gen.state_dict(),             f"{CKPT_DIR}/generator.pth")
torch.save({"s_store": adapt_buffer.s_store,
            "t_store": adapt_buffer.t_store,
            "mu_store": adapt_buffer.mu_store},
           f"{CKPT_DIR}/adapt_replay_buffer.pt")

print("All outputs saved.")
print(f"  Checkpoints : {CKPT_DIR}/")
print(f"  Results     : {RESULTS_DIR}/")
```

---

## Summary of What Each Section Proves

| Section | Stage | Research Question Answered |
|---|---|---|
| 3 | Data | Data pipeline is reproducible |
| 4 | Models | Pre-trained representations are strong |
| 5 | Embeddings | Features are cached and consistent |
| 6 | CL | Can the model learn sequentially without forgetting? |
| 7 | KD | Can knowledge be compressed into a smaller model? |
| 8 | Semantics | Can class names be mapped to visual feature space? |
| 9 | GIL Init | Can a GAN generate realistic visual features? |
| 10 | GIL Incr | Does synthetic replay prevent forgetting? |
| 11 | Meta-CVAE | Does meta-learning improve unseen class generalisation? |
| 12 | ZSL | Can the system recognise classes never seen in training? |
| 13 | Domain Adapt | Can the system adapt to a new video distribution? |
| 14 | Evaluation | What are the final numbers across all stages? |

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: size mismatch` in head | Student LSTM hidden size mismatch | Check STUDENT_HIDDEN=128 matches checkpoint |
| `KeyError: cls_name` in CVAE meta-training | Class in sem_embs but not in gil_buffer | Only meta-train on classes in `gil_buffer.class_list` |
| ZSL accuracy = ~10% (random) | CVAE produces wrong prototypes | Increase CVAE_INIT_EPOCHS and GAN_EPOCHS |
| YouTube accuracy drops after adaptation | SmartReplayBuffer too small | Increase max_per_class to 20 |
| GAN generator loss not decreasing | Learning rate too high | Reduce GAN lr from 1e-4 to 5e-5 |
| TSNE plot shows no class separation | Student embeddings are random | Student checkpoint was not loaded — fix Section 4 |
| `FileNotFoundError: student_ucf101.pt` | KD notebook never run | Run notebook 06 first, or train student inline in Section 7 |
