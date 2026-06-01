# Unified Lifelong Action Learning (ULAL)

> **Continual Learning + Active Domain Adaptation for Video-Based Action Recognition on UCF101**

---

## Overview

A full **Continual Learning (CL)** pipeline for video-based action recognition using [UCF101](https://www.crcv.ucf.edu/data/UCF101.php).

A pretrained **ResNet-50+LSTM teacher** is fine-tuned on 10 base classes, then incrementally trained across 4 additional tasks of 10 new classes each. Five CL strategies are compared. The best-performing model teaches a compact **MobileNetV3+LSTM student** via Knowledge Distillation. An **Active Domain Adaptation** module handles YouTube out-of-domain clips using AL-guided Reptile adaptation.

---

## Project Structure

```
ULAL/
├── notebooks/
│   ├── 01_setup_data.ipynb          # Dataset download, clip extraction, splits
│   ├── 02_backbone_transfer.ipynb   # ResNet50 Task 0 fine-tuning
│   ├── 03_naive_cl.ipynb            # Naive sequential fine-tuning (baseline)
│   ├── 04_ewc_cl.ipynb              # EWC regularisation
│   ├── 05_replay_lwf.ipynb          # Rehearsal buffer + LwF
│   ├── 06_reptile_kd.ipynb          # Reptile meta-learning + KD student
│   ├── 07_active_domain_adapt.ipynb # YouTube AL + Smart Replay + adaptation
│   ├── 08_evaluation.ipynb          # Full comparison tables + plots
│   └── archive/                     # Superseded experiments (ViT, early drafts)
│
├── src/
│   ├── config/
│   │   └── config.py                # All hyperparams, class splits, paths
│   ├── data/
│   │   ├── dataset.py               # UCF101Clips dataset class
│   │   ├── preprocessing.py         # Frame extraction and transforms
│   │   ├── study_dataset.py         # Dataset exploration utilities
│   │   └── youtube.py               # YouTube downloader + clip extractor
│   ├── models/
│   │   ├── teacher.py               # ResNet50LSTMTeacher
│   │   ├── student.py               # MobileNetV3SmallLSTMStudent
│   │   ├── head.py                  # EmbeddingHead (classifier + projector)
│   │   └── baselines.py             # Baseline model variants
│   ├── cl/
│   │   ├── naive.py                 # Naive sequential training loop
│   │   ├── ewc.py                   # EWC: Fisher estimation + penalised training
│   │   ├── rehearsal.py             # Replay buffer + LwF distillation loss
│   │   └── smart_replay.py          # SmartReplayBuffer (prototype + hard/diverse)
│   ├── meta/
│   │   ├── reptile.py               # Reptile outer loop with early stopping
│   │   └── sampler.py               # N-way K-shot episodic sampler
│   ├── kd/
│   │   ├── trainer.py               # finetune_head() — CE + MSE training loop
│   │   └── utils.py                 # extract_embeddings, eval_head, etc.
│   ├── active/
│   │   ├── acquisition.py           # Entropy, margin, coreset AL scoring
│   │   └── domain_shift.py          # Shift measurement + Reptile adaptation loop
│   └── utils/
│       ├── train.py                 # Generic training loop, evaluate_all_tasks
│       ├── visualize.py             # Training curves, confusion matrix plots
│       ├── metrics.py               # AA, Forgetting, BWT, FWT, cl_report
│       └── save.py                  # record() — multi-task result logging
│
├── bash/
│   ├── fetch_src.sh                 # Fetch /src from GitHub branch → Drive
│   └── setup_colab.sh               # Download UCF101, update config paths
│
├── tasks.md                         # Full task tracker
└── README.md
```

---

## Notebook Workflow

Each notebook is **self-contained** — it mounts Drive, fetches `src/` from GitHub, and saves outputs back to Drive. Run them in order on first use; afterwards any notebook can be run independently.

```
01_setup_data          → download UCF101, extract clips, verify splits
02_backbone_transfer   → fine-tune ResNet50 on Task 0, save checkpoint
03_naive_cl            → naive sequential baseline, measure forgetting
04_ewc_cl              → EWC regularisation, tune lambda
05_replay_lwf          → rehearsal + LwF, SmartReplayBuffer ablation
06_reptile_kd          → Reptile meta-learning, KD to MobileNetV3 student
07_active_domain_adapt → YouTube AL + domain adaptation
08_evaluation          → compare all methods, final tables + plots
```

---

## Pipeline

```
UCF101 (50 classes, 5 tasks × 10)
        │
        ▼
ResNet50+LSTM Teacher — Task 0 fine-tune
        │
        ├─── Continual Learning (Tasks 1–4)
        │        Naive / EWC / SmartReplay+LwF / Reptile
        │
        ├─── Active Domain Adaptation
        │        YouTube clips → AL acquisition → Reptile adaptation
        │
        └─── Knowledge Distillation
                 Teacher → MobileNetV3+LSTM Student
```

---

## CL Methods

| Method | Strategy | Key file |
|--------|----------|----------|
| Naive | Sequential fine-tuning (lower bound) | `src/cl/naive.py` |
| EWC | Fisher Information penalty | `src/cl/ewc.py` |
| Rehearsal + LwF | Smart replay + soft targets | `src/cl/rehearsal.py`, `smart_replay.py` |
| Reptile | Meta-learning fast adaptation | `src/meta/reptile.py` |

---

## Changing Branch in Colab

```python
import os
os.environ["BRANCH"] = "kd&mamal"   # ← any branch name
!bash /content/fetch_src.sh
```

Or edit the `BRANCH` variable at the top of `bash/fetch_src.sh`.

---

## Setup

**Requirements:** Python 3.9+, PyTorch ≥ 2.0, torchvision, OpenCV, yt-dlp (for YouTube clips)

**Dataset:** UCF101 via [Kaggle](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) — run `01_setup_data.ipynb` or `bash/setup_colab.sh`.

**Config:** All paths and hyperparameters live in `src/config/config.py`. Paths are auto-updated by `setup_colab.sh` for Colab runs.

---

## Cleanup Note

The following old folders still exist in the repo for git history — they are superseded by the new structure and can be removed:

- `src/meta_learning/` → replaced by `src/meta/` + `src/models/`
- `src/cl_strategies/` → replaced by `src/cl/`
- `src/fine_tune/`     → replaced by `src/utils/`
- `src/imports.py`     → removed (each notebook manages its own imports)

To clean up: `git rm -r src/meta_learning src/cl_strategies src/fine_tune src/imports.py`

---

## Progress

See [`tasks.md`](tasks.md) for the full task tracker.

---

## License

Academic project — UCF101 dataset usage subject to its original license.
