# Unified Lifelong Action Learning (ULAL)

> **Continual Learning for Video-Based Action Recognition on UCF101**

---

## Overview

This project implements a full **Continual Learning (CL)** pipeline for video-based action recognition using the [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) dataset.

A pretrained **ResNet-50** backbone is first fine-tuned on a base set of 10 action classes via transfer learning, then incrementally trained across **4 additional tasks** of 10 new classes each — simulating a realistic lifelong learning scenario where a model must adapt to new data without forgetting what it previously learned.

Five continual learning strategies are implemented and compared. **Active Learning** is integrated to improve rehearsal buffer quality. The best-performing CL model serves as a **teacher** in a final Knowledge Distillation step that trains a compact **ResNet-18 student**.

---

## Pipeline Summary

```
UCF101 Dataset (50 classes)
        │
        ▼
  Task Split Design
  ┌─────────────────────────────────────┐
  │  Task 0 (base): 10 classes          │
  │  Task 1–4 (incremental): 10 each    │
  └─────────────────────────────────────┘
        │
        ▼
  ResNet-50 Pretrained Backbone
  ┌─────────────────────────────────────┐
  │  Transfer Learning on Task 0        │
  │  Frozen / Partial / Full Fine-Tune  │
  └─────────────────────────────────────┘
        │
        ▼
  Continual Learning (Tasks 1–4)
  ┌─────────────────────────────────────┐
  │  1. Naive Fine-Tuning (baseline)    │
  │  2. EWC                             │
  │  3. Rehearsal + Active Learning     │
  │  4. LwF (Knowledge Distillation CL) │
  │  5. La-MAML                         │
  └─────────────────────────────────────┘
        │
        ▼
  Evaluation Protocol
  ┌─────────────────────────────────────┐
  │  Average Accuracy                   │
  │  Forgetting Measure                 │
  │  Forward Transfer                   │
  └─────────────────────────────────────┘
        │
        ▼
  Best Model Selected as Teacher
        │
        ▼
  Final Knowledge Distillation
  ┌─────────────────────────────────────┐
  │  Teacher: ResNet-50 (best CL model) │
  │  Student: ResNet-18                 │
  │  Loss: CE + KL Divergence           │
  └─────────────────────────────────────┘
```

---

## Continual Learning Methods

| Method | Strategy | Forgetting Mitigation |
|---|---|---|
| **Naive** | Sequential fine-tuning | None (lower-bound baseline) |
| **EWC** | Regularization | Fisher Information penalty |
| **Rehearsal** | Replay | Store + replay past exemplars |
| **LwF** | Knowledge Distillation | Soft targets from frozen teacher |
| **La-MAML** | Meta-Learning | Fast adaptation via inner/outer loops |

---

## Active Learning Integration

Active Learning is used to improve **replay buffer selection** in the Rehearsal method, replacing random exemplar selection with **uncertainty-based** and **diversity-based** strategies to maximize information retained per memory slot.

---

## Evaluation Metrics

- **Average Accuracy** — mean accuracy across all seen tasks
- **Final Average Accuracy** — accuracy after the last task
- **Forgetting Measure** — drop in accuracy on old tasks
- **Backward Transfer** — how new tasks affect old-task performance
- **Forward Transfer** — how past learning influences new tasks
- **Training Time & Memory** — efficiency comparison

---

## Project Structure

```
Unified-Lifelong-Action-Learning/
├── notebooks/
│   ├── APAI_cnn_backbone.ipynb       # Phase 3: Backbone + Transfer Learning
│   └── APAI_NAIVE_CL.ipynb           # Phase 5.1: Naive CL baseline
├── src/
│   ├── config/
│   │   └── config.py                 # Classes, task splits, transforms, seeds
│   ├── data/
│   │   ├── dataset.py                # UCF101 clip dataset class
│   │   ├── preprocessing.py          # Frame extraction and augmentation
│   │   └── study_dataset.py          # Dataset exploration utilities
│   ├── models/
│   │   ├── pretrained.py             # ResNet-50 with expandable head
│   │   └── baselines.py              # Baseline model variants
│   ├── fine_tune/
│   │   ├── trainer.py                # Generic training + validation loop
│   │   └── visualizer.py             # Loss/accuracy plotting utilities
│   ├── cl_strategies/
│   │   ├── naive.py                  # Naive sequential fine-tuning
│   │   ├── ewc.py                    # (planned) EWC regularization
│   │   ├── rehearsal.py              # (planned) Experience replay
│   │   ├── lwf.py                    # (planned) Learning without Forgetting
│   │   └── la_maml.py                # (planned) La-MAML meta-learning
│   ├── meta_learning/                # (planned) Meta-learning utilities
│   ├── transfer_learning/            # (planned) Transfer learning wrappers
│   ├── utils/                        # (planned) Shared utilities
│   └── imports.py                    # Common imports
├── bash/                             # Setup and utility shell scripts
├── tasks.md                          # Project task tracker
└── README.md
```

---

## Task Split

| Task | Classes | Role |
|---|---|---|
| Task 0 | 10 classes | Base training (transfer learning) |
| Task 1 | 10 classes | Incremental task 1 |
| Task 2 | 10 classes | Incremental task 2 |
| Task 3 | 10 classes | Incremental task 3 |
| Task 4 | 10 classes | Incremental task 4 |

All class lists and splits are defined in `src/config/config.py`. Random seed is fixed to **42** for reproducibility.

---

## Setup

### Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- torchvision
- OpenCV
- NumPy, pandas, matplotlib

### Dataset

The UCF101 dataset is sourced via [Kaggle](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition).  
Paths are configured in `src/config/config.py` and auto-updated by `bash/setup_colab.sh` for Colab runs.

---

## Configuration

All key hyperparameters live in `src/config/config.py`:

| Parameter | Value |
|---|---|
| Clip length | 16 frames |
| Frame resize | 256 × 256 → crop 224 × 224 |
| Normalization | ImageNet mean/std |
| Batch size | 4 |
| Train split | 80% |
| Random seed | 42 |

---

## Progress

See [`tasks.md`](tasks.md) for the full task tracker with current status.

---

## License
Academic project — UCF101 dataset usage subject to its original license.