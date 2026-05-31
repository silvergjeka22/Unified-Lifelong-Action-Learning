# ULAL — Task Tracker
> Unified Lifelong Action Learning on UCF101

```
[x] done   [/] in progress   [ ] not started
```

---

## Notebook Index

| # | File | Purpose | Status |
|---|------|---------|--------|
| 01 | `APAI_cnn_backbone.ipynb`   | Backbone + Transfer Learning on Task 0 | [x] |
| 02 | `EWC_CL.ipynb`              | EWC continual learning experiments | [x] |
| 03 | `CL_Replay_buffer.ipynb`    | Rehearsal buffer + LwF experiments | [x] |
| 04 | `APAI_NAIVE_CL.ipynb`       | Naive sequential fine-tuning baseline | [/] |
| 05 | `kd_mse&mamal.ipynb`        | KD + Reptile meta-learning (working) | [x] |
| 06 | `06_active_domain_adapt.ipynb` | **YouTube AL + Smart Replay + Reptile adaptation** | [ ] |
| 07 | *(to create)* `07_evaluation.ipynb` | Full comparison table + plots | [ ] |

---

## Scripts

| File | Purpose |
|------|---------|
| `bash/fetch_src.sh`   | Fetch `/src` from any branch → local + Drive. **Change branch at top of file or set `BRANCH` env var in Colab.** |
| `bash/setup_colab.sh` | Download UCF101 from Kaggle, update `config.py` paths |

---

## Phase 1 — Dataset Preparation ✅

- [x] Download and organize UCF101 dataset
- [x] Verify video integrity and organize by class
- [x] Select 50 balanced classes, record class-to-index mapping
- [x] Build 5-task CL split (Task 0–4, 10 classes each) with seed 42
- [x] Extract 16-frame clips at 224×224 with ImageNet normalization
- [x] Create 80/20 train/val splits per class, save metadata

---

## Phase 2 — Dataloaders

- [x] `dataset.py` — UCF101 clip dataset class
- [x] `preprocessing.py` — frame extraction and transforms
- [x] `study_dataset.py` — dataset exploration utilities
- [ ] Task-incremental dataloader wrapper (feeds tasks sequentially)
- [ ] Replay buffer sampler (current task + replay mix)
- [ ] Distillation training dataloader
- [ ] End-to-end dataloader test

---

## Phase 3 — Backbone & Transfer Learning ✅

- [x] ResNet-50 with ImageNet weights, expandable head (`pretrained.py`)
- [x] Baseline variants (`baselines.py`)
- [x] Train Task 0: frozen / partial / full fine-tuning — best config selected
- [x] Task 0 checkpoint saved, loss/accuracy curves logged

---

## Phase 4 — Core Training Framework

- [x] Generic training loop + validation (`trainer.py`)
- [x] Checkpoint save/load, visualizer (`visualizer.py`)
- [ ] Incremental training pipeline (sequential tasks, dynamic head expansion)
- [ ] Per-task accuracy tracking + forgetting measure after each task
- [ ] Save all metrics to CSV / JSON

---

## Phase 5 — Continual Learning Methods

### 5.1 Naive Fine-Tuning (lower-bound baseline)
- [/] `naive.py` — sequential training, no forgetting mitigation
- [ ] Measure forgetting on all previous tasks after each new task

### 5.2 EWC — Elastic Weight Consolidation
- [x] Fisher Information estimation per task
- [x] EWC penalty: `loss = CE + λ * Σ F*(θ-θ*)²`
- [x] `pad_fisher_after_expand()` for head growth
- [x] `ewc_report()` + `check_ewc_penalty()` diagnostics
- [ ] Tune EWC lambda across tasks
- [ ] Validate forgetting reduction

### 5.3 Smart Prototype Replay Buffer
- [x] Basic `ReplayBuffer` FIFO (`reharsal.py`)
- [ ] Implement `SmartReplayBuffer` in `src/cl_strategies/smart_replay.py`:
  - [ ] Per-class prototype (μ, σ) stored instead of raw clips
  - [ ] Hard-example selection: keep samples farthest from μ
  - [ ] Uncertainty selection: keep highest-entropy samples
  - [ ] Diversity selection: coreset / greedy coverage
- [ ] Ablation: FIFO vs hard vs uncertainty vs diversity

### 5.4 LwF — Learning without Forgetting
- [x] `distillation_loss()` with KL divergence + temperature (`reharsal.py`)
- [x] `train_continual()` with `kd=True/False` flag
- [ ] Tune alpha (CE vs KD) and temperature T
- [ ] Compare CE-only vs CE+KD on old-task retention

### 5.5 Meta-Learning — Reptile / La-MAML
- [x] `ResNet50LSTMTeacher` + `MobileNetV3SmallLSTMStudent` (`models.py`)
- [x] `EmbeddingHead` with classifier + projector
- [x] N-way K-shot episodic sampler (`sampler.py`)
- [x] Reptile outer loop with early stopping (`reptile.py`)
- [x] Combined CE + MSE(proj, teacher_emb) loss
- [x] `finetune_head()` baseline (`mseLoss.py`)
- [ ] Wire Reptile into full sequential task stream (Tasks 0 → 4)
- [ ] Validate fast adaptation: test after K inner steps on new task
- [ ] Compare Reptile vs standard fine-tune head
- [ ] Run Reptile + SmartReplayBuffer, measure forgetting

---

## Phase 6 — Active Domain Adaptation (YouTube)

> Handled by **notebook 06** (`06_active_domain_adapt.ipynb`).
> Goal: adapt to YouTube-domain clips using only a small AL-labeled budget,
> without forgetting UCF101 performance.

### 6.1 Domain Shift Setup
- [ ] Download 3 YouTube clips (one per class) using `yt-dlp`
- [ ] Extract 16-frame clips with identical UCF101 transforms
- [ ] Extract teacher + student embeddings for YouTube clips
- [ ] Measure distribution shift: compare UCF101 vs YouTube μ, σ per class
- [ ] Quantify baseline accuracy drop on YouTube clips

### 6.2 Active Learning Acquisition (`src/active_learning/acquisition.py`)
- [ ] Entropy: `H = -Σ p log p`
- [ ] Margin: difference between top-2 softmax scores
- [ ] Coreset: greedy max-coverage in embedding space
- [ ] Score all YouTube clips → select top-K per class
- [ ] Ablation: random vs entropy vs margin vs coreset at K=5/10/20/50

### 6.3 Smart Buffer + Reptile Adaptation
- [ ] Add AL-selected YouTube clips to SmartReplayBuffer (diverse strategy)
- [ ] Mix UCF101 hard exemplars + YouTube diverse exemplars in buffer
- [ ] Reptile adaptation: treat domain shift as a new episode
  - Inner loop: adapt EmbeddingHead on K AL-selected YouTube embs
  - Outer loop: Reptile update moves meta-weights toward adapted weights
- [ ] Optional: EWC penalty to protect UCF101-critical weights during adaptation
- [ ] Evaluate: UCF101 accuracy (forgetting) + YouTube accuracy (gain)

### 6.4 Results
- [ ] Plot: YouTube accuracy gain vs UCF101 forgetting tradeoff
- [ ] Table: strategy × budget ablation
- [ ] Save: adapted head + buffer + results JSON to Drive

---

## Phase 7 — Evaluation Protocol

- [ ] Implement metrics in `src/utils/metrics.py`:
  - [ ] Average Accuracy, Final Average Accuracy
  - [ ] Per-task accuracy matrix
  - [ ] Backward Transfer (Forgetting Measure)
  - [ ] Forward Transfer
  - [ ] Training time + GPU memory
- [ ] Evaluate all methods on same protocol after each task
- [ ] Save structured CSV / JSON results
- [ ] Plots (notebook 07):
  - [ ] Accuracy vs task index (one line per method)
  - [ ] Forgetting bar chart across methods
  - [ ] YouTube accuracy vs UCF101 accuracy tradeoff
  - [ ] Teacher vs student comparison
- [ ] Run 3 seeds, report mean ± std

---

## Phase 8 — Best Method Selection

- [ ] Compare: Naive / EWC / SmartReplay / LwF / Reptile
- [ ] Select winner, justify with metrics
- [ ] Save final teacher checkpoint + hyperparameters

---

## Phase 9 — Final Knowledge Distillation

- [ ] Teacher: ResNet50+LSTM (best CL model)
- [ ] Student: MobileNetV3Small+LSTM (already in `models.py`)
- [ ] Loss: `CE + KL(student ∥ teacher) + MSE(proj, teacher_emb)`
- [ ] Tune temperature T and alpha
- [ ] Use SmartReplayBuffer during student training
- [ ] Compare teacher vs student: accuracy / params / inference time / size
- [ ] Optional: ResNet50 → ResNet34 → MobileNetV3 chain

---

## Phase 10 — Ablations

- [ ] Memory budget: vary replay buffer size (10 / 20 / 50 / 100 per class)
- [ ] Task order: test different class orderings
- [ ] KD: CE only / KD only / CE+KD / temperatures
- [ ] Replay strategy: FIFO / hard / uncertainty / diversity
- [ ] Domain adaptation: acquisition × budget grid
- [ ] Reptile: inner steps / epsilon / episodes

---

## Phase 11 — Report & Documentation

- [ ] Method descriptions (intuition, impl, hyperparams, pros/cons)
- [ ] Final report sections:
  - [ ] Introduction + related work (cite GIL paper: arXiv 2410.10497)
  - [ ] Methodology + pipeline diagram
  - [ ] Dataset + task split design
  - [ ] Experimental results + discussion
  - [ ] Domain adaptation section
  - [ ] Limitations + future work
- [ ] Visuals: pipeline diagram, accuracy curves, t-SNE/UMAP, AL workflow

---

## Phase 12 — Cleanup & Reproducibility

- [ ] Remove unused scripts and duplicate notebooks
- [ ] Rename notebooks to numbered convention (see Notebook Index above)
- [ ] Verify full pipeline reproducibility from config
- [ ] Final checklist:
  - [x] Dataset prepared, task split fixed
  - [x] Backbone fine-tuned on Task 0
  - [x] EWC implemented
  - [x] Basic replay + LwF implemented
  - [x] Reptile meta-learning working
  - [ ] Smart prototype replay buffer
  - [ ] All CL methods evaluated on full task stream
  - [ ] Active Domain Adaptation (YouTube) complete
  - [ ] Best method selected and justified
  - [ ] KD student (MobileNetV3) trained and compared
  - [ ] Results plotted + tabled
  - [ ] Report written
  - [ ] Code clean and reproducible

---

## New Files to Create

| File | Purpose |
|------|---------|
| `src/cl_strategies/smart_replay.py` | SmartReplayBuffer with prototype + hard/uncertainty/diversity |
| `src/active_learning/acquisition.py` | Entropy, margin, coreset acquisition functions |
| `src/active_learning/domain_shift.py` | YouTube setup, shift measurement, AL-Reptile loop |
| `src/utils/metrics.py` | AA, FAA, forgetting, forward/backward transfer, timing |
| `src/cl_strategies/naive.py` | Complete naive sequential training |
| `notebooks/07_evaluation.ipynb` | Full method comparison + plots |
