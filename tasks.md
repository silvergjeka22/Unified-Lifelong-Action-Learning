# Continual Learning on UCF101 — Task Tracker

## Project Overview

A continual learning pipeline for video-based action recognition on UCF101.
A pretrained ResNet-50 is fine-tuned on a base set of 10 classes, then incrementally trained across 4 additional tasks of 10 new classes each.
Five continual learning strategies are compared (Naive, EWC, Rehearsal, LwF, La-MAML), with Active Learning integrated into rehearsal buffer selection.
The best-performing model is used as a teacher in a final Knowledge Distillation step to train a lightweight ResNet-18 student.

---

## Phase 1 — Dataset Preparation

- [x] Download and organize UCF101 dataset
- [x] Verify video integrity and organize by class
- [x] Explore dataset statistics (classes, video counts, durations)
- [x] Select 50 balanced classes for the project
- [x] Record class-to-index mapping in config
- [x] Build the 5-task continual learning split (Task 0–4, 10 classes each)
- [x] Save task split in `config.py` with fixed random seed (42)
- [x] Extract 16-frame clips at 224×224 with ImageNet normalization
- [x] Create train / val / test splits (80/20) per class
- [x] Save split metadata (clip path, label, task id, source video)

---

## Phase 2 — Dataloaders

- [x] Implement `dataset.py` — UCF101 clip dataset class
- [x] Implement `preprocessing.py` — frame extraction and transforms
- [x] Implement `study_dataset.py` — dataset exploration utilities
- [ ] Add task-incremental dataloader wrapper
- [ ] Add class-incremental dataloader wrapper
- [ ] Add replay buffer sampler
- [ ] Add distillation training dataloader
- [ ] Test all dataloaders end-to-end

---

## Phase 3 — Backbone and Baseline Setup

- [x] Choose ResNet-50 as the 2D CNN backbone
- [x] Load ImageNet pretrained weights
- [x] Verify input preprocessing matches pretrained settings
- [x] Implement expandable classification head (`pretrained.py`)
- [x] Implement baseline model variants (`baselines.py`)
- [x] Train on Task 0 — frozen backbone + new head
- [x] Train on Task 0 — partial fine-tuning
- [x] Train on Task 0 — full fine-tuning
- [x] Compare configurations and select best transfer learning setup
- [x] Save Task 0 checkpoint and log accuracy / loss curves

---

## Phase 4 — Core Training Framework

- [x] Implement generic training loop with validation (`trainer.py`)
- [x] Implement checkpoint saving and loading
- [x] Implement training visualizer (`visualizer.py`)
- [ ] Build incremental training pipeline (sequential tasks, dynamic head expansion)
- [ ] Evaluate on all seen tasks after each new task
- [ ] Track per-task accuracy, average accuracy, and forgetting
- [ ] Save all metrics to structured files (CSV / JSON)
---

## Phase 5 — Continual Learning Methods

### 5.1 Naive Fine-Tuning (Lower-Bound Baseline)

- [/] Implement `naive.py` — sequential training with no forgetting mitigation
- [ ] Evaluate forgetting on all previous tasks after each new task
- [ ] Save forgetting statistics

### 5.2 EWC (Elastic Weight Consolidation)

- [ ] Implement Fisher Information estimation
- [ ] Store parameter importance matrices after each task
- [ ] Add EWC penalty to the training loss
- [ ] Tune EWC lambda hyperparameter
- [ ] Validate that stronger regularization reduces forgetting

### 5.3 Rehearsal (Experience Replay)

- [ ] Design and implement replay buffer
- [ ] Set memory budget (per task / total fixed)
- [ ] Implement mixed sampling (current task + replay)
- [ ] Baseline: random replay selection
- [ ] Compare random vs. AL-informed buffer selection

### 5.4 LwF (Learning without Forgetting / Knowledge Distillation for CL)

- [ ] Save frozen teacher model after each task
- [ ] Generate soft targets from teacher logits
- [ ] Implement KL divergence loss with temperature scaling
- [ ] Tune alpha (CE vs. KD balance) and temperature
- [ ] Compare CE-only vs. CE + KD loss

### 5.5 La-MAML (Meta-Continual Learning)

- [ ] Study La-MAML algorithm before implementation
- [ ] Define episodic / meta-training procedure
- [ ] Implement inner-loop (task adaptation) updates
- [ ] Implement outer-loop (meta) updates
- [ ] Integrate with sequential task stream
- [ ] Validate on a small toy setting before full UCF101 run

---

## Phase 6 — Active Learning Integration

- [ ] Decide AL integration point (rehearsal buffer, new-task labeling, or both)

### 6.1 AL for Replay Buffer Selection

- [ ] Implement uncertainty-based sample selection (entropy / margin)
- [ ] Implement diversity-based sample selection
- [ ] Optionally implement centroid / prototype selection
- [ ] Compare AL selection vs. random buffer filling

### 6.2 AL for New Task Labeling

- [ ] Simulate unlabeled pool for each new task
- [ ] Define annotation budget per task
- [ ] Implement acquisition function (entropy / margin / diversity)
- [ ] Retrain using only AL-selected labeled samples
- [ ] Measure label efficiency vs. full supervision

### 6.3 AL Ablation Study

- [ ] Compare: no AL / random selection / uncertainty / diversity
- [ ] Report effect on accuracy and forgetting

---

## Phase 7 — Evaluation Protocol

- [ ] Define and implement all metrics:
  - [ ] Average Accuracy
  - [ ] Final Average Accuracy
  - [ ] Per-task Accuracy
  - [ ] Backward Transfer (Forgetting)
  - [ ] Forward Transfer
  - [ ] Training time and memory usage
- [ ] Evaluate all methods on the same protocol after each task
- [ ] Save results in structured tables (CSV / JSON)
- [ ] Build comparison tables across all CL methods
- [ ] Generate plots:
  - [ ] Accuracy vs. task index
  - [ ] Forgetting vs. method
  - [ ] Memory cost vs. accuracy
  - [ ] AL budget vs. performance
  - [ ] Teacher vs. student comparison
- [ ] Run multiple seeds and report mean ± std

---

## Phase 8 — Best Method Selection

- [ ] Compare all CL methods on accuracy, forgetting, and efficiency
- [ ] Justify and document the winning method
- [ ] Save final teacher model checkpoint with all hyperparameters
- [ ] Write short analysis: why did this method win?

---

## Phase 9 — Final Knowledge Distillation

- [ ] Choose student architecture: ResNet-18
- [ ] Compute teacher logits on the full task stream output
- [ ] Implement KD pipeline (KL divergence + CE combined loss)
- [ ] Tune alpha and temperature for distillation
- [ ] Save best student checkpoint
- [ ] Compare teacher vs. student:
  - [ ] Accuracy
  - [ ] Inference time
  - [ ] Parameter count
  - [ ] Storage size
- [ ] Optional: teacher → assistant (ResNet-34) → student (ResNet-18) chain

---

## Phase 10 — Experiments and Ablations

- [ ] Hyperparameter tuning (LR, batch size, memory size, EWC λ, KD temp/alpha, AL budget)
- [ ] Memory budget ablation — vary replay buffer size
- [ ] Task order ablation — test different class orderings
- [ ] Distillation ablation — CE only / KD only / CE+KD / different temperatures
- [ ] Active Learning ablation — measure benefit vs. added complexity

---

## Phase 11 — Documentation and Reporting

- [ ] Maintain structured experiment logs (config, checkpoint path, metrics, notes)
- [ ] Write method descriptions (intuition, implementation, hyperparameters, pros/cons)
- [ ] Prepare final report:
  - [ ] Introduction and related work
  - [ ] Methodology
  - [ ] Dataset and task split design
  - [ ] Experimental setup and results
  - [ ] Discussion, limitations, future work
- [ ] Prepare visuals (pipeline diagram, accuracy/forgetting curves, KD chart, AL workflow)
- [ ] Prepare presentation / demo summary

---

## Phase 12 — Final Cleanup

- [ ] Remove unused scripts and duplicate notebooks
- [ ] Organize configs, checkpoints, and outputs
- [ ] Verify full pipeline reproducibility from config files
- [ ] Final checklist:
  - [x] Dataset prepared
  - [x] Task split fixed
  - [x] Backbone selected and fine-tuned on Task 0
  - [ ] All CL methods implemented and evaluated
  - [ ] Active Learning integrated
  - [ ] Best method selected and justified
  - [ ] Final KD compression completed
  - [ ] Results plotted and tabled
  - [ ] Report written
  - [ ] Code cleaned and reproducible