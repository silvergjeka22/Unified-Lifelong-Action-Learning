# Agent Prompt: Refactor All Study Notebooks

## Context

This is a Colab-based research project on **Unified Lifelong Action Learning** using UCF101.
The project has the following layout:

```
notebooks/study/
  1_choose_model/1.0_backbone_transfer.ipynb
  2_cl/2.0_naive_cl.ipynb
  2_cl/2.1_ewc_cl.ipynb
  2_cl/2.2_replay_lwf.ipynb
  3_mamal/3.0_reptile_kd.ipynb
  4_active_dom_adopt/4.0_active_domain_adapt.ipynb
  5_gan_reply_buffer/5.0_gil_gan.ipynb

src/
  config/config.py          ← central config file
  imports.py                ← executed via %run to set up all symbols
  data/dataset.py, preprocessing.py, study_dataset.py, youtube.py
  models/pretrained.py, baselines.py, teacher.py, student.py, head.py
  cl/naive.py, ewc.py, rehearsal.py, smart_replay.py
  kd/utils.py, trainer.py, mseLoss.py
  meta/reptile.py, sampler.py
  meta_learning/models.py, reptile.py, sampler.py
  active/acquisition.py, domain_shift.py
  fine_tune/trainer.py, visualizer.py, gil_gan.py
  utils/metrics.py, save.py, train.py, visualize.py

bash/
  fetch_src.sh   ← downloads /src from GitHub into /content/src/
  setup_colab.sh ← downloads UCF101 via Kaggle, updates config.py, restarts kernel
```

---

## The Golden Pipeline Structure

Every notebook must follow **exactly this cell order**, modelled on notebooks 1.0 and 2.0 which already do this correctly:

### Cell 1 — Connect Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2 — Environment variables (tokens/keys)
```python
import os
os.environ["GITHUB_TOKEN"] = "xx"
os.environ["KAGGLE_USERNAME"] = "xx"
os.environ["KAGGLE_KEY"] = "xx"
```

### Cell 3 — Fetch /src from GitHub
```python
!bash /content/drive/MyDrive/apai/fetch_src.sh
```

### Cell 4 — Download & preprocess dataset (runs setup_colab.sh, restarts kernel)
```python
!bash /content/drive/MyDrive/apai/setup_colab.sh
```
> This cell restarts the kernel. Everything that must survive the restart (env vars, Drive mount) must be redone in the next cells.

### Cell 5 — Imports (single cell, no function definitions)
```python
%run /content/src/imports.py
```

### Cell 6 — Preprocess data splits
Only call `preprocess_dataset(...)` here. No function definitions.

### Cell 7 — DataLoaders
Only instantiate `UCF101Clips` and `DataLoader`. No function definitions.

### Cell 8..N — Algorithm / experiment sections
- Load models, run training/evaluation, display results.
- **No function or class definitions anywhere in the notebook.**
- Every function used must live in a `src/` file and be importable.

---

## The Four Refactoring Rules

### Rule 1 — All hyperparameters go to `src/config/config.py`

Scan every notebook for magic numbers and named constants that are not already in `config.py`.
Add them there under clearly labelled sections. Examples of things that must move:

- `EWC_LAMBDA = 5000.0` → `cfg.EWC_LAMBDA`
- `EXEMPLAR_ROOT`, `SUBSET1_ROOT`, etc. → `cfg.EXEMPLAR_ROOT`, etc.
- `K_SHOT`, `K_QUERY`, `K_SUPPORT` → `cfg.K_SHOT`, `cfg.K_QUERY`, `cfg.K_SUPPORT`
- `INNER_LR`, `INNER_STEPS`, `SRC_EPSILON` → `cfg.INNER_LR`, etc.
- `REPTILE_EPOCHS`, `EPISODES_PER_EPOCH`, `REPTILE_PATIENCE` → `cfg.REPTILE_EPOCHS`, etc.
- `CE_WEIGHT`, `MSE_WEIGHT` → `cfg.CE_WEIGHT`, `cfg.MSE_WEIGHT`
- `FINETUNE_EPOCHS`, `FINETUNE_LR`, `FINETUNE_WD`, `FINETUNE_BATCH`, `FINETUNE_PATIENCE` → `cfg.*`
- `TEACHER_HIDDEN`, `STUDENT_HIDDEN`, `HEAD_DROPOUT`, `STUDENT_DROPOUT` → `cfg.*`
- `AL_BUDGET_PER_CLASS`, `AL_STRATEGY` → `cfg.AL_BUDGET_PER_CLASS`, `cfg.AL_STRATEGY`
- `ADAPT_INNER_LR`, `ADAPT_INNER_STEPS`, `ADAPT_EPSILON`, `ADAPT_EPISODES` → `cfg.*`
- `GAN_EPOCHS`, `GAN_LR`, `LAM1`, `LAM2`, `ALPHA_GP`, `N_CRITIC` → `cfg.*`
- `FT_EPOCHS`, `FT_LR`, `BATCH_SIZE_GAN`, `J_SYNTH`, `CVAE_LR`, `CVAE_EPOCHS` → `cfg.*`
- `FEAT_DIM`, `LATENT_DIM`, `NOISE_DIM`, `SEM_DIM` → `cfg.*`
- `LABEL_SMOOTHING` → `cfg.LABEL_SMOOTHING`
- `NUM_WORKERS` → `cfg.NUM_WORKERS`
- `CLIPS_PER_VID`, `limit` (max_samples) → `cfg.*`

Group them in `config.py` by section with a comment header:
```python
# ── EWC ─────────────────────────────────────────────
EWC_LAMBDA = 5000.0

# ── Reptile / Meta-learning ──────────────────────────
K_SHOT     = 40
K_QUERY    = 1
...

# ── Knowledge Distillation ───────────────────────────
CE_WEIGHT    = 1.0
MSE_WEIGHT   = 0.2
...

# ── Active Domain Adaptation ─────────────────────────
AL_BUDGET_PER_CLASS = 10
AL_STRATEGY         = "entropy"
...

# ── GAN / GIL ────────────────────────────────────────
GAN_EPOCHS   = 30
...
```

### Rule 2 — All imports at the beginning, via `%run /content/src/imports.py`

`imports.py` is already the central import hub. It is executed with `%run` so all its names land in the notebook namespace.

- Remove any scattered `import` statements from mid-notebook cells.
- If a notebook needs an import that `imports.py` does not currently export, **add it to `imports.py`** (or the relevant `src/__init__.py`), do not add it inline.
- The only acceptable exception is `from google.colab import drive` in Cell 1.

### Rule 3 — All functions and classes go to `src/`, never defined in a notebook cell

Every function currently defined inline in a notebook must be moved to the appropriate `src/` module:

| Inline function | Move to |
|---|---|
| `remap_teacher_checkpoint(raw_ckpt)` | `src/utils/save.py` or `src/models/teacher.py` |
| `extract_clips(video_path, ...)` | `src/data/youtube.py` |
| `extract_features(model, clip_tensor)` | `src/kd/utils.py` |
| `process_dataloader(...)` | `src/kd/utils.py` |
| `make_loader(root, shuffle)` | `src/data/dataset.py` or `src/utils/train.py` |
| `extract_class_features(model, root, ...)` | `src/utils/train.py` |
| `run_incremental_baseline(strategy, ...)` | `src/fine_tune/gil_gan.py` |
| Any plot helper defined inline | `src/utils/visualize.py` or `src/fine_tune/visualizer.py` |
| Any metric helper defined inline | `src/utils/metrics.py` |
| `ModelWrapper(nn.Module)` wrapper class | `src/utils/train.py` |
| `chunks(lst, n)` | `src/utils/train.py` |

After moving, make sure the function is exported from the module's `__init__.py` if one exists, and is already importable via `imports.py` or via the explicit import already present at the top of the notebook.

### Rule 4 — Notebook cells contain only execution, no logic

After refactoring, every cell should look like one of these patterns:

**Pattern A — call a function**
```python
preprocess_dataset(splits=["train", "val", "test"], target_classes=cfg.TASK_1)
```

**Pattern B — instantiate and use**
```python
train_loader = DataLoader(UCF101Clips(cfg.TASK1_ROOT + "/train", class_to_idx), 
                          batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
```

**Pattern C — assign a result and display**
```python
results = train_model(model, train_loader, val_loader, num_epochs=cfg.FINETUNE_EPOCHS)
plot_training_results(results)
```

**Pattern D — save/load checkpoint**
```python
torch.save(model.state_dict(), cfg.RESNET50_PATH)
```

**Never:**
```python
def my_helper(...):   # ← FORBIDDEN in a notebook cell
    ...
```

---

## Per-Notebook Instructions

### `1.0_backbone_transfer.ipynb` — Backbone Transfer Learning

**This is the reference notebook. It already has good structure. Fix only:**

1. The inline `extract_features` and `process_dataloader` functions near the bottom (feature extraction for drive saving) — move to `src/kd/utils.py`.
2. Confirm all hyperparams (`BATCH_SIZE`, `num_epochs` values) are read from `cfg.*`.
3. The DataLoaders section still hardcodes `train_loader_10`, etc. — rename to use `cfg.BATCH_SIZE` and `cfg.SELECTED_CLASSES` consistently.

**Full pipeline for this notebook:**
1. Connect Drive → set env vars → fetch_src → setup_colab (restarts)
2. `%run imports.py`
3. Study dataset (call `DatasetStudy().run_all()`)
4. Preprocess: `preprocess_dataset()` for full 10-class set
5. DataLoaders: base 10-class train/val/test
6. Section: Custom CNN+LSTM — instantiate, summary, train, test, plot CM
7. Section: ResNet18+LSTM — same
8. Section: ResNet50+LSTM — same, **save best model to Drive at `cfg.RESNET50_PATH`**
9. Section: DenseNet121+LSTM — same
10. Section: VGG19_BN+LSTM — same
11. Section: Extract & save features — load best ResNet50, call `extract_features` / `save_features` from `src/kd/utils.py`

---

### `2.0_naive_cl.ipynb` — Naive Continual Learning

**Already has good structure. Fix:**

1. Add `RESNET50_PATH` usage from `cfg` (currently correct — keep it).
2. Any hardcoded `lr`, `weight_decay`, `num_epochs` values → `cfg.*`.

**Full pipeline:**
1. Connect Drive → env → fetch_src → setup_colab (restarts)
2. `%run imports.py`
3. Preprocess: base (test only), task1 (train/val/test), task2 (train/val/test)
4. DataLoaders: base_test, task1_*, task2_*, combined loaders
5. Section: Upload model — load `ResNet50LSTM` from `cfg.RESNET50_PATH`, expand classifier
6. Section: Baseline evaluation — `evaluate_all_tasks(...)` before training
7. Section: Task 1 — `train_model(...)`, `evaluate_all_tasks(...)`, `plot_confusion_matrix(...)`, `print_detailed_metrics(...)`
8. Section: Task 2 — same pattern, expand classifier again
9. Section: Task 3 — [TODO placeholder]

---

### `2.1_ewc_cl.ipynb` — EWC Continual Learning

**Issues to fix:**

1. `EWC_LAMBDA = 5000.0` is defined inline → move to `cfg.EWC_LAMBDA`.
2. `lr=1e-5`, `weight_decay=1e-4` → `cfg.EWC_LR`, `cfg.EWC_WD`.
3. Optimizer is instantiated inline with `torch.optim.Adam(filter(...))` repeated twice — acceptable inline (it's execution, not a function def), but use `cfg` values.

**Full pipeline:**
1. Connect Drive → env → fetch_src → setup_colab (restarts)
2. `%run imports.py`
3. Preprocess: base (train+test), task1 (all splits), task2 (all splits)
4. DataLoaders: base loaders + fisher loader for base, task1 loaders, task2 loaders
5. Section: Load model — `ResNet50LSTM` from `cfg.RESNET50_PATH`
6. Section: Task 1
   a. `compute_fisher(model, base_fisher_loader, device, task_id=0, ...)`
   b. `expand_classifier(model, total_classes)`
   c. `pad_fisher_after_expand(...)`
   d. `check_ewc_penalty(...)`
   e. `train_ewc(model, task1_train_loader, task1_val_loader, optimizer, device, fisher_dict, optpar_dict, ewc_lambda=cfg.EWC_LAMBDA, epochs=cfg.EWC_EPOCHS)`
   f. `ewc_report(model, fisher_dict, optpar_dict)`
   g. `test_model(...)` → `plot_confusion_matrix(...)`
7. Section: Task 2 — same pattern, compute fisher for task1, expand, train_ewc, report, evaluate

---

### `2.2_replay_lwf.ipynb` — Replay Buffer + LwF

**Issues to fix:**

1. `batch_size = 8` defined inline → use `cfg.BATCH_SIZE`.
2. `limit = 5` (max_samples) → `cfg.EXEMPLAR_LIMIT` (add to config).
3. Hardcoded `lr=1e-4`, `epochs=5`, `epochs=3`, `lambda_distill=5.0 / 2.0` → `cfg.*`.
4. `ALL_CLASSES_LIST` is rebuilt repeatedly in each section — this is fine (it's just an assignment, not a function definition).

**Full pipeline:**
1. Connect Drive → env → fetch_src → setup_colab (restarts)
2. `%run imports.py`
3. Preprocess: exemplars (limited), subset1 (full + limited), subset2 (full)
4. DataLoaders: old exemplars, subset1, subset2 loaders, combined loaders
5. Section: Replay Buffer (no KD)
   a. Init `ReplayBuffer`, fill from exemplar loader
   b. Load `ResNet50LSTM` → expand classifier
   c. `train_continual(model, teacher=None, ..., kd=False)`
   d. `evaluate_all_tasks(...)`, `test_model(...)`, `print_detailed_metrics(...)`, `plot_confusion_matrix(...)`
   e. Subset 2: expand classifier, update buffer, train, evaluate
6. Section: Replay Buffer + Knowledge Distillation (KD=True)
   a. Re-init buffer, load teacher + student (both `ResNet50LSTM` from `cfg.RESNET50_PATH`)
   b. `train_continual(student, teacher=teacher, ..., kd=True, lambda_distill=cfg.LAMBDA_DISTILL)`
   c. Same evaluation
   d. Subset 2: deep-copy student → new teacher, expand both, train, evaluate

---

### `3.0_reptile_kd.ipynb` — Reptile + Knowledge Distillation

**Issues to fix (most inline function definitions here):**

1. All hyperparameters defined in a big block inline → move every one to `cfg.*`:
   - `EXEMPLAR_ROOT` → `cfg.EXEMPLAR_ROOT`
   - `NUM_WORKERS`, `TEACHER_HIDDEN`, `STUDENT_HIDDEN` → `cfg.*`
   - `CE_WEIGHT`, `MSE_WEIGHT` → `cfg.*`
   - `K_SHOT`, `K_QUERY`, `K_SUPPORT` → `cfg.*`
   - `INNER_LR`, `INNER_STEPS`, `SRC_EPSILON` → `cfg.*`
   - `REPTILE_EPOCHS`, `EPISODES_PER_EPOCH`, `REPTILE_PATIENCE` → `cfg.*`
   - `HEAD_DROPOUT`, `STUDENT_DROPOUT`, `FINETUNE_WD`, `LABEL_SMOOTHING` → `cfg.*`
   - `FINETUNE_EPOCHS`, `FINETUNE_LR`, `FINETUNE_BATCH`, `FINETUNE_PATIENCE` → `cfg.*`

2. `remap_teacher_checkpoint(raw_ckpt)` defined inline → move to `src/utils/save.py` as `remap_teacher_checkpoint`.

3. `make_loader(ds, shuffle)` defined inline → move to `src/data/dataset.py` or `src/utils/train.py`.

4. `ModelWrapper(nn.Module)` defined inline → move to `src/utils/train.py`.

**Full pipeline:**
1. Connect Drive → env → fetch_src → setup_colab (restarts)
2. `%run imports.py`
3. Preprocess: exemplars limited to `cfg.EXEMPLAR_LIMIT` samples per class
4. DataLoaders: train/val/test for old classes using `make_loader` from src
5. Section: Load Teacher & Student
   - `teacher = ResNet50LSTMTeacher(cfg.TEACHER_HIDDEN, num_old, 0.4).to(device)`
   - Load from `cfg.RESNET50_PATH` using `remap_teacher_checkpoint` from src
   - `student_base`, `student_meta`, `student_std` = `MobileNetV3SmallLSTMStudent(...)`
   - Display initial teacher vs student accuracy
6. Section: Extract Embeddings
   - `train_t_embs, train_labels = extract_embeddings(teacher, train_loader, device)` — for teacher and both students
   - `limit_to_k_per_class(...)` for 5-shot setup
7. Section: WITH MAMAL
   - `head_meta = EmbeddingHead(cfg.STUDENT_HIDDEN, cfg.TEACHER_HIDDEN, num_classes, cfg.HEAD_DROPOUT)`
   - `head_meta = train_reptile(head_meta, ..., cfg.REPTILE_EPOCHS, cfg.EPISODES_PER_EPOCH, cfg.K_SUPPORT, cfg.K_QUERY, cfg.INNER_LR, cfg.INNER_STEPS, cfg.SRC_EPSILON, cfg.CE_WEIGHT, cfg.MSE_WEIGHT, cfg.REPTILE_PATIENCE)`
   - `head_meta = finetune_head(head_meta, ..., cfg.FINETUNE_EPOCHS, cfg.FINETUNE_LR, cfg.FINETUNE_WD, cfg.FINETUNE_BATCH, cfg.FINETUNE_PATIENCE, cfg.CE_WEIGHT, cfg.MSE_WEIGHT, label_smoothing=cfg.LABEL_SMOOTHING)`
   - `student_meta = restore_head_to_student(student_meta, head_meta)`
   - `meta_class_acc, meta_total_acc = calculate_accuracies(student_meta, test_loader, num_old, device)`
   - Print comparison table
8. Section: NO-MAMAL (standard fine-tune, no reptile)
   - Same but `finetune_head` only, no `train_reptile`
   - Print comparison table
9. Section: Comparison
   - Side-by-side per-class table (teacher / raw student / meta student / std student)
   - `evaluate_all_tasks(wrapped_student_meta, ...)`
10. **Save student checkpoint to Drive:**
    - `torch.save(student_meta.state_dict(), f"{cfg.CKPT_DIR}/student_ucf101.pt")`
    - `torch.save(head_meta.state_dict(), f"{cfg.CKPT_DIR}/head_ucf101.pt")`
    - Add `CKPT_DIR = "/content/drive/MyDrive/apai/checkpoints"` to `cfg`.

---

### `4.0_active_domain_adapt.ipynb` — Active Domain Adaptation

**This is the most critical notebook to refactor. Major issues:**

1. **TEACHER must be loaded from Drive, not student.** The notebook's Section 6 already does this correctly for the teacher. However, the student is loaded from `cfg.CKPT_DIR/student_ucf101.pt` which is correct — but the notebook currently has a broken import chain (`ModuleNotFoundError: No module named 'config'`). Fix the import path by ensuring `sys.path.insert(0, '/content')` is in `imports.py` or that `%run /content/src/imports.py` is used (which already handles path setup).

2. Remove the large "Configuration" cell (cell 0 / `9eeb4391`) that defines `YOUTUBE_CLIPS`, `AL_BUDGET_PER_CLASS`, etc. inline. Move all of it to `cfg`:
   - `cfg.YOUTUBE_CLIPS` (dict of class→URL)
   - `cfg.AL_BUDGET_PER_CLASS = 10`
   - `cfg.AL_STRATEGY = "entropy"`
   - `cfg.ADAPT_INNER_LR = 0.0005`
   - `cfg.ADAPT_INNER_STEPS = 10`
   - `cfg.ADAPT_EPSILON = 0.20`
   - `cfg.ADAPT_EPISODES = 30`
   - `cfg.YT_RAW_DIR = "/content/yt_raw"`
   - `cfg.YT_CLIPS_DIR = "/content/yt_clips"`
   - `cfg.RESULTS_DIR` and `cfg.CKPT_DIR` already in cfg

3. Remove the "Hyperparameters" cell (`5ee47f8e`) — all those values go to `cfg`.

4. `extract_clips(video_path, clip_len, n_clips)` inline → move to `src/data/youtube.py`.

5. The `yt_transform` (torchvision Compose) defined inline → this belongs in `src/data/youtube.py` alongside `extract_clips`, or in `cfg` as `cfg.spatial_transform` (already there — reuse it).

6. Section 3 (imports cell `15204d7e`) currently has all explicit imports — this entire cell must be replaced with `%run /content/src/imports.py`. All the imports it lists must be verified to be available after `%run imports.py`.

7. The `_remap_teacher_ckpt` lambda defined inline in Section 6 → use the moved `remap_teacher_checkpoint` from `src/utils/save.py`.

**Full pipeline:**
1. Connect Drive → env → fetch_src
2. Install extra deps: `!pip install -q yt-dlp umap-learn`
3. setup_colab (restarts kernel)
4. `%run /content/src/imports.py`
5. Preprocess: call `preprocess_dataset(splits=[], target_classes=cfg.SELECTED_CLASSES)` using `cfg.BASE_ROOT`
6. DataLoaders: train/val/test for base 10 classes
7. Section: Load Teacher & Student
   - `teacher = ResNet50LSTMTeacher(cfg.TEACHER_HIDDEN, num_classes, 0.4).to(device)`
   - Load teacher from `cfg.RESNET50_PATH` using `remap_teacher_checkpoint(...)` **from Drive**
   - `student = MobileNetV3SmallLSTMStudent(...).to(device)`
   - Load student from `cfg.CKPT_DIR + "/student_ucf101.pt"` **from Drive** (produced by notebook 3.0)
   - Freeze teacher
8. Section: Extract UCF101 Embeddings (with Drive cache)
   - Load from cache if exists, otherwise call `extract_embeddings(...)` and save
9. Section: Download YouTube Videos
   - `for cls_name, url in cfg.YOUTUBE_CLIPS.items(): subprocess.run(["yt-dlp", ...])`
10. Section: Preprocess YouTube Clips
    - `yt_clips = {}; for cls_name in cfg.YOUTUBE_CLIPS: clips = extract_clips(vpath, cfg.CLIP_LEN, cfg.CLIPS_PER_VID)` — `extract_clips` comes from `src/data/youtube.py`
11. Section: Extract YouTube Embeddings
    - Use teacher and student forward passes inline (execution only)
12. Section: Domain Shift Analysis
    - `compute_class_stats(...)`, `measure_shift(...)` from `src/active/domain_shift`
    - Plot t-SNE inline (calls to `plt`, `TSNE`, `PCA` — acceptable, no function defs)
    - `plt.savefig(f"{cfg.RESULTS_DIR}/domain_shift_tsne.png", ...)`
13. Section: Active Learning Acquisition
    - Load `head_for_scoring` from `cfg.CKPT_DIR + "/head_ucf101.pt"` (from notebook 3.0)
    - `al_selected = select_top_k(yt_s_dict, strategy=cfg.AL_STRATEGY, k=cfg.AL_BUDGET_PER_CLASS, head=head_for_scoring, device=device)`
14. Section: Smart Rehearsal Buffer
    - `buffer = SmartReplayBuffer(max_per_class=10)`
    - Fill with UCF101 exemplars + AL-selected YouTube clips
15. Section: Baseline Accuracy
    - Load head from `cfg.CKPT_DIR + "/head_ucf101.pt"`
    - `ucf_acc_before = eval_head(head, test_s, test_y, device)`
    - `yt_acc_before = eval_head(head, yt_s_all, yt_y_all, device)`
16. Section: Reptile Adaptation
    - `head_adapted = copy.deepcopy(head)`
    - `head_adapted = reptile_adapt(head_adapted, al_s, al_t, al_y, buffer, device, episodes=cfg.ADAPT_EPISODES, inner_lr=cfg.ADAPT_INNER_LR, inner_steps=cfg.ADAPT_INNER_STEPS, epsilon=cfg.ADAPT_EPSILON, ...)`
17. Section: Evaluation Before vs After
    - Inline table print + per-class breakdown
18. Section: Plots
    - Multi-panel plot using `matplotlib` (execution only, no helper function definitions)
    - `plt.savefig(f"{cfg.RESULTS_DIR}/adaptation_results.png", ...)`
19. Section: Save to Drive
    - `torch.save(head_adapted.state_dict(), f"{cfg.CKPT_DIR}/head_adapted_yt.pt")`
    - Save buffer, results JSON, AL selections JSON

---

### `5.0_gil_gan.ipynb` — GIL GAN

**Issues to fix:**

1. Move all hyperparameters from Section 0 (`cell-setup`) to `cfg`:
   - `cfg.FEAT_DIM = 256`
   - `cfg.LATENT_DIM = 256`
   - `cfg.NOISE_DIM = 256`
   - `cfg.SEM_DIM = 384`
   - `cfg.GAN_EPOCHS = 30`, `cfg.GAN_LR = 1e-4`
   - `cfg.LAM1 = 0.01`, `cfg.LAM2 = 0.1`
   - `cfg.ALPHA_GP = 10.0`, `cfg.N_CRITIC = 5`
   - `cfg.FT_EPOCHS = 5`, `cfg.FT_LR = 1e-4`
   - `cfg.BATCH_SIZE_GAN = 64`
   - `cfg.J_SYNTH = 50`
   - `cfg.CVAE_LR = 1e-4`, `cfg.CVAE_EPOCHS = 5`
   - `cfg.CVAE_INIT_EPOCHS = 50`
   - `cfg.GIL_CKPT_DIR = "/content/drive/MyDrive/apai/gil"` (or add under `cfg.CKPT_DIR`)

2. `remap_teacher_checkpoint` defined inline (again) → use the one from `src/utils/save.py`.

3. `extract_class_features(model, root, class_list, ...)` defined inline → move to `src/utils/train.py`.

4. `run_incremental_baseline(strategy, ...)` defined inline → move to `src/fine_tune/gil_gan.py`.

5. `chunks(lst, n)` defined inline → move to `src/utils/train.py`.

6. `forgetting(accs)` defined inline → move to `src/utils/metrics.py`.

7. `avg_intra_dist(emb, labels)` defined inline → move to `src/utils/metrics.py`.

8. The big import block in `cell-setup` (`import sys, os, copy, random` etc.) → replace with `%run /content/src/imports.py`. All those imports must be available after `%run`.

9. Section 0.5 (Data Preprocessing) uses `sys.path.insert(0, '/content/src')` then imports directly as `import config.config as cfg` and `from data.preprocessing import preprocess_dataset` — this must be unified so that after `%run imports.py`, these symbols are available as `cfg` and `preprocess_dataset`.

**Full pipeline:**
1. Connect Drive → env → fetch_src → `!pip install sentence-transformers -q`
2. setup_colab (restarts)
3. `%run /content/src/imports.py`
4. Section: Preprocess data — base, task1, task2, task3 (ZSL evaluation classes)
5. Section: Setup — read all hyperparams from `cfg`, init device, seeds, dirs
6. Section: Semantic Embeddings — `SentenceTransformer`, encode class names, build `sem_embs` dict
7. Section: Load Teacher (frozen)
   - `remap_teacher_checkpoint` from `src/utils/save.py`
   - Freeze all teacher params
8. Section: Extract Base Class Features
   - `base_class_feats = extract_class_features(teacher, cfg.BASE_ROOT, cfg.SELECTED_CLASSES, ...)` — function from `src/utils/train.py`
9. Section: Init Replay Buffer — `buffer = GILReplayBuffer()`, add base class prototypes
10. Section: Train CVAE — `train_cvae(E, buffer, device, lr=cfg.CVAE_LR, epochs=cfg.CVAE_INIT_EPOCHS)` — move loop to `src/fine_tune/gil_gan.py`
11. Section: GAN Architecture — instantiate `F_gen`, `G_dis`, `H_proj`
12. Section: Train GAN — `gan_history = train_gan(F_gen, G_dis, H_proj, buffer, ..., epochs=cfg.GAN_EPOCHS, ...)`; plot curves
13. Section: Incremental Setup — `chunks(seen_classes, ...)` from `src/utils/train.py`; pre-extract task1/task2 features
14. Section: Incremental Loop — call `run_incremental_loop(classifier, buffer, E, F_gen, all_seen_feats, ...)` — move entire loop to `src/fine_tune/gil_gan.py` as a function; plot base-class accuracy curve
15. Section: ZSL Test — `zsl_acc, zsl_preds, zsl_true = zero_shot_test(teacher, F_gen, E, unseen_classes, ...)`; GZSL harmonic mean
16. Section: Ablation — `accs_none = run_incremental_baseline('none', ...)`, `accs_random = run_incremental_baseline('random', ...)` — both from `src/fine_tune/gil_gan.py`; forgetting table
17. Section: t-SNE visualisation — inline matplotlib (no function defs); `avg_intra_dist` from `src/utils/metrics.py`
18. Section: Summary table

---

## Additional Structural Rules

### `imports.py` must export everything notebooks need

After refactoring, `%run /content/src/imports.py` must make all of these available in the notebook namespace without any additional import cells:

- `cfg` (the config module, as `import src.config.config as cfg`)
- `torch`, `nn`, `F`, `optim`, `DataLoader`, `ConcatDataset`, `TensorDataset`
- `np`, `os`, `random`, `copy`, `json`, `subprocess`
- `plt`, `TSNE`, `PCA`
- `summary` (torchinfo)
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- All model classes: `ResNet50LSTM`, `ResNet18LSTM`, `DenseNet121LSTM`, `VGG19BNLSTM`, `ScratchCNNLSTM`, `ResNet50LSTMTeacher`, `MobileNetV3SmallLSTMStudent`, `EmbeddingHead`
- All dataset/data utilities: `UCF101Clips`, `preprocess_dataset`, `DatasetStudy`
- All training utilities: `train_model`, `test_model`, `evaluate_all_tasks`, `expand_classifier`, `unfreeze_all`
- All CL utilities: `compute_fisher`, `train_ewc`, `ewc_report`, `pad_fisher_after_expand`, `check_ewc_penalty`, `ReplayBuffer`, `train_continual`, `SmartReplayBuffer`
- All KD utilities: `extract_embeddings`, `eval_head`, `restore_head_to_student`, `limit_to_k_per_class`, `calculate_accuracies`, `compare_head_vs_student_on_same_batch`
- All meta-learning utilities: `train_reptile`, `finetune_head`
- All active learning utilities: `select_top_k`, `compute_class_stats`, `measure_shift`, `reptile_adapt`
- All GAN utilities: `CVAE`, `FeatureGenerator`, `Discriminator`, `ProjectionHead`, `GILClassifier`, `GILReplayBuffer`, `train_gan`, `fine_tune_last2`, `update_buffer_and_cvae`, `zero_shot_test`, `gradient_penalty`
- All visualization utilities: `plot_confusion_matrix`, `plot_training_results`, `print_detailed_metrics`
- All save utilities: `remap_teacher_checkpoint`
- All metrics utilities: `cl_report`, `build_accuracy_matrix`, `forgetting`, `avg_intra_dist`
- All train utilities: `make_loader`, `extract_class_features`, `chunks`, `ModelWrapper`

### `sys.path` in Colab

`imports.py` must begin with:
```python
import sys
sys.path.insert(0, '/content')
sys.path.insert(0, '/content/src')
```
This fixes the `ModuleNotFoundError: No module named 'config'` seen in notebook 4.0 and 5.0.

### Drive paths convention

All Drive paths must follow `cfg.DRIVE_PROJECT = "/content/drive/MyDrive/apai"` already in use, then:
```python
cfg.CKPT_DIR    = f"{cfg.DRIVE_PROJECT}/checkpoints"
cfg.RESULTS_DIR = f"{cfg.DRIVE_PROJECT}/results"
cfg.GIL_CKPT_DIR = f"{cfg.DRIVE_PROJECT}/gil"
```
No notebook should hardcode Drive paths.

---

## What to Deliver

For each of the 7 notebooks and the supporting `src/` files, produce:

1. **Refactored notebook** — correct cell order, no inline function definitions, all cfg references, single `%run imports.py` cell.
2. **Updated `src/config/config.py`** — all new hyperparameter sections added.
3. **Updated `src/imports.py`** — exports every symbol needed by all notebooks.
4. **Updated `src/` modules** — each function moved from notebooks into the correct module, with proper `__init__.py` exports.

Do **not** change the scientific logic, algorithm implementations, or model architectures. Only restructure where code lives and how it is accessed.
