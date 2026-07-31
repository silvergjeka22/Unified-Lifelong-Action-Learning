# ULAL — `embeddings` branch structure

Everything on this branch trains on **cached embeddings**. The video is read once, in
notebook 2, and never again.

Read this before creating or editing any notebook. The style rules are enforced by the
`ulal-notebook` skill in `.claude/skills/`.

---

## 1. The idea in one page

```
video clip                frozen ResNet50            what we train
(16, 3, 224, 224)   -->   (16, 2048)          -->    LSTM + head
     9.63 MB                 65.5 KB                  2.43M params
```

The ResNet is fine-tuned once on the base classes (notebook 1), then **frozen forever**.
A frozen network gives the same answer for the same clip every time, so we run it once,
save the result, and train everything after it on those saved numbers.

| | per training epoch |
|---|---|
| reading video every epoch | ~80 s |
| reading the cache | ~2 s |

40x faster. That is the entire reason for this branch: it makes the ablations
(memory budget, lambda sweeps, 3 seeds) possible at all.

**We cache at 2048, not at the LSTM's 256 output.** Caching the 256-d vector would
freeze the LSTM permanently and it could never learn the new classes.

---

## 2. File tree

```
notebooks/
  study/
    1_choose_model/
      1.0_backbone_transfer.ipynb      [done]  group-disjoint bake-off, saves ResNet50_10C_groupsplit.pth
    2_extract_features/
      2.0_extract_embeddings.ipynb     [done]  video -> cache, probe the ceiling
    4_continual_learning/
      4.1_naive.ipynb                  [built] no protection (creates head_base.pt)
      4.2_weight_align.ipynb           [built] rescale new-class logit norms
      4.3_ewc.ipynb                    [built] Fisher penalty
      4.4_lwf.ipynb                    [built] self-distillation
      4.5_smart_replay.ipynb           [built] stored exemplars
      4.6_replay_lwf.ipynb             [built] both (iCaRL)
    5_memory_ablation/
      5.0_memory_ablation.ipynb        [built] accuracy vs bytes sweep
    6_gan_replay/
      6.0_gil.ipynb                    [built] 2 KB/class generative replay
    7_meta_learning/
      7.0_meta_learning.ipynb          [built] Reptile vs standard, few-shot on TASK_4
    8_distillation/
      8.0_distillation.ipynb           [built] MobileNet student, CE + MSE
    9_domain_adapt/
      9.0_domain_adapt.ipynb           [built] YouTube AL + Reptile adaptation
    10_results/
      10.0_results.ipynb               [built] all CL arms, accuracy vs memory
  ULAL.ipynb                           [built] unified capstone: reads every section, master figure

src/
  __init__.py
  bootstrap.py            clone repo, mount Drive, download dataset, set paths
  imports.py              loads everything into the notebook namespace
  config/
    config.py             classes, paths, hyperparameters
  data/
    dataset.py            UCF101Clips
    preprocessing.py      preprocess_dataset  (video -> .pt clips)
    cache.py              save / load the embedding cache
  models/
    teacher.py            ResNet50LSTMTeacher
    temporal_head.py      TemporalHead, make_temporal_head, expand_head
    pretrained.py         backbone variants          (notebook 1 only)
    baselines.py          ScratchCNNLSTM             (notebook 1 only)
    student.py            MobileNetV3SmallLSTMStudent (later, KD)
  cl/
    trainer.py            train_task, train_cl_arm, snapshot_teacher, fill_buffer
    ewc.py                compute_fisher, pad_fisher_after_expand
    smart_replay.py       SmartReplayBuffer
    rehearsal.py          distillation_loss
  kd/
    trainer.py            finetune_head
    utils.py              eval_head, extract_embeddings
  utils/
    train.py              evaluate_all_tasks, test_model, print_detailed_metrics,
                          extract_frame_features, ModelWrapper
    evaluate.py           report_accuracy, FullModel, clip_loaders, project_head_space
    probe.py              probe_report, separability, centroid geometry
    visualize.py          plot_confusion_matrix, plot_training_results, plot_latent_space
    metrics.py            cl_report, build_accuracy_matrix, average_accuracy, BWT
    seed.py               set_seed
    save.py               remap_teacher_checkpoint, save_arm_result
  meta/                   [later] Reptile
  gil/                    [later] GAN generative replay
  active/                 [later] YouTube domain adaptation
```

### Delete these

Each one was checked for real call sites first, not just grepped for the name.

| What | Why | Also clean up |
|---|---|---|
| `src/models/head.py` (`EmbeddingHead`) | replaced by `TemporalHead`; remaining mentions are **docstrings only**, no imports | fix the docstrings in `cl/smart_replay.py`, `active/domain_shift.py`, `active/acquisition.py` to say `TemporalHead` |

### Keep these, despite appearances

| What | Why |
|---|---|
| `run_cl_stream` in `cl/trainer.py` | **notebook 5.0 needs it** for the 13-config memory sweep — writing task sections out 13 times is not an option. The per-method notebooks (4.x) still use `train_task` explicitly. |
| `4.2_weight_align.ipynb` + `weight_align` in `temporal_head.py` | kept as a baseline. The old "measured nothing" verdict came from leaky 16-class data; re-decide on the clean 30-class results. |
| `bash/fetch_src.sh`, `bash/setup_colab.sh` | superseded by `src/bootstrap.py`; every notebook now uses the git-clone + `setup()` cells. Safe to delete once you confirm nothing local still calls them. |
| `meta/`, `gil/`, `active/`, `data/youtube.py`, `models/student.py` | used by sections 7 (Reptile), 6 (GAN replay), 9 (YouTube adaptation), 8 (KD). |
| `train_gan_real` in `gil/gil_gan.py` | corrected GAN loop used by 6.0 — the discriminator sees real per-clip features (the old `train_gan` used the class mean and collapsed). |
| `measure_latency` in `utils/train.py` | used by the 1.0 backbone bake-off to compare inference speed. |
| `models/pretrained.py`, `models/baselines.py`, `data/study_dataset.py` | notebook 1 uses them |

---

## 3. Which notebook does what

### Notebook 2.0 — extract embeddings

Runs once. Everything after it depends on the file it writes.

1. Preprocess video into `.pt` clips for base, task 1, task 2
2. Load `ResNet50_10C.pth`, freeze it
3. **Check the model itself**: accuracy on the base test set. If this is not ~98%, the
   checkpoint is wrong and nothing downstream is meaningful. Stop here.
4. Run the frozen ResNet over every clip, save `(16, 2048)` fp16 per clip
5. **Check the cache**: 1-NN and linear probe per task and jointly, silhouette score,
   closest class pairs, t-SNE
6. Save to `Drive/apai/cache/ulal_frame_features.pt`

The two checks in steps 3 and 5 answer different questions:

| Check | Question | If it fails |
|---|---|---|
| teacher accuracy | did the checkpoint load correctly? | wrong path or wrong architecture |
| probe | do the frozen features carry the NEW classes? | base classes are too narrow, add variety |

The **joint linear probe** is the reference number. If it reads 97%, no continual-learning
method can beat 97%, and every later result is read against it.

### Notebooks 3.1 - 3.5 — one per method

Same structure, same hyperparameters. Only the mechanism differs.

| Notebook | Method | What protects the old classes | Memory per class |
|---|---|---|---|
| 3.1 | Naive | nothing | 0 |
| 3.2 | EWC | penalty on weights that mattered before | Fisher + theta*, no data |
| 3.3 | LwF | a frozen copy of the model teaches the new one | 0, only weights |
| 3.4 | Replay | 15 stored exemplars mixed into every batch | ~2 MB |
| 3.5 | Replay + LwF | both | ~2 MB |

Each one trains, evaluates on embeddings, plots a confusion matrix and the latent space,
and saves its head plus a JSON of results.

**WeightAlign is dropped.** It scored 40.92% AA against Naive's 40.55% — inside the noise.
It corrects task-recency bias, and this dataset's forgetting is not task-recency bias, it
is classes overlapping in feature space. The arm measured nothing.

### Notebook 4.0 — final evaluation

Loads every saved head and does the expensive part **once**:

1. Rebuild the raw `.pt` clips (Colab wipes them between sessions)
2. For each head: evaluate on embeddings, then on real video through
   `FullModel` = frozen ResNet + trained LSTM + head
3. Confusion matrix for both
4. Side-by-side table: embeddings vs real images, per method
5. Final comparison of all methods, plus accuracy-vs-memory

**Why this is a separate notebook.** Real-image evaluation needs the raw clips, which cost
20-40 minutes to rebuild. Doing it inside each of the five method notebooks pays that cost
five times. Here it is paid once, and the five training notebooks stay fast.

It is also the honest end-to-end check: the head was trained on cached features, and this
confirms it still works on actual video. If the two numbers disagree by more than ~2%, the
cache and the clips have drifted apart.

---

## 4. Notebook structure — every notebook, no exceptions

Copied from `notebooks/Study/CL/APAI_NAIVE_CL.ipynb` on `main`. Headers are bold and
exactly these:

```
## **Connect Colab**
     git clone cell + tokens

## **Download DATASET**
     setup + %run imports + set_seed

## **Preprocess data**
     preprocess per task
     print the classes
     ALL_CLASSES_LIST + class_to_idx + print

## **Create Loaders**
     datasets, loaders, combined loader

## **Upload Model**
     build / load the head
     expand for the new classes
     evaluate_all_tasks  (before training)
     print the classifier dims

## **Task One**
     train_task1 = train_task(...)
     plot_training_results(train_task1)
     evaluate_all_tasks  (after training)
     test_model -> plot_confusion_matrix
     print_detailed_metrics

## **Task 2**
     loaders for task 2
     expand the head
     evaluate_all_tasks  (before)
     train_task2 = train_task(...)
     plot_training_results(train_task2)
     evaluate_all_tasks  (after)
     test_model -> plot_confusion_matrix
     print_detailed_metrics

## **Latent Space**
     t-SNE coloured by task and by class

## **Save**
     save_arm_result + torch.save(head)
```

Tasks are written out **explicitly**, one section each. Never a `for t in tasks:` loop.
Adding task 3 means copying the Task 2 section — that is how main does it, and it keeps
every intermediate number visible.

---

## 5. Code rules

1. **No `def` or `class` inside a notebook.** Helpers go in `src/`.
2. **One idea per cell.**
3. **Comments are rare and short**: `# BASE`, `# TASK 1`, `# imports`. Nothing longer.
   Explanation belongs in the markdown cell above the code.
4. **ASCII only.** No emoji, no arrows, no box characters.
5. **Plain assignments.** No multi-line comprehensions, no lambdas, no clever unpacking.
6. **Names copied from main**: `ALL_CLASSES_LIST`, `class_to_idx`, `base_test_loader`,
   `combined_test_loader`, `results_after_t1`, `train_task1`.
7. **`set_seed(cfg.SEED)`** after imports and again before each training call.
8. **Print everything worth knowing.** Class counts, clip counts, layer dims, per-task
   accuracy and the combined number. A silent cell is a bug.

### src/ rules

1. One job per function. If describing it needs "and", split it.
2. Positional arguments first, few keywords, no `**kwargs` passthrough.
3. Return plain values: a tensor, a float, a dict of floats.
4. A one-line docstring saying what it returns.
5. No printing inside a function unless printing is its job.
6. Match main's signatures so notebook cells look the same:

```
train_model(model, train_loader, val_loader)                     -> history
train_task(head, train_s, train_y, val_s, val_y, device, ...)    -> (head, history)
evaluate_all_tasks(model, device, base_loader, task_loaders, combined_loaders)
test_model(model, test_loader, device)                           -> (acc, preds, labels)
plot_confusion_matrix(labels, preds, num_classes, classes_list, name)
print_detailed_metrics(all_labels, all_preds, num_classes, classes_list,
                       split_old, split_new)
plot_training_results(history)
```

---

## 6. What every notebook must print or plot

Not optional. If a number is computed, it is shown.

**Notebook 2.0**

- clips per class per split
- teacher accuracy on base test
- cache size in MB, and how much smaller than the raw video
- probe table: 1-NN and linear, per task and joint, against chance
- silhouette per task and joint
- the 10 closest class pairs, flagged when they cross tasks
- 4 plots: probe bars, t-SNE by task, t-SNE by class, centroid distance heatmap

**Notebooks 3.1 - 3.5**

- class list and `class_to_idx`
- clip counts per loader
- classifier in/out dims after each expansion
- `evaluate_all_tasks` before and after every task
- training curve per task
- confusion matrix after every task
- `print_detailed_metrics` with the old/new split
- final per-task accuracy and the combined number
- t-SNE by task and by class
- for EWC and LwF: the CE and penalty terms per epoch, so the balance is visible

**Notebook 4.0**

- per method: embeddings accuracy, real-image accuracy, and the gap
- confusion matrix on real video
- final table of all methods
- accuracy vs memory

---

## 7. Config

`src/config/config.py` is the only place class lists and paths are defined.

```python
SELECTED_CLASSES  # base classes,     10
TASK_1            # first increment,  10
TASK_2            # second increment, 10
TASK_3, TASK_4    # reserved for later, 10 each
```

The CL stream is `10 -> 20 -> 30`. Equal class counts per task are not cosmetic: the
metrics in `utils/metrics.py` average over **tasks**, not classes, so a 3-class task
would carry the same weight in AA as a 10-class one while having a third of the test
clips. Under the old `10 / 3 / 3` split one Task1 test clip moved AA by 0.63%, against
0.20% for a Base clip — the small tasks dominated the noise and nothing under about 2%
was measurable.

To change the split, edit the lists here, delete the cache, and re-run notebook 2.0.
`check_cache_labels` refuses a cache whose label space no longer matches, so a stale
cache fails loudly instead of mislabelling.

Paths come from environment variables with sensible defaults, so `config.py` is never
edited at runtime:

```
ULAL_DATASET_ROOT   where the raw UCF101 lives
ULAL_DATA_ROOT      where the .pt clips are written
ULAL_DRIVE_PROJECT  Drive folder for cache, checkpoints, results
```

---

## 8. What gets written where

```
Drive/apai/
  cache/
    ulal_frame_features.pt              the embeddings, plus the Task-0 LSTM weights
  resnet50/models/
    ResNet50_10C_groupsplit.pth         notebook 1.0 (the frozen teacher)
  checkpoints/
    head_base.pt                        notebook 4.1 (the others load it)
    head_naive.pt ... head_replay_lwf.pt one per CL arm (4.1 - 4.6)
    student_head.pt, student_ucf101.pt  notebook 8.0
    head_adapted_yt.pt                  notebook 9.0
  gil/generator.pth                     notebook 6.0
  results/
    stage1_choose_backbone.json         notebook 1.0
    stage2_embedding_diagnostics.json   notebook 2.0 (holds the ceiling)
    stage1_arm_*.json                   per-CL-arm matrices and metrics (4.1 - 4.6)
    stage5_memory_ablation.json         notebook 5.0
    stage6_gil.json                     notebook 6.0
    stage7_fewshot.json                 notebook 7.0
    stage8_kd.json                      notebook 8.0
    stage9_da.json                      notebook 9.0
    final_comparison.json               notebook 10.0
    ulal_unified_summary.json           ULAL.ipynb
    *.png                               every plot
```

`head_base.pt` is trained once by notebook 4.1 and loaded by 4.2 - 4.6, 5.0 and 6.0, so
every method starts from identical weights. Delete it if the class configuration or the
teacher checkpoint changes.

---

## 9. Run order

```
1.0  choose backbone        group-disjoint bake-off, saves the teacher
2.0  extract embeddings     once, ~30 min. Check the probe before continuing.
4.1  naive                  run first, it creates head_base.pt
4.2  weight align           any order after 4.1
4.3  ewc
4.4  lwf
4.5  smart replay
4.6  replay + lwf
5.0  memory ablation        needs head_base.pt (from 4.1)
6.0  gan replay             needs head_base.pt; pip installs sentence-transformers
7.0  meta learning          extracts TASK_4 features (dataset needed)
8.0  distillation           needs a CL head (default head_replay_lwf.pt from 4.6)
9.0  domain adapt           needs a CL head; live YouTube downloads
10.0 results                reads stage1_arm_*.json, accuracy vs memory
ULAL unified capstone       reads every stage JSON, master figure
```

Sections 5/6 branch off after 4.1; 8/9 need a trained CL arm; 10 and ULAL only read
JSONs. Delete a stale `head_base.pt` before re-running so 4.1 retrains it on the current
cache.

Always push before running. The notebooks clone `src/` from GitHub, so uncommitted
changes are invisible to Colab.

---

## 10. Reading the results

Three questions the table should answer:

1. **Does Naive collapse?** It must. If base accuracy stays high with no protection,
   forgetting is not happening and the setup is wrong.
2. **Does the paid method beat the free ones?** Replay costs ~2 MB per class. EWC and
   LwF cost nothing. If Replay only matches them, the memory is not earning its cost.
3. **How far is the best method from the joint probe?** That gap is what the later
   sections exist to close.

Read every number against the probe from notebook 2.0. "71%" alone means nothing.
"71% against a 97% ceiling" is a sentence.

### Known about this dataset

`ApplyEyeMakeup`, `ApplyLipstick` (task 1) and `BrushingTeeth` (task 2) are all close-up
face-and-hand videos. They occupy the same region of feature space and compete. The base
classes are full-body sports, far away, so they are easy to protect.

Expect task 1 to suffer most and the two makeup classes to be predicted as
`BrushingTeeth`. The centroid heatmap in notebook 2.0 predicts this before any training,
and the confusion matrices confirm it. That is a result worth reporting, not a bug.
