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
    1_backbone/
      1.0_choose_backbone.ipynb        [done]  picks ResNet50, saves ResNet50_10C.pth
    2_embeddings/
      2.0_extract_embeddings.ipynb     [todo]  video -> cache, and check the cache is good
    3_continual/
      3.1_naive.ipynb                  [todo]  no protection
      3.2_ewc.ipynb                    [todo]  Fisher penalty
      3.3_lwf.ipynb                    [todo]  self-distillation
      3.4_replay.ipynb                 [todo]  stored exemplars
      3.5_replay_lwf.ipynb             [todo]  both (iCaRL)
    4_final/
      4.0_final_evaluation.ipynb       [todo]  all heads on real video + comparison
  ULAL.ipynb                           [later] one clean end-to-end run, winners only

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
| `notebooks/study/4_continual_learning/4.2_weight_align.ipynb` | measured nothing, see section 3 | — |
| `weight_align` in `models/temporal_head.py` | only its notebook called it | `imports.py`, `models/__init__.py`, `cl/trainer.py` |
| `run_cl_stream` in `cl/trainer.py` | notebooks write task sections out explicitly now | `cl/__init__.py`, the example in `src/__init__.py` |
| `src/models/head.py` (`EmbeddingHead`) | replaced by `TemporalHead`; remaining mentions are **docstrings only**, no imports | fix the docstrings in `cl/smart_replay.py`, `active/domain_shift.py`, `active/acquisition.py` to say `TemporalHead` |
| old `notebooks/study/2_extract_features/`, `4_continual_learning/` | replaced by the tree above | — |

### Keep these, despite appearances

| What | Why |
|---|---|
| `bash/fetch_src.sh`, `bash/setup_colab.sh` | **notebook 1.0 still calls them** (cells 3 and 5). They can only go once notebook 1 is migrated to `src/bootstrap.py`. |
| `meta/`, `gil/`, `active/`, `data/youtube.py`, `models/student.py` | later sections: Reptile, GAN replay, YouTube adaptation, KD |
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
SELECTED_CLASSES  # base classes,  currently 10
TASK_1            # first increment,  currently 3
TASK_2            # second increment, currently 3
TASK_3, TASK_4    # reserved for later
```

To scale from `10 / 3 / 3` to `30 / 10 / 10`, edit the lists here and re-run notebook
2.0. Nothing else changes.

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
    ulal_frame_features.pt        the embeddings, plus the Task-0 LSTM weights
  checkpoints/
    ResNet50_10C.pth              notebook 1
    head_base.pt                  notebook 3.1 (the others load it)
    head_naive.pt
    head_ewc.pt
    head_lwf.pt
    head_replay.pt
    head_replay_lwf.pt
  results/
    embeddings_quality.json       notebook 2.0
    arm_naive.json                per-method matrices and metrics
    arm_ewc.json
    arm_lwf.json
    arm_replay.json
    arm_replay_lwf.json
    final_comparison.json         notebook 4.0
    *.png                         every plot
```

`head_base.pt` is trained once by notebook 3.1 and loaded by 3.2 - 3.5, so every method
starts from identical weights. Delete it if the class configuration changes.

---

## 9. Run order

```
1.0  choose backbone        once, already done
2.0  extract embeddings     once, ~30 min. Check the probe before continuing.
3.1  naive                  run first, it creates head_base.pt
3.2  ewc                    any order after 3.1
3.3  lwf
3.4  replay
3.5  replay + lwf
4.0  final evaluation       after all five
```

Always push before running. The notebooks clone `src/` from GitHub, so uncommitted
changes are invisible to Colab. The bootstrap cell checks for the key files and tells
you to push if they are missing.

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
