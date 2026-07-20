---
name: ulal-notebook
description: Write or edit ULAL study notebooks and src/ helpers. Use whenever creating a notebook under notebooks/study/, changing an existing one, or adding helper functions in src/ for them. Enforces the exact cell structure copied from notebooks/Study/CL/APAI_NAIVE_CL.ipynb on the main branch, plus the project's simple-code rules.
---

# ULAL notebook style

Every study notebook follows the structure of `notebooks/Study/CL/APAI_NAIVE_CL.ipynb`
on the `main` branch. Do not invent a new layout. Do not "improve" it.

## Cell structure — copy this exactly

Markdown headers are bold, in `## **Header**` form, with these exact names:

```
## **Connect Colab**
  1  git clone cell (unchanged, see below)
  2  os.environ tokens

## **Download DATASET**
  3  setup / dataset download
  4  %run imports  +  set_seed(cfg.SEED)

## **Preprocess data**
  5  preprocess per task, one comment line per task (# BASE, # TASK 1, # TASK 2)
  6  cfg.SELECTED_CLASSES
  7  print(cfg.SELECTED_CLASSES + cfg.TASK_1)
  8  ALL_CLASSES_LIST + class_to_idx + print

## **Create Loaders**
  9  datasets and loaders, then the combined loader

## **Upload Model**
 10  build the model / head, load the checkpoint
 11  expand to the new class count
 12  results_baseline = evaluate_all_tasks(...)
 13  print the classifier in/out dims

## **Task One**
 14  train_task1 = train_task(...)
 15  plot_training_results(train_task1)
 16  results_after_t1 = evaluate_all_tasks(...)
 17  test_model -> plot_confusion_matrix
 18  print_detailed_metrics

## **Task 2**
 19  task2 loaders + combined
 20  expand the head
 21  results_baseline = evaluate_all_tasks(...)
 22  train_task2 = train_task(...)
 23  results_after_t2 = evaluate_all_tasks(...)
 24  test_model -> plot_confusion_matrix
 25  print_detailed_metrics

## **Latent Space**
 26  t-SNE of the head's own embeddings, coloured by task and by class

## **Evaluation on Real Images**
 27  FullModel (frozen backbone + trained head) on raw clips
 28  test_model -> plot_confusion_matrix
 29  compare_embeddings_vs_real

## **Save**
 30  save_arm_result + torch.save(head)
```

Tasks are written out **explicitly**, one section each — never a `for t in CL_TASKS:`
loop. Adding a third task means copying the `## **Task 2**` section and renaming it.
That is how main does it, and it keeps every intermediate result visible.

## Code rules

1. **No `def` or `class` in a notebook.** Ever. Helpers live in `src/`.
2. **One idea per cell.** If a cell does two things, split it.
3. **Comments are rare and short.** `# BASE`, `# TASK 1`, `# imports`. No paragraphs,
   no explanation blocks inside code. Explanation belongs in the markdown cell above.
4. **No emoji, no icons, no arrows** (`->` is fine, `→` is not). ASCII only.
5. **Plain assignments over cleverness.** No comprehensions spanning lines, no
   `dict(**a, **b)` tricks, no lambdas.
6. **Name things like main does:** `ALL_CLASSES_LIST`, `class_to_idx`, `base_test`,
   `task1_train_loader`, `combined_test_loader`, `results_after_t1`, `train_task1`.
7. **Every notebook is self-contained** and starts from the same first four cells.
8. **`set_seed(cfg.SEED)`** right after `%run imports.py`, and again before training.

## The git clone cell — never change it

```python
import os, sys, subprocess
from getpass import getpass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO   = "silvergjeka22/Unified-Lifelong-Action-Learning"
BRANCH = "embeddings"
ROOT   = "/content/ulal"

token = os.environ.get("GITHUB_TOKEN") or getpass("GitHub token: ")
os.environ["GITHUB_TOKEN"] = token

if os.path.isdir(os.path.join(ROOT, ".git")):
    subprocess.run(["git", "-C", ROOT, "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", ROOT, "reset", "--hard", "FETCH_HEAD"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    "https://" + token + "@github.com/" + REPO + ".git", ROOT], check=True)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

`subprocess` takes a **list**, never a shell string — a branch name containing `&`
breaks under a shell.

## src/ helper rules

Helpers exist so notebooks stay flat. Keep them small.

1. **One job per function.** If you need "and" to describe it, split it.
2. **Positional args first, few keywords.** No `**kwargs` passthrough dicts.
3. **Return plain values** — a tensor, a float, a dict of floats. Not objects.
4. **A short docstring saying what it returns.** Skip the essay.
5. **No printing inside a function** unless printing is its job
   (`report_accuracy`, `cl_report`).
6. **Match main's helper signatures** where one already exists, so notebook cells
   look the same:
   - `train_model(model, train_loader, val_loader)` -> history dict
   - `evaluate_all_tasks(model, device, base_loader, task_loaders, combined_loaders)`
   - `test_model(model, test_loader, device)` -> (acc, preds, labels)
   - `plot_confusion_matrix(labels, preds, num_classes, classes_list, name)`
   - `print_detailed_metrics(all_labels, all_preds, num_classes, classes_list, split_old, split_new)`
   - `plot_training_results(history)`

## Embeddings vs main

Main trains the whole model on raw clips. These notebooks train on cached
`(16, 2048)` features, so three things differ and nothing else:

| main | here |
|---|---|
| `ResNet50LSTM(...)` + `load_state_dict` | `make_temporal_head(...)` + load `head_base.pt` |
| `expand_classifier(model, n)` | `expand_head(head, n)` |
| `UCF101Clips` + `DataLoader` | `emb_loaders(CACHE, ...)` |

`evaluate_all_tasks`, `test_model`, `plot_confusion_matrix`,
`print_detailed_metrics` and `plot_training_results` are used unchanged — wrap the
head in `ModelWrapper` so it returns bare logits.

**The label space is fixed by the cache.** Main grows `class_to_idx` inside each task
section; here it is built once in **Preprocess data** because the cached labels were
assigned when the cache was written. Call `check_cache_labels(META, ALL_CLASSES_LIST)`
after loading so a config change fails loudly instead of mislabelling silently.

## Every notebook ends the same way

In this order, no exceptions:

1. **Evaluation on embeddings** — per-task accuracy and the combined number
2. **Confusion matrix** on embeddings
3. **Latent space** — t-SNE by task and by class
4. **Evaluation on real images** — the same head behind the frozen backbone
5. **Confusion matrix** on real images
6. **Save**

Real-image evaluation needs the raw `.pt` clips, which Colab wipes between sessions.
Guard it with `clips_available(TASK_ROOTS, "test")` and print a note instead of
crashing when they are gone.

## Checklist before saving a notebook

- [ ] Header names and order match the list above
- [ ] No `def` or `class` anywhere in the notebook
- [ ] No emoji or non-ASCII symbols
- [ ] `set_seed(cfg.SEED)` after imports and before training
- [ ] Tasks written out explicitly, not looped
- [ ] Ends with embeddings eval, CM, latent space, real-image eval, CM, save
- [ ] Every code cell parses (`ast.parse` after stubbing `%` and `!` lines)
