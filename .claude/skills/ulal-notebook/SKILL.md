---
name: ulal-notebook
description: Write or edit ULAL study notebooks and src/ helpers. Use whenever creating a notebook under notebooks/study/, changing an existing one, or adding helper functions in src/ for them. Enforces the real-video notebook structure (copied from main's notebooks) and the project's simple-code rules.
---

# ULAL notebook style

Every study notebook trains on **real video clips** (no feature cache) and follows the
structure of the study notebooks on `main`. Do not invent a new layout. Do not add
embeddings, caching, or a `TemporalHead` - those were removed.

## Cell structure - copy this order

Markdown headers are bold, `## **Header**`, with these exact names:

```
## **Connect Colab**       git clone cell + tokens
## **Download DATASET**     src.bootstrap.setup() + %run imports + set_seed
## **Study Dataset**        (stage 1 only) DatasetStudy + the split study
## **Preprocess data**      preprocess_group_split per task, to the Colab local disk
## **Create Loaders**       UCF101Clips + DataLoader, plus the combined loader
## **Upload Model**         build / load the model, expand_classifier, evaluate_all_tasks (before)
## **Task One**             train ; plot_training_results ; evaluate_all_tasks (after) ; test_model -> plot_confusion_matrix ; print_detailed_metrics
## **Task 2**               loaders ; expand_classifier ; evaluate (before) ; train ; evaluate (after) ; confusion matrix ; metrics
## **Save**                 torch.save(model) + a small result JSON
```

Tasks are written out **explicitly**, one section each. Never a `for t in tasks:` loop.
Adding a task means copying the `## **Task 2**` section - that is how main does it, and
it keeps every intermediate number visible.

## Code rules

1. **No `def` or `class` in a notebook.** Helpers go in `src/`.
2. **One idea per cell.**
3. **Comments are rare and short**: `# BASE`, `# TASK 1`. Explanation goes in the
   markdown cell above.
4. **ASCII only.** No emoji, no arrows (`->` is fine, not the unicode arrow).
5. **Plain assignments.** No multi-line comprehensions, no lambdas.
6. **`set_seed(cfg.SEED)`** after imports and again before each training call.
7. **Print what matters** - class counts, clip counts, layer dims, per-task accuracy.

## src/ rules

1. One job per function; positional args first; return plain values.
2. A one-line docstring saying what it returns.
3. No printing unless printing is the job (`describe_group_split`, `print_detailed_metrics`).
4. No defensive noise - no `try/except`-and-print, no logging.

## Helper signatures (real video)

```
train_model(model, train_loader, val_loader)                 -> history
evaluate_all_tasks(model, device, base_loader, task_loaders, combined_loaders)
test_model(model, test_loader, device)                       -> (acc, preds, labels)
plot_confusion_matrix(labels, preds, num_classes, classes_list, name)
print_detailed_metrics(all_labels, all_preds, num_classes, classes_list, split_old, split_new)
plot_training_results(history)
expand_classifier(model, new_num_classes)
```

Models return **bare logits** from `forward`, so these helpers take the model directly
(no wrapper). `ResNet50LSTM` and the student also expose `.features(x)` for distillation.

## The git clone cell - never change it

```python
import os, sys, subprocess
from getpass import getpass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO   = "silvergjeka22/Unified-Lifelong-Action-Learning"
BRANCH = "real-video"
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

## Checklist before saving a notebook

- [ ] Header names and order match the list above
- [ ] No `def` or `class` anywhere in the notebook
- [ ] No emoji or non-ASCII symbols
- [ ] `set_seed(cfg.SEED)` after imports and before training
- [ ] Tasks written out explicitly, not looped
- [ ] Every code cell parses
