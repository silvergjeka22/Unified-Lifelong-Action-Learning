# ULAL — Pipeline Design

> Master design document. Read before implementing anything.
> Replaces the deleted `ULAL_AGENT_PROMPT.md`.

---

## The one idea

The model has three parts. **The freeze line moves.** That is the whole project.

```
ResNet50   →   LSTM   →   head
 (23.5M)      (2.4M)     (70K)
```

- Line **left** → more trains → representation adapts → more memory needed
- Line **right** → less trains → cheaper → representation cannot adapt

**Research question:**

> How much of a video model must stay trainable for lifelong learning,
> and what does it cost in memory?

Every topic is one point on that curve. Nothing is decoration.

### The trade-off table (this is the thesis)

| Freeze boundary | Trainable | Replay unit | Memory/class (15 ex) | Adapts? |
|---|---|---|---|---|
| Nothing | 25.9M | raw clip | 144 MB | fully |
| **After layer4** | **2.4M** (LSTM+head) | `16×2048` | **960 KB** | **yes** |
| After LSTM | 70K (head) | `256-d` | 15 KB | no |
| After LSTM + GIL | 70K | (μ, σ) only | **2 KB** | no |

Five orders of magnitude. That span is the contribution.

### Why caching frame features works

```
raw clip (fp32)       : 16 × 3 × 224 × 224 × 4  =  9.63 MB   ← cannot cache
frame features (fp16) : 16 × 2048 × 2           =   65.5 KB  ← 147× smaller
h_last (fp32)         : 256 × 4                 =    1.0 KB  ← too lossy, LSTM frozen
```

Freeze the ResNet, cache `(16, 2048)`, train `LSTM + head` on the cache. The LSTM — the
real bottleneck — stays trainable, and 12 ablation runs become a coffee break.

**Storage:** fp16 on disk, fp32 in RAM (the LSTM needs fp32).

### Layer-by-layer (verified)

```
conv1     9,408  🔒            ┐
bn1         128  🔒            │
layer1  215,808  🔒            │ 8,543,296 stay ImageNet
layer2 1,219,584 🔒            │
layer3 7,098,368 🔒            ┘
─────────────────────────────
layer4 14,964,736 ✏️           ┐
LSTM    2,361,344 ✏️           │ 17,328,650 train on Task 0 = 67%
fc          2,570 ✏️           ┘
```
The ImageNet `fc` (2048→1000) is dropped by `children()[:-1]`.

---

## Class configuration

| Group | Count | Role |
|---|---|---|
| `SELECTED_CLASSES` | 10 | Base — Task 0 |
| `TASK_1` | 10 | CL increment 1 |
| `TASK_2` | 10 | CL increment 2 |
| `TASK_3` | 10 | **Unseen** — zero-shot only, never trained |
| `TASK_4` | 10 | Held out — few-shot probe (Section 7) |

> **ACTION:** `TASK_1`/`TASK_2` have 7 of 10 commented out → only 16 classes total.
> Uncomment them. One edit makes the README's "50 classes / 5 tasks × 10" true, makes
> GIL's "10% at a time" true, and gives a 10 → 20 → 30 CL stream.

Label space: `Base(0–9) → Task1 → Task2 → Task4`. `TASK_3` never gets a training label.

---

## Notebook layout

```
notebooks/
  ULAL.ipynb                       [built] principal: unified capstone, reads every section
  study/
    1_choose_model/       [done]   1.0_backbone_transfer.ipynb  (group-disjoint bake-off)
    2_extract_features/   [done]   2.0_extract_embeddings.ipynb (cache + ceiling probe)
    4_continual_learning/ [built]  4.1_naive ... 4.6_replay_lwf  (one per method)
    5_memory_ablation/    [built]  5.0_memory_ablation.ipynb
    6_gan_replay/         [built]  6.0_gil.ipynb
    7_meta_learning/      [built]  7.0_meta_learning.ipynb
    8_distillation/       [built]  8.0_distillation.ipynb
    9_domain_adapt/       [built]  9.0_domain_adapt.ipynb
    10_results/           [built]  10.0_results.ipynb
```

The freeze-boundary study (old Section 3) is folded into notebook 2.0: the joint linear
probe there is the ceiling. The 4.x notebooks each run one method explicitly; 10.0 and
ULAL only aggregate the saved JSONs.

Study notebooks **compare options**. The principal notebook assembles the results.

---

## Bootstrap (replaces fetch_src.sh + setup_colab.sh)

Every notebook opens with the same three cells. The first is self-contained — nothing
needs to pre-exist on Drive.

```python
import os, sys, subprocess
from getpass import getpass
REPO, BRANCH, ROOT = "silvergjeka22/Unified-Lifelong-Action-Learning", "GAN&ActiveLearning", "/content/ulal"
token = os.environ.get("GITHUB_TOKEN") or getpass("GitHub token (hidden): ")
if os.path.isdir(f"{ROOT}/.git"):
    subprocess.run(["git","-C",ROOT,"fetch","--depth","1","origin",BRANCH], check=True)
    subprocess.run(["git","-C",ROOT,"reset","--hard","FETCH_HEAD"], check=True)
else:
    subprocess.run(["git","clone","--depth","1","--branch",BRANCH,
                    f"https://{token}@github.com/{REPO}.git", ROOT], check=True)
sys.path.insert(0, ROOT)
```
```python
from src.bootstrap import setup
setup(drive=True, dataset=True)
```
```python
%run /content/ulal/src/imports.py
```

**What this fixed**

| Old | Problem | Now |
|---|---|---|
| `fetch_src.sh` | had to already be on Drive (chicken-and-egg) | self-contained cell |
| `fetch_src.sh` | one GitHub API call **per file** | one `git clone --depth 1` |
| `BRANCH="GAN&ActiveLearning"` in bash | `&` backgrounds the command | `subprocess` with a **list**, no shell |
| `setup_colab.sh` | sed-patched `config.py`, forced kernel restart | `ULAL_*` env vars; file immutable |
| token pasted in a cell | saved into the notebook | `getpass`, never stored |

`config.py` reads `ULAL_DATASET_ROOT`, `ULAL_OUTPUT_ROOT`, `ULAL_DATA_ROOT`,
`ULAL_DRIVE_PROJECT`, `ULAL_RESNET50_PATH`, with the old values as defaults.

---

## Checkpoint contract

Everything under `/content/drive/MyDrive/apai/`.

| Section | Consumes | Produces |
|---|---|---|
| 1 Backbone | UCF101 raw | `ResNet50_10C.pth` |
| 2 Cache | `ResNet50_10C.pth` | `cache/ulal_frame_features.pt` |
| 3 Freeze study | cache | `stage0_freeze_study.json` |
| 4 CL | cache + stage0 | `head_cl_best.pt`, `stage1_cl_results.json` |
| 5 Memory | cache | `stage2_memory_ablation.json` |
| 6 GIL | cache | `generator.pth`, `stage3_gil.json` |
| 7 Reptile | `head_cl_best.pt` | `head_meta.pt`, `stage4_fewshot.json` |
| 8 KD | cache + MobileNet cache | `student_ucf101.pt`, `stage5_kd.json` |
| 9 Domain adapt | student + head | `head_adapted_yt.pt`, `stage6_da.json` |
| 10 Results | all JSONs | `final_results.csv` |

---

## Dimension constants

| Constant | Value | Why |
|---|---|---|
| `BACKBONE_DIM` | 2048 | ResNet50 pooled output per frame |
| `CLIP_LEN` | 16 | frames per clip |
| `TEACHER_HIDDEN` | 256 | LSTM hidden — matches `ResNet50_10C.pth` |
| `STUDENT_HIDDEN` | 128 | MobileNetV3 student LSTM |
| `STUDENT_BACKBONE_DIM` | 576 | MobileNetV3-small features |
| `SEM_DIM` | 384 | all-MiniLM-L6-v2 |

---

# Sections

## Section 1 — Backbone baseline ✅

Train 5 backbones + LSTM on Task 0, pick the best.
`ResNet50 [TRAIN] → LSTM [TRAIN] → head [TRAIN]` — the only full-backbone training, and
correctly so: this is the transfer-learning stage.
**→ `ResNet50_10C.pth`**

## Section 2 — Cache frame features ✅

Run the frozen ResNet once, save `(16, 2048)` fp16 per clip, per split, per task. Also
stores `teacher.lstm.state_dict()` so Section 3's probes share an initialisation.

Splits cached **separately** — train fits, val selects, test reports, never mixed.

> Runs once (~20–40 min). Later notebooks load in ~30 s.

## Section 3 — Freeze-boundary study ⭐ ✅

Train on **all classes jointly** (no CL) at two boundaries:

| Probe | Model | Trainable |
|---|---|---|
| **A** | `ResNet 🔒 → LSTM 🔒 → head ✏️` | 70,416 |
| **B** | `ResNet 🔒 → LSTM ✏️ → head ✏️` | 2,431,760 |

Both load the same Task-0 LSTM. Only `requires_grad` differs.

**Two outputs from one experiment:**
1. **The boundary.** `B ≫ A` → the LSTM was the bottleneck, use B. `B ≈ A` → frozen
   features suffice, use A (34× fewer params, 64× cheaper replay).
2. **THE CEILING.** "SmartReplay 71%, ceiling 85%" is a sentence. "71%" is not.

**Expected weak spots for A:** `ApplyEyeMakeup` vs `ApplyLipstick` (close-up faces) and
`BoxingPunchingBag` vs `BoxingSpeedBag` (fine-grained). Base is all full-body sports, so
a CE-trained 256-d bottleneck likely discarded what those need.

## Section 4 — Continual learning ⭐ ✅

Base → Task1 → Task2. Head grows. Four arms:

| Arm | Mechanism | Memory/class |
|---|---|---|
| Joint (Sec 3) | ceiling | — |
| Naive | none | 0 |
| **Weight Aligning** | rescale new-class logit norms | **0** |
| EWC | Fisher penalty | Fisher + θ* |
| SmartReplay | 15 hard exemplars | 960 KB |

**Two protocol rules:**

1. **ONE loop** (`train_cl_arm`) for every arm. `buffer` → replay, `fisher` → EWC,
   neither → naive.
2. **Fixed epochs, NO best-val selection.** An earlier run selected the best epoch by val
   over all seen classes; that stopped Naive at epoch 2 — before it forgot anything *or*
   learned the task (`Base=97%, Task1=43%`) — while EWC trained all 15. And selecting on
   a val set spanning old classes **is itself memory**, so "Naive, 0 bytes" was false.

> **Why Weight Aligning.** Most forgetting in a classifier is task-recency bias. WA
> (Zhao et al., CVPR 2020) fixes that with zero stored data. Without it, "replay beats
> naive" is beating a strawman.

**EWC λ:** tune from the printed CE/EWC split — aim for EWC at 5–20% of CE. λ=5000 gave
0.5% (inert); λ=1 gave ~6% on the real data.

## Section 5 — Memory-budget ablation

Sweep {1, 5, 15, 50} per class × {random, hard, diverse}.
**→ accuracy vs bytes-per-class, log x.** The money figure — a curve, not a point.

Also tests the Task1-collapse hypothesis: Task2 gets ~100 samples/class while replayed
Task1 gets 15, a 6.7× vote imbalance.

## Section 6 — GIL generative replay

Store only (μ, σ); a frozen generator synthesises features.
`ResNet 🔒 → LSTM 🔒 → head ✏️` — GIL assumes a static feature space, so this arm runs at
boundary A. That's the point: it's the far-left end of the curve, **2 KB/class**.

**Two mandatory fixes:**
1. **Discriminator must see real per-clip features.** `gil_gan.py` passes `mu_b` — the
   class *mean* — as real, so the real distribution is 10 vectors and the generator
   collapses to `x̂ = μ`, ignoring `z` and `σ`.
2. **Drop the zero-shot half.** The CVAE learns text→prototype from 10 pairs and must
   extrapolate to unseen names. That is the entire ZSL problem; the paper uses all 101
   classes.

> The t-SNE claim "lower intra-class distance = better" is backwards — with bug #1 it's
> guaranteed true and measures **mode collapse**.

## Section 7 — Meta-learning (Reptile)

Two heads from Section 4's winner — standard vs Reptile. Probe both on **TASK_4** with
K ∈ {1, 5, 10} shots, same inner loop and lr.
**→ accuracy vs gradient steps.**

`TASK_4` is referenced **zero times** by any notebook — free held-out data.

**Two gotchas in `src/meta/reptile.py`:**
- Discards the query set (correct first-order Reptile) — `k_query` is inert, don't tune it.
- `n_way = num_classes`, so every episode samples **all** classes. Don't call it
  "5-way 1-shot episodic" in the report.

Lab test for Section 9, where `AL_BUDGET_PER_CLASS = 10` is exactly this regime.

## Section 8 — Knowledge distillation

Cache MobileNetV3 frame features `(16, 576)` = 18 KB/clip. Train student
`LSTM(576→128) + head` with `CE + KL(student‖teacher) + MSE(proj, teacher_emb)`.

> **Fixes a real bug.** `3.0_reptile_kd.ipynb` has `student.parameters()` × 0,
> `requires_grad` × 0, `torch.save` × 0. The student's LSTM is **randomly initialised and
> never trained**. `restore_head_to_student` copies back only `norm` + `classifier`, so
> the projector — the one piece that touched the teacher — is discarded. And `4.0`
> expects `student_ucf101.pt`, which `3.0` never saves.

## Section 9 — Active domain adaptation

`yt-dlp` → identical transforms → `measure_shift` → `select_top_k(entropy, k=10)` → mix
YouTube AL clips + UCF exemplars → `reptile_adapt`.

> **State the limit.** The ResNet is frozen, so low-level shift (compression, lighting,
> resolution) cannot be adapted away. LSTM+head absorb a roughly linear feature shift;
> they cannot recover what conv layers discarded.

## Section 10 — Unified results

One table (every arm × ceiling / AA / BWT / memory / params / latency) and one figure
(**accuracy vs memory**, log-x).

---

## Scope statement (use this wording)

> This project studies **continual learning of a video action recogniser under a moving
> freeze boundary**, and measures the accuracy/memory trade-off across five orders of
> magnitude of replay budget.

Do **not** claim "continual learning for video action recognition" unqualified — the
ResNet never adapts after Task 0 and a reviewer will find that immediately. The scoped
claim is real, current (latent replay; L2P / DualPrompt / SimpleCIL), and practically
motivated: you cannot backprop 25.9M params on an edge device, but you can update 2.4M.

**Also relevant:** more trainable parameters = more drift = *more* forgetting. The frozen
backbone acts as a regulariser. Freezing is a design choice, not a shortcut.

---

## Known repo issues

### Fixed

| Issue | Was | Now |
|---|---|---|
| `src/meta_learning/`, `src/cl_strategies/` loaded **after** `src/models/`, `src/utils/` | 21 silent name collisions; editing `src/models/teacher.py` had **no effect** | deleted; `imports.py` loads only real modules |
| `src/fine_tune/gil_gan.py` | README's `git rm -r src/fine_tune` would delete the GAN | moved to `src/gil/gil_gan.py` |
| `config.py` executed **twice** (`config.config` *and* `src.config.config`) | two module objects, two copies of every value | single root; only `src.config.config` resolves |
| `setup_colab.sh` sed-patching `config.py` | tree diverged from git; re-clone reverted paths | env vars, file immutable |
| `SmartReplayBuffer.norm(dim=1)` | crashed on `(N,T,D)` sequences | `.flatten(1)` — shape-agnostic |
| buffer `summary()` printed `dim=16` | that was the *time* axis | prints `shape=(16,2048) 655 KB` |
| best-val epoch selection in CL | arms not comparable; val over old classes **is** memory | `train_cl_arm`, fixed epochs |
| `finetune_head` MSE at `mse_weight=0` | shape crash on sequences | `zeros_t()` helper |

### Fixed since

| Issue | Was | Now |
|---|---|---|
| data leakage | shipped split scattered a video's clips across train/test (80% of groups); every accuracy inflated (teacher 98%, ceiling 97%) | group-disjoint split in 1.0 + 2.0; honest teacher 93%, ceiling 82% |
| `TASK_1`/`TASK_2` classes commented out | 16 classes | now a real 30-class stream (10 -> 20 -> 30) |
| GIL discriminator sees `mu_b` as "real" | generator collapses to `x = mu` | `train_gan_real` feeds real per-clip features (Section 6) |
| Student LSTM never trained | KD distilled nothing | 8.0 trains a real student head on cached MobileNet features |

### Still open

| Issue | Impact | Fix in |
|---|---|---|
| WeightAlign kept vs dropped | docs disagreed on leaky data | re-decide from 10.0 on clean 30-class results |
| Sections 7/8/9 not yet run on Colab | API wiring unverified end-to-end | first run may need a small fixup |
| `TASK_3`/`TASK_4` extend the stream to 50 | 30 classes today, 50 available | uncomment nothing needed; just extend the task list |

---

## Order of work

1. **1.0** — bake-off on the group-disjoint split, save the teacher
2. **2.0** — cache features (once, slow); read the ceiling before continuing
3. **4.1 - 4.6** — CL comparison (4.1 creates `head_base.pt`)
4. Then **5 -> 6 -> 7 -> 8 -> 9 -> 10**
5. **ULAL** last — the unified capstone that reads every section
