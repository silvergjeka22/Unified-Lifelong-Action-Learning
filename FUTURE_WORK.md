# ULAL — full pipeline and future-work roadmap

Honest scope map for the 101-class study. Read alongside `PIPELINE.md` and
`STRUCTURE.md`.

---

## 1. Verdict: does the full pipeline make sense?

Yes — as a staged research *program*, not as one deliverable. Every link is
coherent. The risk is scope: the full chain (backbone -> cache -> CL comparison ->
KD student -> domain adaptation) is really 3-4 studies bolted together. Build it in
tiers, and label the generative / CLIP parts as future work.

---

## 2. Scope decision — core now vs future work

**Core (finish this project):**
- Larger base backbone, cache all 101, the CL comparison (exemplar and
  regularisation arms), then KD -> Reptile + domain adaptation on the student.
- Decision (this pass): the **GAN is dropped**, and the **Gaussian prototype is
  deferred with it** — it was mainly the GAN's control, and the memory-vs-accuracy
  curve already comes from the exemplar sweep in notebook 5.0. Re-add the Gaussian
  in ~10 lines if a parametric-memory point is wanted (it stores statistics, not
  data — a genuinely different memory type).

**Future work (its own repo / study):**
- Full **GAN generative replay** (GIL), the **Gaussian prototype** control studied
  beside it (parametric vs exemplar memory), and the **CVAE zero-shot-by-name** head.
- The **CLIP / video-CLIP** zero-shot pipeline (the product variant).

**Why defer the GAN, not the KD:**
- The Gaussian prototype already answers "can 2 KB of cheap memory rival 1 MB of
  stored exemplars?" reliably and at almost no cost. It carries the cheap-memory
  narrative without a GAN.
- The GAN is the single biggest source of complexity and risk (unstable training,
  mode collapse, needs the full 101 to show anything), and its honest expected
  outcome is a *tie* with the Gaussian. A possibly-null result is fine as a focused
  study, bad as a load-bearing part of the main project.
- KD and domain-adaptation notebooks already exist and are stable; they round out
  the story cheaply.

---

## 3. The pipeline, phase by phase

Format: consumes -> produces | method | honest caveat.

**Phase 0 — Backbone**  [CORE]
- ~50 base-class videos -> frozen ResNet50+LSTM teacher.
- Method: transfer-learn layer4 + LSTM + head on the base, then freeze.
- Caveat: a small base makes the frozen features weak for the other ~50 classes.
  Use ~50 base so the late classes are still separable.

**Phase 1 — Cache**  [CORE]
- All 101 classes -> (16, 2048) fp16 per clip + the joint ceiling.
- Method: run the frozen backbone once; probe 1-NN / linear per task and jointly.
- Caveat: the joint probe is the ceiling; every later number is read against it.

**Phase 2 — CL comparison**  [CORE — the research question]
- cache -> per-arm accuracy matrices, AA, BWT, forgetting.
- Arms: naive, ewc, lwf, replay, replay+lwf. (Gaussian prototype deferred to the
  future generative-memory study — see below.)
- Stream: 50 -> 60 -> 70 -> 80 -> 90 -> 101 (five increments).
- Output: the money figure — accuracy vs memory-per-class (log x).
- Caveat: everything sits under the ceiling; replay is expected to win.

**Phase 3 — Compress**  [CORE — apply the winner]
- best CL teacher -> small MobileNet student.
- Method: KD with **soft targets**. The student has different eyes, so the stored
  memory does NOT transfer — only the teacher's probabilities do.
- Caveat: student <= teacher <= ceiling.

**Phase 4 — Adapt**  [CORE — apply the winner]
- student + a new task / shifted domain + a tiny label budget -> adapted student.
- Method: active learning picks the most informative clips; few-shot fine-tune
  (Reptile optional — likely ties plain fine-tune on frozen features).
- Caveat: the backbone is frozen, so low-level corruption cannot be adapted away.

**Phase 5 — Results**  [CORE]
- all JSONs -> one table + accuracy-vs-memory figure + teacher/student/adapt gains.

**Future — Generative memory**  [FUTURE, own study]
- cache -> GAN (F) synthesises old-class features from (mu, sigma); CVAE (E) maps a
  class NAME to a prototype for unseen classes.
- Why its own study: needs the full 101, careful GAN ablation, and an honest
  Gaussian control. This is the GIL paper reproduced at scale.

**Future — CLIP zero-shot**  [FUTURE, own study]
- swap the eyes for video-CLIP: new classes handled by name for free, no generator.
- The product-grade answer to class-hunger; a different backbone, so a clean break.

---

## 4. Class budget

```
~50 base            train the eyes
+10 x 5  -> 101     the CL stream (long stream = visible forgetting)
optional held-out   only if the future zero-shot arm is built
```

---

## 5. Honest limits (state these in any writeup)

- The frozen backbone caps everything (the ceiling). This is a design choice, not a
  bug: more trainable params = more drift = more forgetting; the frozen backbone is a
  regulariser.
- The GAN may tie the Gaussian even at 101.
- KD transfers behaviour (soft targets), not the stored memory.
- Domain adaptation cannot recover what the frozen conv layers discarded.
- Reptile likely ties plain few-shot fine-tuning on a small head over frozen features.

---

## 6. One-line scope statement

> Continual learning of a video action recogniser under a moving freeze boundary,
> measuring the accuracy/memory trade-off; generative and CLIP-based zero-shot
> memory are deferred to a dedicated follow-up study.
