# Unified Lifelong Action Learning (ULAL)

Continual learning for video action recognition on UCF101, trained end-to-end on
**real video frames** (no feature cache). A ResNet50 + LSTM is fine-tuned on a base
set of classes, then extended to new tasks with several continual-learning methods,
distilled into a small student, and adapted to a shifted domain with and without
meta-learning.

The study uses a 20-class subset: 10 base + 5 + 5, a `10 -> 15 -> 20` task stream.
Everything runs on Google Colab.

## Study notebooks

Run in order. Each is self-contained and follows the same layout.

```
notebooks/study/
  1_dataset/           dataset study, shipped-vs-group-disjoint split, preprocess clips
  2_choose_model/      backbone bake-off on real clips -> pick + save the teacher
  3_continual_learning/  naive, EWC, LwF, replay, replay + LwF
  4_distillation/      teacher -> MobileNet student: CE, KD, cosine, MSE
  5_domain_adapt/      adapt to a shifted domain, with vs without meta-learning
```

## Source layout

```
src/
  bootstrap.py          Colab: mount Drive, download UCF101, set paths
  imports.py            load every symbol into the notebook namespace
  config/config.py      classes, paths, hyperparameters
  data/
    dataset.py          UCF101Clips
    preprocessing.py    frames -> clips, group-disjoint split, leakage checks
    study.py            DatasetStudy
  models/
    backbones.py        ScratchCNN / ResNet18 / ResNet50 / DenseNet121 / VGG19 (+ LSTM)
    student.py          MobileNetV3-Small + LSTM (the KD student)
  training/
    train.py            train_model, evaluate_model, test_model, evaluate_all_tasks
    metrics.py          print_detailed_metrics
    visualize.py        plot_training_results, plot_confusion_matrix, plot_cl_forgetting
    seed.py             set_seed
  cl/
    ewc.py              EWC (Fisher penalty)
    rehearsal.py        ReplayBuffer, distillation_loss, train_continual (replay + LwF)
  kd/
    distill.py          train_student (ce / kd / cosine / mse)
  da/
    shift.py            synthetic domain corruption
    youtube.py          download real out-of-domain clips
    acquisition.py      active-learning selection (entropy / margin)
    adapt.py            reptile_adapt (meta) vs finetune_adapt (no meta)
```

## Data split

UCF101 clips named `v_Class_gXX_cYY.avi` come from source videos: every clip sharing
a group `gXX` is cut from the same recording. The shipped split mixes clips of one
recording across train / val / test, which leaks near-duplicate frames into test and
inflates accuracy. This project splits by whole group instead (`build_group_split`),
so no recording appears in two splits. Notebook 1 measures the leakage and builds the
clean split.

## Running on Colab

Each notebook opens with the same cells: clone `src/` from GitHub, run
`src.bootstrap.setup()`, then `%run src/imports.py`. Preprocessed clips are written to
the Colab local disk (not Drive); the trained model and result JSONs go to Drive.
Push before running, since Colab clones `src/` from GitHub.
