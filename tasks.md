# Continual Learning on UCF101 — Project Task Plan

## Project Goal

Build, compare, and analyze a continual learning pipeline on the UCF101 action recognition dataset using a pretrained backbone.  
The project will compare multiple continual learning strategies, optionally integrate active learning, and finally apply knowledge distillation to compress the best model.

---

## Main Objectives

- Prepare UCF101 clips for continual learning experiments.
- Choose and implement a pretrained backbone model.
- Define incremental tasks from the selected classes.
- Implement and compare continual learning methods:
  - Naive fine-tuning
  - EWC
  - Rehearsal
  - Knowledge Distillation for CL (LwF)
  - MAML or preferably La-MAML
- Optionally integrate Active Learning in rehearsal buffer selection or task labeling.
- Evaluate all methods with the same protocol.
- Select the best method.
- Apply final knowledge distillation from the best model to a smaller student model.
- Produce final plots, tables, and conclusions.

---

## Phase 1 — Dataset Preparation

### Task 1.1 — Download and organize UCF101
- Download the dataset.
- Verify video integrity.
- Organize videos by class.
- Store original files in a read-only dataset folder.

### Task 2.2 — Explore dataset statistics
- Count:
  - number of classes
  - number of videos per class
  - average video duration
- Identify classes with too few or problematic samples.
- Create a small dataset summary file.

### Task 2.3 — Select the classes for the project
- Choose the set of classes to use.
- Prefer a balanced subset.
- Save the chosen class list in a config file.
- Record the class-to-index mapping.

### Task 2.4 — Build the continual learning task split
- Create:
  - Task 0 = 10 classes
  - Task 1 = 10 classes
  - Task 2 = 10 classes
  - Task 3 = 10 classes
  - Task 4 = 10 classes
- Save the split in a reproducible file:
  - `task_split.json` or `task_split.yaml`
- Fix the random seed for reproducibility.

### Task 2.5 — Extract clips
- Decide clip length:
  - e.g. 8, 16, or 32 frames
- Decide frame sampling strategy:
  - uniform sampling
  - stride-based sampling
  - center clip extraction
- Decide frame resolution.
- Extract clips from videos.
- Save metadata:
  - clip path
  - class label
  - task id
  - source video id

### Task 2.6 — Train/validation/test split
- Create train/val/test splits per class.
- Ensure no leakage between splits.
- Save split files for reuse.

### Task 2.7 — Build dataloaders
- Implement loaders for:
  - normal supervised training
  - task-incremental learning
  - class-incremental learning
  - replay buffer sampling
  - distillation training
- Test dataloaders carefully.

---

## Phase 3 — Backbone and Baseline Setup

### Task 3.1 — Choose backbone architecture
- Decide between:
  - 2D ResNet-50
  - 3D ResNet-based model
- If time is limited, start with 2D ResNet-50.
- If action quality matters more, consider a video model.

### Task 3.2 — Load pretrained weights
- Use pretrained weights from a standard source.
- Verify input preprocessing matches pretrained settings.
- Document normalization and resize parameters.

### Task 3.3 — Implement the classification head
- Replace the final layer for the chosen number of classes.
- Make the head expandable for new classes.
- Decide whether to use:
  - a single growing classifier, or
  - task-specific heads

### Task 3.4 — Transfer learning baseline
- Train the pretrained model on Task 0 only.
- Try:
  - frozen backbone + new head
  - partially fine-tuned backbone
  - full fine-tuning
- Compare these quickly.
- Choose the best setup for the CL experiments.

### Task 3.5 — Baseline evaluation
- Measure:
  - Task 0 accuracy
  - loss curves
  - training time
- Save the Task 0 checkpoint.

---

## Phase 4 — Core Training Framework

### Task 4.1 — Build a generic training engine
- Implement:
  - training loop
  - validation loop
  - checkpoint saving
  - early stopping if needed
  - logging
- Make it reusable for all methods.

### Task 4.2 — Build an incremental training pipeline
- Add support for:
  - sequential tasks
  - dynamic class expansion
  - evaluation after each task
- Save checkpoints after each task.

### Task 4.3 — Build metrics tracking
- Track:
  - training loss
  - validation loss
  - accuracy per task
  - overall average accuracy
- Save all metrics in structured files.

### Task 4.4 — Reproducibility setup
- Fix random seeds.
- Log hyperparameters.
- Log dataset split version.
- Log model version and training setup.

---

## Phase 5 — Implement Continual Learning Methods

### Task 5.1 — Naive fine-tuning
- Train sequentially on each task.
- No forgetting mitigation.
- Use this as the lower-bound baseline.

#### Subtasks
- Implement sequential task training.
- Evaluate old tasks after each new task.
- Save forgetting statistics.

---

### Task 5.2 — EWC
- Implement Fisher Information estimation.
- Compute parameter importance after each task.
- Add EWC penalty to the loss.

#### Subtasks
- Write code to estimate Fisher Information.
- Store parameter importance matrices.
- Tune EWC lambda.
- Validate that stronger regularization reduces forgetting.

---

### Task 5.3 — Rehearsal
- Implement a replay buffer.
- Store exemplars from previous tasks.
- Replay them while training on the current task.

#### Subtasks
- Design buffer memory format.
- Decide memory budget:
  - per class
  - per task
  - total fixed memory
- Implement sampling from current task + replay memory.
- Compare random replay first.

---

### Task 5.4 — LwF (Knowledge Distillation for CL)
- Use the previous model as a teacher.
- Distill old knowledge into the current model while learning the new task.

#### Subtasks
- Save frozen teacher after each task.
- Generate soft targets from teacher.
- Add KD loss with temperature.
- Tune alpha and temperature.
- Compare CE only vs CE + KD.

---

### Task 5.5 — MAML / La-MAML
- Decide whether to implement:
  - MAML, or
  - La-MAML
- Prefer La-MAML if feasible.

#### Subtasks
- Study the exact algorithm carefully before coding.
- Define episodic/meta-training procedure.
- Implement inner-loop and outer-loop updates.
- Integrate with sequential tasks.
- Validate on a small toy setting before full UCF101.

---

## Phase 6 — Add Active Learning

### Task 6.1 — Decide where AL is used
- Choose one of:
  - AL for rehearsal buffer selection
  - AL for selecting which new samples to label
  - both

### Task 6.2 — Implement AL for replay buffer selection
- Replace random replay selection with informative sample selection.

#### Subtasks
- Implement uncertainty-based selection.
- Implement diversity-based selection.
- Optionally implement prototype/centroid selection.
- Compare against random buffer filling.

### Task 6.3 — Implement AL for labeling new task data
- Simulate unlabeled pools for each new task.
- Select a subset for annotation.

#### Subtasks
- Define annotation budget per task.
- Implement acquisition function:
  - entropy
  - margin
  - diversity
- Retrain using only selected labeled samples.
- Compare label efficiency.

### Task 6.4 — AL ablation study
- Compare:
  - no AL
  - random selection
  - uncertainty selection
  - diversity selection
- Report effect on accuracy and forgetting.

---

## Phase 7 — Unified Evaluation Protocol

### Task 7.1 — Define evaluation metrics
- Average Accuracy
- Final Average Accuracy
- Per-task Accuracy
- Backward Transfer
- Forward Transfer
- Forgetting Measure
- Training time
- Memory usage
- Model size

### Task 7.2 — Evaluate after each task
- After Task 0, Task 1, Task 2, Task 3, and Task 4:
  - evaluate on all seen tasks
- Save results in tables.

### Task 7.3 — Build comparison tables
- Create tables comparing all CL methods.
- Include mean and standard deviation if multiple runs are used.

### Task 7.4 — Build plots
- Accuracy vs task index
- Forgetting vs method
- Memory cost vs accuracy
- Time cost vs accuracy
- AL budget vs performance
- Teacher vs student comparison

### Task 7.5 — Perform statistical validation
- Run multiple seeds if possible.
- Report average and variance.
- Identify unstable methods.

---

## Phase 8 — Choose the Best Method

### Task 8.1 — Select the best CL method
- Decide best based on:
  - accuracy
  - forgetting
  - efficiency
- Justify the choice clearly.

### Task 8.2 — Save the final teacher model
- Save the checkpoint of the best method after the last task.
- Record all relevant hyperparameters.

### Task 8.3 — Analyze why it won
- Was performance driven by:
  - replay memory
  - regularization
  - KD
  - meta-learning
- Write a short method analysis.

---

## Phase 9 — Final Knowledge Distillation Compression

### Task 9.1 — Choose the student architecture
- Select a smaller model:
  - ResNet-18
  - MobileNetV2
  - EfficientNet-lite option
- Keep the student significantly smaller than the teacher.

### Task 9.2 — Build the distillation pipeline
- Use the best CL model as teacher.
- Train the student on the final dataset/task stream output.

#### Subtasks
- Compute teacher logits.
- Implement KL divergence loss.
- Combine CE + KD loss.
- Tune alpha and temperature.
- Save best student checkpoint.

### Task 9.3 — Compare teacher and student
- Compare:
  - accuracy
  - inference time
  - number of parameters
  - storage size
- Quantify the compression-performance tradeoff.

### Task 9.4 — Optional teacher assistant
- If time permits, test:
  - Teacher = ResNet-50
  - Assistant = ResNet-34
  - Student = ResNet-18
- Check if intermediate distillation improves performance.

---

## Phase 10 — Experiments and Ablations

### Task 10.1 — Hyperparameter tuning
- Tune:
  - learning rate
  - batch size
  - replay memory size
  - EWC lambda
  - KD temperature
  - KD alpha
  - AL budget
- Keep tuning fair across methods.

### Task 10.2 — Memory budget ablation
- Test different replay sizes.
- See how memory affects forgetting.

### Task 10.3 — Task order ablation
- Try different class orders.
- Check whether results are robust.

### Task 10.4 — Base task size ablation
- Optionally compare:
  - base 10 classes
  - base 20 classes
- Check if stronger initial learning improves CL.

### Task 10.5 — Active learning ablation
- Study if AL really helps.
- Measure benefit per extra complexity added.

### Task 10.6 — Distillation ablation
- Compare:
  - CE only
  - KD only
  - CE + KD
- Compare different temperatures.

---

## Phase 11 — Documentation and Reporting

### Task 11.1 — Maintain experiment logs
- Keep a record of every run.
- Save:
  - config
  - checkpoint path
  - metrics
  - notes

### Task 11.2 — Write method descriptions
- For each method, document:
  - intuition
  - implementation details
  - hyperparameters
  - strengths and weaknesses

### Task 11.3 — Prepare the final report
- Introduction
- Related work
- Methodology
- Dataset setup
- Task split design
- Experimental setup
- Results
- Discussion
- Limitations
- Future work

### Task 11.4 — Prepare visuals
- Pipeline diagram
- Task split figure
- Accuracy curves
- Forgetting curves
- KD compression chart
- AL selection workflow

### Task 11.5 — Prepare presentation/demo
- Summarize:
  - problem
  - pipeline
  - compared methods
  - key results
  - best model
  - distilled student model

---

## Phase 12 — Final Cleanup

### Task 12.1 — Clean repository
- Remove unused scripts.
- Remove duplicate notebooks.
- Organize configs and outputs.

### Task 12.2 — Verify reproducibility
- Run the full pipeline from config files.
- Confirm results can be regenerated.

### Task 12.3 — Final checklist
- Dataset prepared
- Tasks fixed
- Backbone selected
- All CL methods implemented
- Active Learning added or clearly excluded
- Best method selected
- Final KD completed
- Results plotted
- Report written
- Code cleaned