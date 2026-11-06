# Results

Small outputs of the final model selection and the test evaluation, committed so the results can be read without
re-running anything. They were written on Google Colab by the notebooks named below; the paths inside the JSON files
point to the team's Google Drive.

The trained DINOv3 weights (`final.pt`, 87 MB) are not in git; see the main README for the download link.

## `final_selection/`: how the two final models were chosen (validation set only)

Written by `notebooks/modeling_experiments/18_final_model_comparison.ipynb`. Every number here is a **validation**
metric (`val.csv`); the test sets were not read.

| file | content |
|---|---|
| `final_models.json` | The two chosen models, their hyperparameters, validation metrics and thresholds, and the runner-up of each series. `src/invasive_clf/evaluate.py` reads the validation metrics from it for `test_vs_val.csv`. |
| `selected_models.csv`, `finalists.csv` | The two chosen models, one row each. |
| `final_finetune_config.json`, `final_bioclip2_config.json` | The complete configuration of each chosen run (`configs/dinov3_vits16.yaml` and `configs/bioclip2_zeroshot.yaml` are their counterparts in this repository). |
| `final_finetune_val_metrics.json`, `final_bioclip2_val_metrics.json` | Every validation metric of each chosen run, per class included. |
| `all_runs.csv` | Every run of every notebook (fine-tuning iterations, logistic-regression baselines, BioCLIP 2 candidates) with its decision and metrics. |
| `backbone_comparison.csv` | One row per pretrained backbone: size, baseline and best fine-tuned macro-F1, the frozen-feature baseline, number of iterations. |
| `series_candidates.csv`, `shortlist.csv`, `frozen_series.csv` | The candidates of each series and the shortlist the choice was made from. |
| `bootstrap_macro_f1.csv` | Bootstrap 95% intervals of macro-F1 for the shortlist, and of each candidate's difference from the chosen model. |
| `per_class_f1.csv`, `group_breakdown.csv`, `agreement_and_oracle.csv` | Per-class F1, error rates by kind of photo, and how often the shortlisted models agree. |
| `*.png` | The figures of notebook 18: every run as a point (`all_runs_scatter`), baseline vs fine-tuned (`probe_vs_finetune`, `finetune_gain`), the accept/reject trajectories (`iteration_trajectories_finetune`), the training curves of the chosen runs, the bootstrap intervals, the per-class F1 heatmap, false positives by group, and the BioCLIP 2 methods. |

## `final_test/`: the one-shot test evaluation

Written by `notebooks/modeling_experiments/19_final_evaluation.ipynb`; `python -m invasive_clf.evaluate` writes the same
files. Both models score `test.csv` and `test_unseen.csv` once, at the threshold tuned on validation (0.06 for
BioCLIP 2, 0.14 for DINOv3), with mirror-image averaging, on a Colab T4 GPU.

| file | content |
|---|---|
| `final_results.json` | Everything below in one file, per model: test metrics, validation metrics, threshold, timing, unseen-species result. |
| `test_metrics.csv` | Every test metric of both models. |
| `test_vs_val.csv` | The validation numbers the models were chosen on next to their test numbers. |
| `per_class_f1_test.csv/.png` | F1 of each invasive class on test, both models. |
| `pr_roc_test.png` | Binary "invasive or not": precision-recall and ROC curves of both models, with the operating point. |
| `errors_by_group_test.csv/.png` | Error rates for invasive photos (missed, wrong species) and false-positive rates for each kind of `not_invasive` photo. |
| `accuracy_vs_runtime.csv`, `timing.csv` | Accuracy next to model size, inference time per image and GPU memory. |
| `unseen_results.csv`, `unseen_top1.png` | `test_unseen.csv` (792 photos of 30 hold-out species, all `not_invasive`): how many each model flags as invasive. |
| `split_overview.csv/.png`, `class_counts.csv` | Photos per split and kind of photo, and per invasive class. |
| `bioclip2/`, `finetune/` | Per model: `test_predictions.csv` and `unseen_predictions.csv` (one row per photo), `test_probs.npz` (the 40 class probabilities per test photo), `metrics_test.json`, `config.json`, and the PR/ROC, per-class F1 and confusion-matrix figures. `finetune/` also has the training history and curves of the train + val run. |
