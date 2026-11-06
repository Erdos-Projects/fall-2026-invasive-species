# Notebooks

The notebooks are the record of how the project got to its final models. They were run on Google Colab against the
team's Google Drive, and they are kept with their outputs (tables, plots, metrics) as evidence for each decision. The
reproducible pipeline is the Python package in `src/invasive_clf/`; nothing there imports these notebooks.

They are numbered in reading order. Each notebook starts with a short **In this repository** note saying what it was
for. To re-run one, point its data path at the image folder made by `src/invasive_clf/data/download_images.py` (see
`data/README.md`).

## `eda/`: preliminary work

| notebook | what it does |
|---|---|
| `01_preliminary_inat_dataset_builder_top_species.ipynb` | An early dataset builder (native vs. introduced species, top species by observation count). Replaced by `src/invasive_clf/data/build_dataset.py`. |
| `02_preliminary_dataset_EDA.ipynb` | Exploratory analysis of an early 10,000-photo sample: labels, species frequencies, image sizes and aspect ratios. |

## `data/`: species lists behind the dataset

| notebook | what it does |
|---|---|
| `03_build_species_lists.ipynb` | The published invasive and noxious species lists (entered by hand) and the labeling rule. |
| `04_inat_counts.ipynb` | iNaturalist taxon ids and observation counts for those species. Together with 03, makes `data/reference/`. |

## `modeling_experiments/`: model search on the validation set

Notebooks 05 to 18 train on `train.csv` and are scored on `val.csv`; none of them reads `test.csv` or `test_unseen.csv`.
Notebook 19 is the only one that touches the test sets, once, after the two final models were chosen.

Notebooks 05 to 13 fine-tune one pretrained backbone each. They are copies of one shared template: the same data,
model, metric and training code, with a manual accept/reject log (`decisions.json`) for each change tried.

| notebook | what it does |
|---|---|
| `05_experiments_resnet18.ipynb` … `09_experiments_resnet152.ipynb` | The ResNet family: ResNet-18, 34, 50, 101 and 152. |
| `10_experiments_ConvNeXt_tiny.ipynb` | ConvNeXt-Tiny. |
| `11_experiments_dinov3_vits.ipynb` | DINOv3 ViT-S/16. Its run `04_pool_cls_patchmean` is the **final fine-tuned model**. |
| `12_experiments_plantnet_dinov3_vits.ipynb` | DINOv3 ViT-S/16 fine-tuned by Pl@ntNet on plant photos. |
| `13_experiments_SigLIP2ViT-B16.ipynb` | SigLIP 2 ViT-B/16 image tower. |
| `14_experiments_bioclip2_zeroshot_final.ipynb` | BioCLIP 2 (ViT-L/14) zero-shot, with no training. |
| `15_experiments_bioclip2_robustness_and_probe.ipynb` | How robust the zero-shot result is (unknown species, larger vocabulary), and whether light training helps. |
| `16_experiments_bioclip2_final_candidates.ipynb` | BioCLIP 2 candidates written in the common run format. `bioclip2_zeroshot_full` is the **final BioCLIP 2 model**. |
| `17_logreg_baselines.ipynb` | Baselines: logistic regression on the frozen features of each backbone. |
| `18_final_model_comparison.ipynb` | Compares every run on val and picks the two models that go to the test set. |
| `19_final_evaluation.ipynb` | The one-shot test: retrains the DINOv3 model on train + val, scores `test.csv` and `test_unseen.csv` with both final models, and times them. |
