# Notebooks

The notebooks are the record of how the project got to its final models. They were run on Google Colab against the
team's Google Drive, and they are kept with their outputs (tables, plots, metrics) as evidence for each decision. The
reproducible pipeline is the Python package in `src/invasive_clf/`; nothing there imports these notebooks.

Each notebook starts with a short **In this repository** note saying what it was for. To re-run one, point its data
path at the image folder made by `src/invasive_clf/data/download_images.py` (see `data/README.md`).

## `data/`: species lists behind the dataset

| notebook | what it does |
|---|---|
| `01_build_species_lists.ipynb` | The published invasive and noxious species lists (entered by hand) and the labeling rule. |
| `02_inat_counts.ipynb` | iNaturalist taxon ids and observation counts for those species. Together with 01, makes `data/reference/`. |

## `eda/`: preliminary work

| notebook | what it does |
|---|---|
| `preliminary_inat_dataset_builder_top_species.ipynb` | An early dataset builder (native vs. introduced species, top species by observation count). Replaced by `src/invasive_clf/data/build_dataset.py`. |
| `preliminary_dataset_EDA.ipynb` | Exploratory analysis of an early 10,000-photo sample: labels, species frequencies, image sizes and aspect ratios. |

## `modeling_experiments/`: model search on the validation set

All of these train on `train.csv` and are scored on `val.csv`. None of them reads `test.csv` or `test_unseen.csv`.

| notebook | what it does |
|---|---|
| `experiments_TEMPLATE.ipynb` | The shared template for fine-tuning one pretrained backbone, with a manual accept/reject log (`decisions.json`). |
| `experiments_dinov3_vits.ipynb` | DINOv3 ViT-S/16. Its run `04_pool_cls_patchmean` is the **final fine-tuned model**. |
| `experiments_plantnet_dinov3_vits.ipynb` | DINOv3 ViT-S/16 fine-tuned by Pl@ntNet on plant photos. |
| `experiments_SigLIP2ViT-B16.ipynb` | SigLIP 2 ViT-B/16 image tower. |
| `experiments_ConvNeXt_tiny.ipynb` | ConvNeXt-Tiny. |
| `experiments_resnet18.ipynb`, `…34`, `…50`, `…101`, `…152` | The ResNet family. |
| `logreg_baselines.ipynb` | Baselines: logistic regression on the frozen features of each backbone. |
| `04_bioclip2_zeroshot_final.ipynb` | BioCLIP 2 (ViT-L/14) zero-shot, with no training. |
| `05_bioclip2_robustness_and_probe.ipynb` | How robust the zero-shot result is (unknown species, larger vocabulary), and whether light training helps. |
| `06_bioclip2_final_candidates.ipynb` | BioCLIP 2 candidates written in the common run format. `bioclip2_zeroshot_full` is the **final BioCLIP 2 model**. |
| `final_model_comparison.ipynb` | Compares every run on val and picks the two models that go to the test set. |
