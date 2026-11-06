# Data

The images themselves are **not** in this repository. This folder holds the small tables that define the dataset
exactly, so anyone can download the same photos.

## `splits/` — the dataset manifest

| File | Rows | What it is |
|---|---|---|
| `train.csv` | 7,342 | training photos (any upload date) |
| `val.csv` | 3,228 | validation photos, uploaded on/after 2025-07-01; used for model selection and threshold tuning |
| `test.csv` | 7,931 | test photos, uploaded on/after 2025-07-01; used once, for the final results |
| `test_unseen.csv` | 792 | test photos of 30 background species the models never saw in training (expected answer: `not_invasive`) |
| `classes.csv` | 4,639 | every candidate class considered, with counts, labels and why each was kept or dropped |

One row per photo. Key columns: `image_path` (where the photo goes, relative to the dataset folder), `label` (the
invasive species' name, or `not_invasive`), `invasive`, `group` (finer category used in the analysis),
`scientific_name`, `photo_url`, `photo_license`, `photo_attribution`, `observation_url`.

Train / val / test contain 40 invasive species plus 401 other species labeled `not_invasive` (441 species in total).
Each photographer appears in only one of train / val / test, and val / test only contain photos uploaded on or after
2025-07-01, so the pretrained models we start from cannot have seen them.

These tables were produced by `src/invasive_clf/data/build_dataset.py` (settings in `configs/dataset.yaml`) on
2026-11-01.

## `reference/` — the species lists

`ranked_desert_sw.csv` and `ranked_full_bbox.csv`: invasive and noxious-weed species from published state, federal
and Cal-IPC lists, with their iNaturalist taxon ids and observation counts. `classifier_label` gives the starting label
(`invasive`, `noxious only`, `native, noxious`); `src/invasive_clf/data/label_policy.py` holds the team's exceptions.

They were made by two notebooks: `notebooks/data/03_build_species_lists.ipynb` (the published lists, entered by hand, and
the labeling rule) and `notebooks/data/04_inat_counts.ipynb` (iNaturalist taxon ids and observation counts).

## Getting the images

```bash
pip install -e .                               # once, from the repository root: makes invasive_clf importable
python -m invasive_clf.data.download_images
```

downloads every photo listed in `splits/` to `data/inat_dataset_final/images/` (about 19,300 photos at 500 px).
Photos removed from iNaturalist since 2026-11-01 are listed in `data/inat_dataset_final/download_failures.csv`.

## License of the photos

All photos are from iNaturalist, licensed CC0, CC BY or CC BY-NC by their photographers (86% are CC BY-NC). The
`photo_attribution` and `photo_license` columns give the credit each license requires. The photos are not
redistributed here; the download script fetches them from iNaturalist.
