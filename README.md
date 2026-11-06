# Invasive Plant Identification from Citizen-Science Photos
A fine-tuned vision model that flags invasive plant species in iNaturalist photos from the Desert Southwest.

**[Erdos Institute](https://www.erdosinstitute.org) Deep Learning Project, Fall 2026**

**Team: Craig Corsi, Jay Desai, Rachel Neville, and Apostolos Zournas**
---
<img width="506" height="332" alt="newplotinvasive_vis" src="https://github.com/user-attachments/assets/5d2fc700-d371-4b83-a320-8465df37852d" />

## Overview
**Problem.** Invasive plants degrade ecosystems across the Desert Southwest. They displace native vegetation, change fire regimes, and disrupt food websManagement costs rise sharply once an infestation is established, so identifying small, localized populations is a more cost effective management strategy. Field surveys cannot cover the land at the needed scale. Citizen-science platforms such as iNaturalist produce more observations. 

**Opportunity.** iNaturalist collects many community photos from volunteers. 

## Results at a glance


## Data
- **Source:**[iNaturalist](https://www.inaturalist.org/) research-grade observations, Desert Southwest region (place_id 118341: Arizona, New Mexico, eastern California).
- **Invasive species list:**
- **Classes:**
- **Splits:**

- The images are **not** included in this repository. See [Reproducing the results](#reproducing-the-results) for the download step.

## Method 

## Repository structure
```
.
├── README.md
├── EXPERIMENTS.md            # iteration-by-iteration log of accept/reject decisions
├── requirements.txt
├── configs/
│   └──  base.yaml           <img width="506" height="332" alt="newplotinvasive_vis" src="https://github.com/user-attachments/assets/987f973a-48f3-43ad-ae12-b473bcea46a5" />
<img width="506" height="332" alt="newplotinvasive_vis" src="https://github.com/user-attachments/assets/8e535dfd-e958-4fcc-9b37-b0e8a015ca02" />
<img width="506" height="332" alt="newplotinvasive_vis" src="https://github.com/user-attachments/assets/302daf12-f66b-40e1-947f-d8692e8226ef" />
 # initial configuration
├── notebooks/                # exploration and experiment logs (read in order)
│   ├── eda/                  # 01–02: preliminary dataset and exploratory analysis
│   ├── data/                 # 03–04: species lists behind the dataset
│   └── modeling_experiments/ # 05–18: model search on the validation set
├── src/invasive_clf/         # the pipeline
│   ├── data/                 # species lists, download, dataset + transforms
│   ├── models/               # backbones, classifier head
│   ├── metrics.py
│   ├── plots.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── experiments.py
└── results/                  # small committed outputs: metrics, figures, decisions.json
```

## Reproducing the results

Every step runs from the repository root. The model steps need a GPU (times below are for a Colab T4).

**1. Install.** Python 3.10 or later.
```bash
pip install -e .            # installs the invasive_clf package and its dependencies (see pyproject.toml)
```
The DINOv3 weights are gated on Hugging Face: accept the license on the
[model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), then log in with `hf auth login`
(or set `HF_TOKEN`). BioCLIP 2 needs no login.

**2. Download the photos** listed in `data/splits/` (about 19,300 photos; see [`data/README.md`](data/README.md)).
```bash
python -m invasive_clf.data.download_images
```
Rebuilding the split itself from iNaturalist is optional: `python -m invasive_clf.data.build_dataset`
(settings in `configs/dataset.yaml`).

**3. The fine-tuned model: DINOv3 ViT-S/16** (settings in `configs/dinov3_vits16.yaml`).
```bash
# optional: the validation run that chose the epoch count and the threshold (train only, scored on val; ~70 min)
python -m invasive_clf.train_dinov3 --config configs/dinov3_vits16.yaml --out-dir runs/dinov3_val --validate
# the final model: train + val, full schedule (~70 min) -> runs/dinov3_final/final.pt
python -m invasive_clf.train_dinov3 --config configs/dinov3_vits16.yaml --out-dir runs/dinov3_final
```

**4. The zero-shot model: BioCLIP 2 ViT-L/14** (settings in `configs/bioclip2_zeroshot.yaml`; nothing is trained).
```bash
# optional: score val and tune the threshold (~20 min)
python -m invasive_clf.build_bioclip2 --config configs/bioclip2_zeroshot.yaml --out-dir runs/bioclip2_val --validate
# the final model: the 4,573-name vocabulary and its text embeddings (a few minutes) -> runs/bioclip2_final/bioclip2_zeroshot.pt
python -m invasive_clf.build_bioclip2 --config configs/bioclip2_zeroshot.yaml --out-dir runs/bioclip2_final
```

**5. The test evaluation**: scores `test.csv` and `test_unseen.csv` once with both models at their validation
thresholds, times them, and writes every table and figure (~25 min).
```bash
python -m invasive_clf.evaluate --dinov3 runs/dinov3_final/final.pt \
    --bioclip2 runs/bioclip2_final/bioclip2_zeroshot.pt --out-dir runs/final_test
```
Either model can be left out. The validation-vs-test table reads the validation metrics in
`results/final_selection/final_models.json`.

**Checking the pipeline without a GPU.** Each step has a smoke test that runs on the CPU in about a minute, with tiny
untrained stand-in models and random photos (no downloads):
```bash
python tests/smoke_test_dinov3.py
python tests/smoke_test_bioclip2.py
python tests/smoke_test_evaluate.py
```
