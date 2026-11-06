# Invasive Plant Identification from Citizen-Science Photos
A fine-tuned vision model that flags invasive plant species in iNaturalist photos from the Desert Southwest.

**[Erdos Institute](https://www.erdosinstitute.org) Deep Learning Project, Fall 2026**

**Team: Craig Corsi, Jay Desai, Rachel Neville, and Apostolos Zournas**
---

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
│   ├── base.yaml             # initial configuration
│   └── siglip2_base_final.yaml
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
