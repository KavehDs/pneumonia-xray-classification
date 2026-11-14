# Pneumonia Detection from Chest X-rays — PneumoniaMNIST

🚀 **Live demo:** [Try the app](https://pfdubaypbxrm8zkk7tjv9t.streamlit.app/)

Binary classification of paediatric chest X-rays: **pneumonia vs normal**,
using the PneumoniaMNIST benchmark from MedMNIST v2. Two models are compared:
a Logistic Regression baseline on flattened pixels and a small CNN in PyTorch.

> ⚠️ **Educational demo only — not a medical diagnosis tool.** The model was
> trained on 28x28 downscaled images of children aged 1–5. Never use it for
> real clinical decisions.

## Dataset

- **PneumoniaMNIST** (MedMNIST v2) — 5,856 paediatric chest X-ray images,
  28x28 grayscale, from Guangzhou Women and Children's Medical Center.
- Official splits used as-is: **train 4,708 / validation 524 / test 624**.
- Class balance: train 1,214 normal / 3,494 pneumonia (74.2% pneumonia);
  test 234 normal / 390 pneumonia (62.5% pneumonia).
- **Licence: CC BY 4.0.** Source: `pneumoniamnist.npz` from
  [Zenodo (MedMNIST v2)](https://zenodo.org/records/5208230).
- The full 4.2 MB `.npz` is downloaded at runtime by `src/train.py` and by
  the demo; a 200-row sample (`data/sample.csv`, 100 normal / 100 pneumonia)
  is bundled as a fallback if the download fails.
- **Note:** 28x28 is a heavily downscaled benchmark version of the original
  X-rays (originally 384–2,916 px wide). Fine detail that a radiologist would
  use is lost at this resolution — that is a deliberate limitation of the
  MedMNIST benchmark, not of this project.

## Citation

- Kermany, D. S., Goldbaum, M., et al. "Identifying Medical Diagnoses and
  Treatable Diseases by Image-Based Deep Learning." *Cell*, 2018.
- Yang, J., et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D
  and 3D Biomedical Image Classification." *Scientific Data*, 2023.
  https://medmnist.com

## Results (test set, n = 624, positive = pneumonia)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.8542 | 0.8174 | 0.9872 | 0.8943 | 0.9205 |
| **CNN (2 conv blocks)** | **0.8381** | 0.7979 | **0.9923** | 0.8846 | **0.9310** |

Confusion matrices (test set):

- Logistic Regression: TN 148, FP 86, FN 5, TP 385
- CNN: TN 136, FP 98, FN 3, TP 387

Training: Logistic Regression < 1 s; CNN 12 epochs, ~51–63 s on a 1.5-CPU /
6 GB machine (best validation accuracy 0.9656, epoch 10).

All numbers are reproduced by re-running `src/train.py` (seed 42);
see `metrics.json`.

## Honest findings

- **The CNN only beats the baseline on ROC-AUC (0.931 vs 0.921).** On
  accuracy and F1 the simple Logistic Regression is better (0.854 / 0.894
  vs 0.838 / 0.885). At 28x28 resolution there is not much spatial detail
  for a CNN to exploit — the pixels themselves carry most of the signal.
- **Both models over-flag normal cases.** The CNN calls 98 of 234 normal
  X-rays pneumonia (42%); the baseline calls 86. Precision is only ~0.80,
  so roughly 1 in 5 "pneumonia" predictions is wrong.
- **The CNN misses very few pneumonia cases (3 of 390, recall 0.992),**
  but the 3 it misses it misses with low confidence (mean predicted
  pneumonia probability 0.258) — borderline cases, not confident errors.
  The 98 false positives are the opposite: confidently wrong
  (mean probability 0.855).
- **Class imbalance matters.** With 74% pneumonia in training, a model
  that always predicted pneumonia would already score 62.5% accuracy on
  the test set. Accuracy alone is misleading here; recall and precision
  are the metrics that matter.
- **Paediatric data only** (children aged 1–5, single centre in Guangzhou).
  Nothing here says anything about adult chest X-rays.
- **No external validation** and a single train/test split — results could
  shift on another hospital's data.
- The sample grid and confusion matrix are in `plots/`.

## How to run

```bash
pip install -r requirements.txt
python src/train.py          # trains both models, writes metrics.json + plots
streamlit run app.py         # demo: pick a test X-ray or upload your own
```

The demo trains the CNN once at startup (~1 minute on CPU) and caches it.
Model weights (`models/cnn_model.pt`, 1.7 MB) are saved locally by training
but are not part of the repo — the demo retrains from the same seed instead.

## Files

- `src/train.py` — training + evaluation (seed 42, reproducible)
- `app.py` — Streamlit demo
- `metrics.json` — all measured results
- `plots/` — ROC curve, confusion matrix, sample image grid, training curve (SVG)
- `data/sample.csv` — 200-row fallback sample (full data downloads from Zenodo)
