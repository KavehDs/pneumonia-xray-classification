"""PneumoniaMNIST — chest X-ray classification (binary: pneumonia vs normal).

Compares:
  1. Logistic Regression on flattened pixels (baseline)
  2. A small CNN in PyTorch (2 conv blocks + dropout)

Dataset: PneumoniaMNIST (MedMNIST v2), 5,856 paediatric chest X-rays,
28x28 grayscale. Official train/val/test splits are used as-is.
Source: https://zenodo.org/records/5208230 (pneumoniamnist.npz, ~4.2 MB)
Licence: CC BY 4.0.

The .npz is downloaded at runtime if missing; if the download fails it
falls back to the bundled sample CSV (data/sample.csv, 200 rows).
Reproducible: random_state / seed = 42 everywhere.

Outputs: metrics.json, plots/*.svg, models/cnn_model.pt
"""

import json
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "pneumoniamnist.npz"
SAMPLE = BASE / "data" / "sample.csv"
URL = "https://zenodo.org/records/5208230/files/pneumoniamnist.npz?download=1"
PLOTS = BASE / "plots"
MODELS = BASE / "models"
METRICS = BASE / "metrics.json"

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


def load_data():
    """Return (X_train, y_train, X_val, y_val, X_test, y_test, source)."""
    if not DATA.exists():
        print(f"Downloading {URL} ...")
        DATA.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(URL, DATA)
        except Exception as exc:  # noqa: BLE001 - fall back to sample
            print(f"Download failed ({exc}); falling back to bundled sample.")
            return load_sample()
    d = np.load(DATA)
    return (
        d["train_images"],
        d["train_labels"].ravel(),
        d["val_images"],
        d["val_labels"].ravel(),
        d["test_images"],
        d["test_labels"].ravel(),
        "full pneumoniamnist.npz (Zenodo, MedMNIST v2)",
    )


def load_sample():
    """Bundled 200-row fallback (100 normal / 100 pneumonia, flattened)."""
    import csv

    X, y = [], []
    with open(SAMPLE, newline="") as f:
        for row in csv.reader(f):
            y.append(int(row[0]))
            X.append([int(v) for v in row[1:]])
    X = np.array(X, dtype=np.uint8).reshape(-1, 28, 28)
    y = np.array(y)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(y))
    n_test, n_val = 40, 40
    return (
        X[idx[n_test + n_val:]],
        y[idx[n_test + n_val:]],
        X[idx[n_test:n_test + n_val]],
        y[idx[n_test:n_test + n_val]],
        X[idx[:n_test]],
        y[idx[:n_test]],
        "bundled sample.csv (200 rows) — download failed",
    )


def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_cnn(X_tr, y_tr, X_val, y_val, epochs=12, batch=64):
    torch.manual_seed(SEED)
    Xtr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1) / 255.0
    ytr = torch.tensor(y_tr, dtype=torch.long)
    Xva = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1) / 255.0
    yva = torch.tensor(y_val, dtype=torch.long)

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr)
    history = {"train_acc": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch))
        correct = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            correct += int((out.argmax(1) == yb).sum())
        model.eval()
        with torch.no_grad():
            vacc = float((model(Xva).argmax(1) == yva).float().mean())
        history["train_acc"].append(round(correct / n, 4))
        history["val_acc"].append(round(vacc, 4))
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch + 1}/{epochs} train_acc={correct / n:.4f} val_acc={vacc:.4f}")
    train_seconds = round(time.time() - t0, 1)
    model.load_state_dict(best_state)
    return model, history, train_seconds, best_acc


def plot_roc(y_test, probs, path):
    plt.figure(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.title("ROC — PneumoniaMNIST test set (positive = pneumonia)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    mat = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i][j], ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted normal", "Predicted pneumonia"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True normal", "True pneumonia"])
    ax.set_title("Confusion matrix — CNN, test set (n=624)")
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def plot_sample_grid(X_test, y_test, path):
    rng = np.random.RandomState(SEED)
    norm_idx = rng.choice(np.where(y_test == 0)[0], 4, replace=False)
    pneu_idx = rng.choice(np.where(y_test == 1)[0], 4, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for ax, idx, label in zip(
        axes[0], norm_idx, ["Normal"] * 4
    ):
        ax.imshow(X_test[idx], cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    for ax, idx, label in zip(axes[1], pneu_idx, ["Pneumonia"] * 4):
        ax.imshow(X_test[idx], cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Sample test images — PneumoniaMNIST (28x28, downscaled)")
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def plot_training_curve(history, path):
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_acc"], marker="s", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr, X_val, y_val, X_test, y_test, source = load_data()
    print(f"Data: {source}")
    print(f"train={len(y_tr)} val={len(y_val)} test={len(y_test)}")

    # --- 1. Logistic Regression baseline on flattened pixels ---
    t0 = time.time()
    Xtr_f = (X_tr.reshape(len(X_tr), -1) / 255.0).astype(np.float32)
    Xte_f = (X_test.reshape(len(X_test), -1) / 255.0).astype(np.float32)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                            random_state=SEED)
    lr.fit(scaler.fit_transform(Xtr_f), y_tr)
    lr_prob = lr.predict_proba(scaler.transform(Xte_f))[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    lr_metrics = evaluate(y_test, lr_pred, lr_prob)
    lr_seconds = round(time.time() - t0, 1)
    print("Logistic Regression:", lr_metrics, f"({lr_seconds}s)")

    # --- 2. Small CNN ---
    cnn, history, cnn_seconds, best_val = train_cnn(X_tr, y_tr, X_val, y_val)
    cnn.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
        logits = cnn(Xte_t)
        cnn_prob = torch.softmax(logits, dim=1)[:, 1].numpy()
    cnn_pred = (cnn_prob >= 0.5).astype(int)
    cnn_metrics = evaluate(y_test, cnn_pred, cnn_prob)
    print("CNN:", cnn_metrics, f"({cnn_seconds}s)")

    torch.save(cnn.state_dict(), MODELS / "cnn_model.pt")

    # --- error pattern: where does the CNN fail? ---
    fn_idx = np.where((y_test == 1) & (cnn_pred == 0))[0]
    fp_idx = np.where((y_test == 0) & (cnn_pred == 1))[0]
    error_patterns = {
        "false_negatives_missed_pneumonia": int(len(fn_idx)),
        "false_positives_normal_flagged": int(len(fp_idx)),
        "fn_mean_prob_pneumonia": round(float(cnn_prob[fn_idx].mean()), 3) if len(fn_idx) else None,
        "fp_mean_prob_pneumonia": round(float(cnn_prob[fp_idx].mean()), 3) if len(fp_idx) else None,
    }

    metrics = {
        "dataset": {
            "name": "PneumoniaMNIST (MedMNIST v2)",
            "source": source,
            "licence": "CC BY 4.0",
            "total": 5856,
            "train": {"n": int(len(y_tr)), "normal": int((y_tr == 0).sum()),
                      "pneumonia": int((y_tr == 1).sum())},
            "val": {"n": int(len(y_val)), "normal": int((y_val == 0).sum()),
                    "pneumonia": int((y_val == 1).sum())},
            "test": {"n": int(len(y_test)), "normal": int((y_test == 0).sum()),
                     "pneumonia": int((y_test == 1).sum())},
            "image_size": "28x28 grayscale (downscaled benchmark version)",
            "population": "paediatric chest X-rays (children aged 1-5, Guangzhou)",
        },
        "models": {
            "logistic_regression": {**lr_metrics, "train_seconds": lr_seconds,
                                    "note": "flattened 784 pixels, StandardScaler, class_weight=balanced"},
            "cnn": {**cnn_metrics, "train_seconds": cnn_seconds,
                    "epochs": 12, "batch_size": 64, "best_val_accuracy": round(best_val, 4),
                    "note": "2 conv blocks (32/64 filters), dropout 0.25/0.5, Adam lr=0.001"},
        },
        "cnn_training_history": history,
        "error_patterns": error_patterns,
        "positive_class": "pneumonia (label 1)",
        "seed": SEED,
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(y_test, {"Logistic Regression": lr_prob, "CNN": cnn_prob},
             PLOTS / "roc_curve.svg")
    plot_confusion(cnn_metrics["confusion_matrix"], PLOTS / "confusion_matrix.svg")
    plot_sample_grid(X_test, y_test, PLOTS / "sample_grid.svg")
    plot_training_curve(history, PLOTS / "training_curve.svg")
    print("Wrote metrics.json, models/cnn_model.pt, plots/*.svg")


if __name__ == "__main__":
    main()
