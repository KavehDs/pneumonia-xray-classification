"""PneumoniaMNIST demo — chest X-ray classification (pneumonia vs normal).

The CNN is trained at first startup (12 epochs on 4,708 28x28 images,
~1 minute on CPU) and cached afterwards. The dataset is downloaded from
Zenodo on first run if it is not already present locally.
"""

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "pneumoniamnist.npz"
URL = "https://zenodo.org/records/5208230/files/pneumoniamnist.npz?download=1"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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


@st.cache_resource(show_spinner=False)
def load_data():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    d = np.load(DATA)
    return (
        d["train_images"],
        d["train_labels"].ravel(),
        d["val_images"],
        d["val_labels"].ravel(),
        d["test_images"],
        d["test_labels"].ravel(),
    )


@st.cache_resource(show_spinner=False)
def get_model(_X_tr, _y_tr, _X_val, _y_val):
    """Train the small CNN once (same config as src/train.py), then cache it."""
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xtr = torch.tensor(_X_tr, dtype=torch.float32).unsqueeze(1) / 255.0
    ytr = torch.tensor(_y_tr, dtype=torch.long)
    Xva = torch.tensor(_X_val, dtype=torch.float32).unsqueeze(1) / 255.0
    yva = torch.tensor(_y_val, dtype=torch.long)
    n = len(Xtr)
    best_acc, best_state = -1.0, None
    for epoch in range(12):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch))
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vacc = float((model(Xva).argmax(1) == yva).float().mean())
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, img28):
    """img28: 28x28 uint8 array -> (label, prob_pneumonia)."""
    with torch.no_grad():
        x = torch.tensor(img28, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        prob = float(torch.softmax(model(x), dim=1)[0, 1])
    return ("Pneumonia" if prob >= 0.5 else "Normal"), prob


st.set_page_config(page_title="Pneumonia Detection — Chest X-ray", layout="centered")
st.title("Pneumonia Detection from Chest X-rays")
st.markdown(
    """<style>
    [data-testid="stSlider"] { direction: ltr !important; }
    </style>""",
    unsafe_allow_html=True,
)
st.warning(
    "⚠️ Educational demo only — not a medical diagnosis tool. "
    "The model was trained on 28x28 downscaled paediatric X-rays; "
    "never use it for real clinical decisions."
)

with st.spinner("Loading data and training the CNN (first run takes ~1 minute)..."):
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_data()
    model = get_model(X_tr, y_tr, X_val, y_val)
st.success("Model ready.")

tab_test, tab_upload = st.tabs(["Pick a test X-ray", "Upload your own"])

with tab_test:
    idx = st.slider("Test image #", 0, len(X_test) - 1, 0)
    img = X_test[idx]
    true_label = "Pneumonia" if y_test[idx] == 1 else "Normal"
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(Image.fromarray(img).resize((280, 280), Image.NEAREST),
                 caption=f"Test image #{idx} — true label: {true_label}")
    with col2:
        pred, prob = predict(model, img)
        st.metric("Prediction", pred)
        st.metric("Pneumonia probability", f"{prob:.1%}")
        st.progress(prob)
        if pred == true_label:
            st.success("✅ Correct prediction")
        else:
            st.error("❌ Wrong prediction — the model can miss cases; see README")

with tab_upload:
    st.write("Upload a chest X-ray (any image format). It is resized to 28x28 "
             "grayscale, exactly like the training data.")
    up = st.file_uploader("Chest X-ray image", type=["png", "jpg", "jpeg", "bmp"])
    if up is not None:
        pil = Image.open(io.BytesIO(up.read())).convert("L").resize((28, 28))
        arr = np.array(pil, dtype=np.uint8)
        pred, prob = predict(model, arr)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil.resize((280, 280), Image.NEAREST), caption="Uploaded (28x28)")
        with col2:
            st.metric("Prediction", pred)
            st.metric("Pneumonia probability", f"{prob:.1%}")
            st.progress(prob)

st.subheader("Measured test-set performance (n = 624)")
try:
    with open(BASE / "metrics.json") as f:
        m = json.load(f)["models"]
    st.write(
        f"- **CNN** — Accuracy {m['cnn']['accuracy']:.4f} · Precision "
        f"{m['cnn']['precision']:.4f} · Recall {m['cnn']['recall']:.4f} · "
        f"F1 {m['cnn']['f1']:.4f} · ROC-AUC {m['cnn']['roc_auc']:.4f}\n"
        f"- **Logistic Regression** — Accuracy {m['logistic_regression']['accuracy']:.4f} · "
        f"Precision {m['logistic_regression']['precision']:.4f} · Recall "
        f"{m['logistic_regression']['recall']:.4f} · F1 {m['logistic_regression']['f1']:.4f} · "
        f"ROC-AUC {m['logistic_regression']['roc_auc']:.4f}"
    )
except FileNotFoundError:
    st.write("metrics.json not found in the repo.")
st.caption("Dataset: PneumoniaMNIST (MedMNIST v2), CC BY 4.0. "
           "Source: Kermany et al., Cell 2018; Yang et al., MedMNIST v2, 2023.")
