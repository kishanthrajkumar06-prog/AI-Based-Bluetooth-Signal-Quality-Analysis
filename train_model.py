"""Train and evaluate the Bluetooth link-quality classification model."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "bluetooth_signal_dataset.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
NUMERIC = ["distance_m", "rssi_dbm", "packet_loss_pct", "latency_ms", "obstacles"]
CATEGORICAL = ["interference", "connection_status"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "quality"

def main():
    if not DATA.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_dataset.py")
    df = pd.read_csv(DATA).dropna(subset=FEATURES + [TARGET])
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "bluetooth_quality_model.joblib")
    metrics = {
        "dataset_type": "synthetic demonstration data",
        "rows": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy), 4), "macro_f1": round(float(macro_f1), 4),
        "classification_report": report,
        "note": "Metrics on synthetic data do not prove real-world Bluetooth performance."
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Synthetic Demo Dataset")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["good", "moderate", "poor"]
    df["quality"].value_counts().reindex(order, fill_value=0).plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution — Synthetic Demo Dataset")
    ax.set_xlabel("Predicted/assigned quality class")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Saved model to {MODEL_DIR / 'bluetooth_quality_model.joblib'}")
    print(f"Saved metrics and plots to {OUT_DIR}")

if __name__ == "__main__":
    main()
