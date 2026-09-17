"""
train_baseline.py
------------------
Pipeline base (baseline) para el proyecto de reconocimiento de actividad
humana (HAR). Entrena un clasificador clásico sobre las 561 features
pre-extraídas del UCI HAR Dataset y reporta las métricas de ML definidas
en la propuesta del proyecto (accuracy, precision/recall/F1 por clase,
matriz de confusión).

Este script es un punto de partida (línea base) para comparar contra
modelos más complejos (p. ej. arquitecturas DRNN/LSTM como en Murad & Pyun,
2017) en etapas posteriores del proyecto.

Uso:
    python src/train_baseline.py --model random_forest
    python src/train_baseline.py --model svm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC

from data_loader import ACTIVITY_NAMES, load_features

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

MODELS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    ),
    "svm": lambda: SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
}

CLASS_ORDER = [ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES)]


def run(model_name: str) -> dict:
    print(f"Cargando dataset (features de 561 dimensiones)...")
    X_train, y_train, _ = load_features("train")
    X_test, y_test, _ = load_features("test")

    print(f"Entrenando modelo: {model_name}")
    model = MODELS[model_name]()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    n_test = len(X_test)
    inference_time = time.time() - t0
    latency_ms_per_window = (inference_time / n_test) * 1000

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time_sec": train_time,
        "inference_latency_ms_per_window": latency_ms_per_window,
        "n_train": len(X_train),
        "n_test": n_test,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_order": CLASS_ORDER,
    }

    print("\n--- Resultados ---")
    print(f"Accuracy global:      {acc:.4f}")
    print(f"F1 macro:             {f1_macro:.4f}")
    print(f"F1 ponderado:         {f1_weighted:.4f}")
    print(f"Tiempo entrenamiento: {train_time:.2f} s")
    print(f"Latencia inferencia:  {latency_ms_per_window:.4f} ms/ventana")
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print(CLASS_ORDER)
    print(cm)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"metrics_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nMétricas guardadas en {out_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena una línea base para HAR sobre UCI HAR Dataset")
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default="random_forest",
        help="Modelo a entrenar (default: random_forest)",
    )
    args = parser.parse_args()
    run(args.model)
