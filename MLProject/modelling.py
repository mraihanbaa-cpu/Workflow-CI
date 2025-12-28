import os
import glob
import shutil
import logging
import argparse
import sys

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

import matplotlib.pyplot as plt


# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------
# DATA HELPERS
# --------------------------------------------------
def find_books_csv():
    """
    Find dataset file in common locations or recursively.
    Expected file: top_1000_books_preprocessing.csv
    """
    candidates = [
        "top_1000_books_preprocessing.csv",
        os.path.join("data", "top_1000_books_preprocessing.csv"),
        os.path.join("dataset", "top_1000_books_preprocessing.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    matches = glob.glob("**/top_1000_books_preprocessing.csv", recursive=True)
    if matches:
        return matches[0]

    # fallback: any csv that contains "books" and "preprocess"
    matches2 = glob.glob("**/*books*preprocess*.csv", recursive=True)
    if matches2:
        return matches2[0]

    raise FileNotFoundError(
        "Dataset not found. Ensure 'top_1000_books_preprocessing.csv' exists "
        "in project root (or provide --data_path)."
    )


# --------------------------------------------------
# MLFLOW HELPERS
# --------------------------------------------------
def setup_mlflow_tracking(experiment_name: str = "books-ml-project"):
    """
    Set MLflow tracking to LOCAL filesystem by default.
    - In CI: you may set MLFLOW_TRACKING_URI (recommended).
    - Locally: fallback to ./mlruns inside this script directory.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        mlflow.set_tracking_uri(env_uri)
        logging.info("Using MLFLOW_TRACKING_URI from env: %s", env_uri)
    else:
        tracking_dir = os.path.join(os.path.dirname(__file__), "mlruns")
        os.makedirs(tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{tracking_dir}")
        logging.info("Using local tracking dir: %s", tracking_dir)

    mlflow.set_experiment(experiment_name)


def ensure_active_run() -> bool:
    """
    Ensure there is an active MLflow run.
    Returns:
      already_active (bool): True if there was already an active run,
      False if we had to start a new run.
    """
    if mlflow.active_run() is not None:
        return True

    try:
        mlflow.start_run()
        return False
    except Exception:
        # Some environments can have MLFLOW_RUN_ID set but not resolvable; make it robust
        os.environ.pop("MLFLOW_RUN_ID", None)
        mlflow.start_run()
        return False


# --------------------------------------------------
# CORE
# --------------------------------------------------
def evaluate_and_log(model, X_test, y_test, out_dir: str):
    """
    Evaluate model, log metrics + evaluation artifacts.
    """
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="binary", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)
    mlflow.log_metric("test_f1", f1)

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba))
            mlflow.log_metric("test_roc_auc", roc_auc)
        except Exception:
            roc_auc = None

    logging.info("Metrics: acc=%.4f prec=%.4f rec=%.4f f1=%.4f", acc, prec, rec, f1)
    if roc_auc is not None:
        logging.info("Metrics: roc_auc=%.4f", roc_auc)

    # ---- artifacts
    os.makedirs(out_dir, exist_ok=True)

    # classification report
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, digits=4, zero_division=0))
    mlflow.log_artifact(report_path, artifact_path="evaluation")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix (Test Set)")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path="evaluation")

    # roc curve (optional)
    if y_proba is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("ROC Curve (Test Set)")
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close(fig)
        mlflow.log_artifact(roc_path, artifact_path="evaluation")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(
    n_estimators=200,
    max_depth=None,
    test_size=0.2,
    random_state=42,
    data_path=None,
    out_dir="artifacts",
    experiment_name="books-ml-project",
    run_name="RandomForest_LocalMLflow",
):
    os.makedirs(out_dir, exist_ok=True)

    # ✅ LOCAL MLflow tracking (submission-safe: NO DagsHub)
    setup_mlflow_tracking(experiment_name=experiment_name)

    already_active = ensure_active_run()

    try:
        # Resolve dataset path
        if data_path:
            if not os.path.exists(data_path):
                alt = os.path.join(os.path.dirname(__file__), data_path)
                if os.path.exists(alt):
                    data_path = alt
                else:
                    raise FileNotFoundError(f"Dataset not found: {data_path}")
            logging.info("Using dataset: %s", data_path)
        else:
            data_path = find_books_csv()
            logging.info("Auto dataset: %s", data_path)

        df = pd.read_csv(data_path)

        if "bestseller_status" not in df.columns:
            raise ValueError("Target column 'bestseller_status' not found in dataset")

        X = df.drop(columns=["bestseller_status"])
        y = df["bestseller_status"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # set run name if not already started by wrapper
        if mlflow.active_run() is not None and mlflow.active_run().info.run_name is None:
            mlflow.set_tag("mlflow.runName", run_name)

        # Params
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", int(n_estimators))
        mlflow.log_param("max_depth", None if max_depth in (None, "None") else int(max_depth))
        mlflow.log_param("test_size", float(test_size))
        mlflow.log_param("random_state", int(random_state))

        # Autolog ON (sklearn)
        mlflow.sklearn.autolog(log_models=True)

        # Train
        logging.info("Training RandomForest...")
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth in (None, "None") else int(max_depth),
            random_state=int(random_state),
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate + log
        evaluate_and_log(model, X_test, y_test, out_dir=out_dir)

        # Save best model to artifacts/model (MLflow model format)
        best_model_dir = os.path.join(out_dir, "model")
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        mlflow.sklearn.save_model(model, best_model_dir)

        # sanity check
        if not os.path.exists(os.path.join(best_model_dir, "MLmodel")):
            raise RuntimeError("MLmodel not created")

        mlflow.log_artifacts(best_model_dir, artifact_path="model")
        logging.info("Model saved to %s and logged as MLflow artifact.", best_model_dir)

    finally:
        # Only end run if we started it
        if not already_active:
            mlflow.end_run()


# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--experiment_name", type=str, default="books-ml-project")
    parser.add_argument("--run_name", type=str, default="RandomForest_LocalMLflow")
    args = parser.parse_args()

    try:
        main(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            test_size=args.test_size,
            random_state=args.random_state,
            data_path=args.data_path,
            out_dir=args.out_dir,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
# --------------------------------------------------