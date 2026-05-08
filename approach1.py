import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "y_pred": y_pred
    }


def save_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading extracted MobileNetV1 features...")

    X_train = np.load("results/X_train_features.npy")
    y_train = np.load("results/y_train.npy")
    X_val = np.load("results/X_val_features.npy")
    y_val = np.load("results/y_val.npy")
    X_test = np.load("results/X_test_features.npy")
    y_test = np.load("results/y_test.npy")

    experiments = []

    logistic_c_values = [0.1, 1, 3, 10]
    svm_c_values = [0.1, 1, 3, 10]

    print("\nTuning Logistic Regression...")

    for c in logistic_c_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=c,
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1
            ))
        ])

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        val_results = evaluate_model(model, X_val, y_val)

        experiments.append({
            "Model": "Logistic Regression",
            "C": c,
            "Validation Accuracy": val_results["accuracy"],
            "Validation Precision": val_results["precision"],
            "Validation Recall": val_results["recall"],
            "Validation F1": val_results["f1"],
            "Training Time Seconds": train_time,
            "Pipeline": model
        })

        print(f"LR C={c} | Val Acc={val_results['accuracy']:.4f} | Val F1={val_results['f1']:.4f}")

    print("\nTuning Linear SVM...")

    for c in svm_c_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=c,
                max_iter=5000,
                dual="auto"
            ))
        ])

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        val_results = evaluate_model(model, X_val, y_val)

        experiments.append({
            "Model": "Linear SVM",
            "C": c,
            "Validation Accuracy": val_results["accuracy"],
            "Validation Precision": val_results["precision"],
            "Validation Recall": val_results["recall"],
            "Validation F1": val_results["f1"],
            "Training Time Seconds": train_time,
            "Pipeline": model
        })

        print(f"SVM C={c} | Val Acc={val_results['accuracy']:.4f} | Val F1={val_results['f1']:.4f}")

    best_experiment = max(experiments, key=lambda x: x["Validation F1"])
    best_model = best_experiment["Pipeline"]

    print("\nBest model based on validation F1:")
    print("Model:", best_experiment["Model"])
    print("C:", best_experiment["C"])
    print("Validation F1:", best_experiment["Validation F1"])

    print("\nEvaluating best model on test set...")
    test_results = evaluate_model(best_model, X_test, y_test)

    print("\n=== Best ML Classifier Test Results ===")
    print("Model:", best_experiment["Model"])
    print("C:", best_experiment["C"])
    print("Accuracy:", test_results["accuracy"])
    print("Precision:", test_results["precision"])
    print("Recall:", test_results["recall"])
    print("F1-score:", test_results["f1"])

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_results["y_pred"],
        target_names=CLASS_NAMES
    ))

    results_rows = []

    for exp in experiments:
        results_rows.append({
            "Model": exp["Model"],
            "C": exp["C"],
            "Validation Accuracy": exp["Validation Accuracy"],
            "Validation Precision": exp["Validation Precision"],
            "Validation Recall": exp["Validation Recall"],
            "Validation F1": exp["Validation F1"],
            "Training Time Seconds": exp["Training Time Seconds"]
        })

    tuning_df = pd.DataFrame(results_rows)
    tuning_df.to_csv("results/ml_classifier_tuning_results.csv", index=False)

    final_metrics_df = pd.DataFrame({
        "Approach": [f"MobileNetV1 Features + {best_experiment['Model']}"],
        "C": [best_experiment["C"]],
        "Accuracy": [test_results["accuracy"]],
        "Precision": [test_results["precision"]],
        "Recall": [test_results["recall"]],
        "F1-score": [test_results["f1"]],
        "Training Time Seconds": [best_experiment["Training Time Seconds"]]
    })

    final_metrics_df.to_csv("results/metrics_best_feature_extraction_ml.csv", index=False)

    save_confusion_matrix(
        y_test,
        test_results["y_pred"],
        f"Confusion Matrix - MobileNetV1 Features + {best_experiment['Model']}",
        "results/confusion_matrix_best_feature_extraction_ml.png"
    )

    joblib.dump(best_model, "models/best_feature_extraction_ml.pkl")

    print("\nSaved:")
    print("- results/ml_classifier_tuning_results.csv")
    print("- results/metrics_best_feature_extraction_ml.csv")
    print("- results/confusion_matrix_best_feature_extraction_ml.png")
    print("- models/best_feature_extraction_ml.pkl")


if __name__ == "__main__":
    main()