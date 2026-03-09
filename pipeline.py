"""
Pipeline Runner
Orchestrates: ingest → preprocess → train → evaluate
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from data_ingestion import ingest_data
from preprocessing import feature_engineering, preprocess_data, save_preprocessor
from train import optimize_logistic_regression, train_model, save_model
from evaluation import evaluate_model

BASE_DIR     = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
RANDOM_STATE = 42
ACCURACY_THRESHOLD = 0.75


def run_pipeline():
    # Step 1: Data Ingestion
    print("=" * 50)
    print("Step 1: Data Ingestion")
    print("=" * 50)
    ingest_data()

    # Step 2: Preprocessing
    print("\n" + "=" * 50)
    print("Step 2: Preprocessing")
    print("=" * 50)
    df = pd.read_csv(INGESTED_DIR / "train.csv")
    df = feature_engineering(df)
    X, y, feature_columns, preprocessor = preprocess_data(df)
    save_preprocessor(preprocessor, feature_columns)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # Step 3: Training
    print("\n" + "=" * 50)
    print("Step 3: Training (Logistic Regression + Optuna)")
    print("=" * 50)
    best_params = optimize_logistic_regression(X_train, y_train, n_trials=30)
    model = train_model(X, y, best_params)  # train pada full data
    save_model(model)

    # Step 4: Evaluation
    print("\n" + "=" * 50)
    print("Step 4: Evaluation")
    print("=" * 50)
    model_val = train_model(X_train, y_train, best_params)  # train pada train set untuk eval
    metrics = evaluate_model(model_val, X_val, y_val)

    # Result
    accuracy = metrics['accuracy']
    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED for deployment (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < threshold={ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    run_pipeline()