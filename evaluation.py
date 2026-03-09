"""
Step 4: Evaluation
Menghitung metrics dari model yang sudah ditraining.
"""

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_val, y_val) -> dict:
    """Evaluate model dan return metrics."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        'accuracy':               accuracy_score(y_val, y_pred),
        'roc_auc':                roc_auc_score(y_val, y_prob),
        'classification_report':  classification_report(y_val, y_pred),
        'confusion_matrix':       confusion_matrix(y_val, y_pred),
    }

    print(f"[evaluation] Accuracy : {metrics['accuracy']:.4f}")
    print(f"[evaluation] ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"\n[evaluation] Classification Report:\n{metrics['classification_report']}")

    return metrics