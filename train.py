"""
Step 3: Training
Hyperparameter optimization dengan Optuna + train Logistic Regression.
"""

from pathlib import Path
import pickle
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR   = Path(__file__).parent
MODEL_FILE = BASE_DIR / "models" / "model.pkl"
RANDOM_STATE = 42


def optimize_logistic_regression(X, y, n_trials: int = 30) -> dict:
    """Run Optuna hyperparameter search for Logistic Regression."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            'C':        trial.suggest_float('C', 0.001, 100, log=True),
            'penalty':  trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver':   trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 2000),
            'random_state': RANDOM_STATE
        }
        model = LogisticRegression(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[train] Best CV Accuracy : {study.best_value:.4f}")
    print(f"[train] Best Params      : {study.best_params}")
    return study.best_params


def train_model(X, y, params: dict) -> LogisticRegression:
    """Train final Logistic Regression with best params."""
    model = LogisticRegression(**params, random_state=RANDOM_STATE)
    model.fit(X, y)
    print("[train] Model trained successfully.")
    return model


def save_model(model):
    """Save model to pickle."""
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"[train] Model saved → {MODEL_FILE}")


def load_model():
    """Load model from pickle."""
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)