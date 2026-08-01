import pandas as pd
import numpy as np
import json
import joblib
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import preprocess, scale_age
import warnings

warnings.filterwarnings('ignore')

# --- Load configuration ---
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = config['data']['raw_path']
PROCESSED_DATA_PATH = config['data']['processed_path']
TARGET_COL = config['data']['target_column']
TEST_SIZE = config['data']['test_size']
RANDOM_STATE = config['data']['random_state']

print("--- Step 1: Loading & Preprocessing Data ---")
df_clean = preprocess(RAW_DATA_PATH, verbose=True)

df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\nProcessed data saved to '{PROCESSED_DATA_PATH}'")

X = df_clean.drop(columns=[TARGET_COL])
y = df_clean[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Fit the age scaler on TRAINING data only, then apply that same scaler to test data.
X_train, age_scaler = scale_age(X_train)
X_test, _ = scale_age(X_test, scaler=age_scaler)

joblib.dump(age_scaler, 'models/age_scaler.joblib')
with open('models/feature_columns.json', 'w') as f:
    json.dump(list(X_train.columns), f)
print("Saved age_scaler.joblib and feature_columns.json to models/")

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Step 2: Build models from config ---
model_configs = config['models']

models = {
    "Random_Forest_default": RandomForestClassifier(**model_configs['random_forest_default']),
    "Random_Forest_deeper": RandomForestClassifier(**model_configs['random_forest_deeper']),
    "Gradient_Boosting_default": GradientBoostingClassifier(**model_configs['gradient_boosting_default']),
    "Gradient_Boosting_slow_lr": GradientBoostingClassifier(**model_configs['gradient_boosting_slow_lr']),
    "Neural_Network": MLPClassifier(
        hidden_layer_sizes=tuple(model_configs['neural_network']['hidden_layer_sizes']),
        activation=model_configs['neural_network']['activation'],
        solver=model_configs['neural_network']['solver'],
        max_iter=model_configs['neural_network']['max_iter'],
        random_state=model_configs['neural_network']['random_state'],
        early_stopping=model_configs['neural_network']['early_stopping'],
        validation_fraction=model_configs['neural_network']['validation_fraction'],
    )
}

# --- Step 3: Train, Evaluate, and Track with MLflow (one run per model) ---
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"Training {name}...")

        params = model.get_params()
        logged_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        mlflow.log_params(logged_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_prob)
        }

        results[name] = metrics_dict
        mlflow.log_metrics(metrics_dict)
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")

        print(f"  -> Accuracy: {metrics_dict['accuracy']:.4f}, "
              f"F1: {metrics_dict['f1_score']:.4f}, AUC: {metrics_dict['auc_roc']:.4f}\n")

# --- Step 4: Select Best Model ---
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_metrics = results[best_model_name]

print("-" * 30)
print("--- Final Results Summary ---")
print("-" * 30)
for name, metrics in results.items():
    marker = "<-- BEST MODEL" if name == best_model_name else ""
    print(f"{name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f} {marker}\n")

print(f"The best performing model is {best_model_name}.")
print(f"It achieved an F1-score of {best_metrics['f1_score']:.4f}.")
print("\nTo view detailed logs, charts, and artifacts, run: mlflow ui")