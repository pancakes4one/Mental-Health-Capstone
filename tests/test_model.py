import os
import pandas as pd
import numpy as np
import joblib
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_DATA_PATH = "data/processed/survey_clean.csv"
SCALER_PATH = "models/age_scaler.joblib"
TARGET_COL = "treatment"
RANDOM_STATE = 42

# These minimums are set well below the model's actual observed performance
# (~0.80 accuracy, ~0.82 F1) -- they catch a broken/degraded model, not meant
# to be a tight bar.
MIN_ACCURACY = 0.60
MIN_F1 = 0.60


def _artifacts_available():
    return os.path.exists(PROCESSED_DATA_PATH) and os.path.exists(SCALER_PATH)


@pytest.fixture(scope="module")
def trained_model():
    if not _artifacts_available():
        pytest.skip(
            "Model artifacts not found. Run `python src/train_models.py` first "
            "to generate the processed data, trained model, and scaler."
        )
    from src.evaluate import get_best_run
    import mlflow.sklearn

    best_run = get_best_run()
    return mlflow.sklearn.load_model(f"runs:/{best_run['run_id']}/model")


@pytest.fixture(scope="module")
def test_data():
    if not _artifacts_available():
        pytest.skip(
            "Processed data / scaler not found. Run `python src/train_models.py` first."
        )
    from src.preprocess import scale_age

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Same split parameters used in train_models.py, so this reproduces
    # the exact same held-out test set the model was actually evaluated on.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = joblib.load(SCALER_PATH)
    X_test, _ = scale_age(X_test, scaler=scaler)

    return X_test, y_test


def test_predictions_are_binary_array_of_correct_length(trained_model, test_data):
    """Predictions should be a 1D array of 0/1 values, one per test row."""
    X_test, y_test = test_data
    predictions = trained_model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X_test),)
    assert set(predictions).issubset({0, 1})


def test_predict_proba_shape_is_correct(trained_model, test_data):
    """predict_proba should return one probability pair (class 0, class 1) per row."""
    X_test, y_test = test_data
    if not hasattr(trained_model, "predict_proba"):
        pytest.skip("Model does not support predict_proba")

    probabilities = trained_model.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)
    # Each row's probabilities should sum to 1
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_meets_minimum_performance_threshold(trained_model, test_data):
    """The model should clear a low performance bar -- catches a broken/degraded model."""
    X_test, y_test = test_data
    predictions = trained_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)

    assert accuracy >= MIN_ACCURACY, f"Accuracy {accuracy:.3f} fell below minimum {MIN_ACCURACY}"
    assert f1 >= MIN_F1, f"F1 score {f1:.3f} fell below minimum {MIN_F1}"


def test_predict_on_single_row(trained_model, test_data):
    """The model should handle a single-row prediction (as used by the LLM interface), not just batches."""
    X_test, y_test = test_data
    single_row = X_test.iloc[[0]]  # keep as a 1-row DataFrame, not a Series

    prediction = trained_model.predict(single_row)
    assert prediction.shape == (1,)
    assert prediction[0] in {0, 1}