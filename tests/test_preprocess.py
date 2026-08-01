import pandas as pd
import pytest
from src.preprocess import clean_data, encode_data, scale_age


@pytest.fixture
def raw_sample():
    """A small fake dataset mimicking the real survey structure, with known edge cases."""
    return pd.DataFrame({
        "Age": [25, 30, 150, 40, 22],  # 150 is an invalid age, should get filtered out
        "Gender": ["Female", "M", "male", "non-binary", "F"],
        "Country": ["United States", "Canada", "United States", "United States", "Germany"],
        "self_employed": ["No", "Yes", "No", None, "No"],  # one missing value
        "family_history": ["Yes", "No", "Yes", "No", "Yes"],
        "treatment": ["Yes", "No", "Yes", "No", "No"],
        "work_interfere": ["Often", None, "Rarely", None, "Sometimes"],  # missing values
        "no_employees": ["6-25", "1-5", "26-100", "6-25", "1-5"],
        "remote_work": ["Yes", "No", "Yes", "No", "Yes"],
        "tech_company": ["Yes", "Yes", "No", "Yes", "No"],
        "benefits": ["Yes", "Don't know", "No", "Not sure", "Yes"],
        "care_options": ["Yes", "No", "Not sure", "Yes", "No"],
        "wellness_program": ["No", "Don't know", "Yes", "No", "No"],
        "seek_help": ["Yes", "No", "Don't know", "Yes", "No"],
        "anonymity": ["Yes", "Don't know", "No", "Yes", "Don't know"],
        "leave": ["Somewhat easy", "Don't know", "Very easy", "Somewhat difficult", "Very difficult"],
        "mental_health_consequence": ["No", "Maybe", "Yes", "No", "Maybe"],
        "phys_health_consequence": ["No", "No", "Maybe", "Yes", "No"],
        "coworkers": ["Some of them", "No", "Yes", "Some of them", "No"],
        "supervisor": ["Yes", "No", "Some of them", "Yes", "No"],
        "mental_health_interview": ["No", "Maybe", "No", "Yes", "No"],
        "phys_health_interview": ["Maybe", "No", "No", "Yes", "No"],
        "mental_vs_physical": ["Yes", "Don't know", "No", "Yes", "Don't know"],
        "obs_consequence": ["No", "Yes", "No", "No", "Yes"],
    })


def test_clean_data_does_not_mutate_original(raw_sample):
    """clean_data should not modify the DataFrame it was given."""
    original_copy = raw_sample.copy(deep=True)
    clean_data(raw_sample)
    pd.testing.assert_frame_equal(raw_sample, original_copy)


def test_clean_data_filters_invalid_age(raw_sample):
    """Ages outside 18-100 should be removed."""
    result = clean_data(raw_sample)
    assert result["age"].between(18, 100).all()
    assert 150 not in result["age"].values


def test_clean_data_handles_missing_self_employed(raw_sample):
    """Rows with missing self_employed should be dropped."""
    result = clean_data(raw_sample)
    assert result["self_employed"].isnull().sum() == 0


def test_clean_data_handles_missing_work_interfere(raw_sample):
    """Missing work_interfere should be filled with 'Not applicable', not left as NaN."""
    result = clean_data(raw_sample)
    assert result["work_interfere"].isnull().sum() == 0
    assert "Not applicable" in result["work_interfere"].values


def test_clean_data_standardizes_unsure_values(raw_sample):
    """'Don't know', 'Not sure', and 'Maybe' should all become 'Unsure'."""
    result = clean_data(raw_sample)
    for col in ["benefits", "care_options", "wellness_program"]:
        assert "Don't know" not in result[col].values
        assert "Not sure" not in result[col].values
        assert "Maybe" not in result[col].values


def test_encode_data_binary_columns_are_numeric(raw_sample):
    """Binary Yes/No columns should become 1/0 integers after encoding."""
    cleaned = clean_data(raw_sample)
    encoded = encode_data(cleaned)
    assert set(encoded["family_history"].unique()).issubset({0, 1})
    assert set(encoded["remote_work"].unique()).issubset({0, 1})


def test_encode_data_creates_one_hot_columns(raw_sample):
    """Nominal columns like gender should be one-hot encoded."""
    cleaned = clean_data(raw_sample)
    encoded = encode_data(cleaned)
    assert "gender_female" in encoded.columns
    assert "gender_male" in encoded.columns
    assert "gender" not in encoded.columns  # original column should be gone


def test_encode_data_does_not_mutate_input(raw_sample):
    """encode_data should not modify the DataFrame it was given."""
    cleaned = clean_data(raw_sample)
    original_copy = cleaned.copy(deep=True)
    encode_data(cleaned)
    pd.testing.assert_frame_equal(cleaned, original_copy)


def test_scale_age_fits_to_mean_zero(raw_sample):
    """When fitting a new scaler, the scaled age column should have mean ~0."""
    cleaned = clean_data(raw_sample)
    encoded = encode_data(cleaned)
    scaled, scaler = scale_age(encoded)
    assert abs(scaled["age"].mean()) < 1e-6


def test_scale_age_reuses_existing_scaler(raw_sample):
    """Passing an existing scaler should transform without refitting (different mean is fine)."""
    cleaned = clean_data(raw_sample)
    encoded = encode_data(cleaned)
    _, scaler = scale_age(encoded)

    new_data = encoded.copy()
    new_data["age"] = [50] * len(new_data)
    scaled_new, reused_scaler = scale_age(new_data, scaler=scaler)

    assert reused_scaler is scaler  # confirms it reused the same scaler, didn't fit a new one