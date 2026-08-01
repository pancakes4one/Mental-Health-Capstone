import os
import pytest
from src.llm_interface import (
    parse_user_input,
    validate_and_clean,
    normalize_value,
    handle_message,
    ALLOWED_VALUES,
)


def test_parse_user_input_extracts_fields_from_real_message():
    """
    One real call to the LLM, confirming it correctly extracts values from
    natural language. Uses the cheap parsing model. Skipped if no API key
    is configured (e.g. a fresh clone without a .env file yet).
    """
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set -- skipping live API test")

    result = parse_user_input("I'm 29, female, and I work remotely")

    assert result["found"].get("age") == "29"
    assert result["found"].get("gender") == "female"
    assert result["found"].get("remote_work") == "Yes"


def test_validate_and_clean_rejects_out_of_range_age():
    """An unrealistic age should be dropped, not silently accepted."""
    cleaned, rejected = validate_and_clean({"age": "788"})
    assert "age" not in cleaned
    assert "age" in rejected


def test_validate_and_clean_accepts_valid_age():
    """A realistic age should pass through cleanly."""
    cleaned, rejected = validate_and_clean({"age": "29"})
    assert cleaned["age"] == "29"
    assert "age" not in rejected


def test_validate_and_clean_rejects_invalid_category_value():
    """A value not in the field's allowed list should be dropped, not accepted."""
    cleaned, rejected = validate_and_clean({"gender": "not a real answer"})
    assert "gender" not in cleaned
    assert "gender" in rejected


def test_normalize_value_maps_common_abbreviations():
    """Common shorthand should map to the exact allowed value."""
    assert normalize_value("gender", "f") == "female"
    assert normalize_value("gender", "m") == "male"
    assert normalize_value("work_interfere", "na") == "Not applicable"


def test_normalize_value_is_case_insensitive():
    """A correct value with different casing should still match."""
    assert normalize_value("remote_work", "yes") == "Yes"
    assert normalize_value("remote_work", "YES") == "Yes"


def test_handle_message_asks_for_missing_field_when_incomplete(monkeypatch):
    """
    With core fields still missing, handle_message should return a follow-up
    question instead of attempting a prediction. Uses a stubbed parser so
    this test is free and doesn't depend on network access.
    """
    def fake_parse(user_message, pending_fields=None):
        return {"found": {"age": "29"}, "missing": []}

    monkeypatch.setattr("src.llm_interface.parse_user_input", fake_parse)

    response, collected, prediction, pending = handle_message("I'm 29", {})

    assert prediction is None  # should not have predicted yet
    assert "age" in collected
    assert len(pending) > 0  # should have queued up the next question


def test_handle_message_explains_rejected_value(monkeypatch):
    """When a parsed value is invalid, the follow-up should explain why, not just repeat blindly."""
    def fake_parse(user_message, pending_fields=None):
        return {"found": {"age": "788"}, "missing": []}

    monkeypatch.setattr("src.llm_interface.parse_user_input", fake_parse)

    response, collected, prediction, pending = handle_message("I'm 788", {})

    assert prediction is None
    assert "age" not in collected  # invalid age should have been rejected
    assert "realistic" in response.lower()  # explanation should be present, not a silent re-ask