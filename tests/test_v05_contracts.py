"""Contract tests for FlipRate v0.5.0 features.

Tests canonical Decision type, FlipRate engine class, assert_stable assertion helper,
and CLI regression gate exit codes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import pytest

import fliprate as fr
from fliprate.gating import PASS, TRANSFORM, REVIEW


def _dummy_model(text: str) -> str:
    t = text.lower()
    return "positive" if any(w in t for w in ["accha", "acha", "good", "sahi", "badiya"]) else "negative"


def _flaky_model(text: str) -> str:
    # Intentionally sensitive to spelling variants
    if "accha" in text:
        return "positive"
    return "negative"


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "sample.csv"
    content = (
        "text,label\n"
        "bohot accha product hai,positive\n"
        "bilkul bekar service,negative\n"
        "acha service laga,positive\n"
        "delivery nahi aayi,negative\n"
    )
    path.write_text(content, encoding="utf-8")
    return str(path)


# ─── 1. Canonical Decision Type Tests ─────────────────────────────────────────

def test_decision_type_and_gate_result_equivalence():
    res = fr.gate(_dummy_model, "bohot accha product", n_variants=5, seed=42)
    assert isinstance(res, fr.Decision)
    assert isinstance(res, fr.GateResult)
    assert res.decision in (PASS, TRANSFORM, REVIEW)
    assert isinstance(res.is_safe, bool)
    assert isinstance(res.should_route_to_human, bool)
    assert res.prediction == "positive"
    assert res.text_used != ""
    assert res.original_text == "bohot accha product"
    assert isinstance(res.warnings, list)
    assert isinstance(res.flipped_variants, list)

    # Dictionary and boolean access
    d = res.to_dict()
    assert d["decision"] == res.decision
    assert d["is_safe"] == res.is_safe
    assert res["decision"] == res.decision
    assert bool(res) == res.is_safe

    # JSON serialization
    j = json.loads(res.to_json())
    assert j["decision"] == res.decision


def test_decision_properties_on_review():
    # Force a REVIEW decision (e.g. unknown script/empty letters)
    res = fr.gate(_dummy_model, "!!! ??? 123", script_aware=True)
    assert res.decision == REVIEW
    assert not res.is_safe
    assert res.should_route_to_human
    assert not bool(res)


# ─── 2. FlipRate Configuration Object Tests ───────────────────────────────────

def test_fliprate_engine_with_model():
    engine = fr.FlipRate(
        model=_dummy_model,
        stability_threshold=0.8,
        n_variants=5,
        seed=42,
    )

    # Gating
    dec = engine.gate("bohot accha product")
    assert isinstance(dec, fr.Decision)
    assert dec.decision == PASS

    # Protection
    safe_m = engine.protect()
    assert callable(safe_m)
    assert safe_m("acha product") == "positive"

    # Perturbation & Normalization
    variants = engine.perturb("bohot accha product")
    assert len(variants) <= 5
    assert all(isinstance(v, fr.Variant) for v in variants)

    norm = engine.normalize("bohot accha product")
    assert isinstance(norm, fr.NormalizeResult)
    assert norm.normalized != ""

    # Dataset testing
    data = [("bohot accha product", "positive"), ("bekar", "negative")]
    rep = engine.test(data)
    assert isinstance(rep, fr.Report)


def test_fliprate_engine_without_initial_model():
    engine = fr.FlipRate(n_variants=5, seed=42)

    with pytest.raises(ValueError, match="No model provided"):
        engine.gate("some text")

    # Override model at call time
    dec = engine.gate("bohot accha product", model=_dummy_model)
    assert isinstance(dec, fr.Decision)


# ─── 3. assert_stable Helper Tests ───────────────────────────────────────────

def test_assert_stable_passes():
    dataset = [
        ("bohot accha product hai", "positive"),
        ("acha service", "positive"),
        ("bekar delivery", "negative"),
    ]
    rep = fr.assert_stable(
        _dummy_model,
        dataset,
        max_flip_rate=0.50,
        min_accuracy=0.50,
        max_accuracy_drop=0.50,
        n_variants=3,
        seed=42,
    )
    assert isinstance(rep, fr.Report)


def test_assert_stable_fails_on_flip_rate():
    dataset = [
        ("accha product hai", "positive"),
    ]
    with pytest.raises(AssertionError) as exc_info:
        fr.assert_stable(
            _flaky_model,
            dataset,
            max_flip_rate=0.0,  # 0% allowable
            n_variants=5,
            seed=42,
        )
    assert "FlipRate stability gate failed" in str(exc_info.value)
    assert "flip rate" in str(exc_info.value)


def test_assert_stable_fails_on_min_accuracy():
    dataset = [
        ("bohot accha product hai", "negative"),  # Ground truth negative, model predicts positive
    ]
    with pytest.raises(AssertionError) as exc_info:
        fr.assert_stable(
            _dummy_model,
            dataset,
            min_accuracy=0.99,  # Impossible accuracy
            n_variants=3,
            seed=42,
        )
    assert "below min_accuracy" in str(exc_info.value)


# ─── 4. CLI Regression Gate Tests ─────────────────────────────────────────────

def test_cli_test_threshold_pass(sample_csv):
    cmd = [
        sys.executable,
        "-m",
        "fliprate.cli",
        "test",
        "--data",
        sample_csv,
        "-n",
        "3",
        "--max-flip-rate",
        "1.0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "[CI GATE PASSED]" in proc.stderr or proc.returncode == 0


def test_cli_test_threshold_failure(sample_csv):
    # Using toy model with max-flip-rate=0.0 where flips happen
    cmd = [
        sys.executable,
        "-m",
        "fliprate.cli",
        "test",
        "--data",
        sample_csv,
        "-n",
        "5",
        "--max-flip-rate",
        "0.0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # If any flips occur, exit code must be 1
    if "flip rate 0.0%" not in proc.stdout:
        assert proc.returncode == 1
        assert "[CI GATE FAILED]" in proc.stderr


def test_cli_test_invalid_data_exit_code():
    cmd = [
        sys.executable,
        "-m",
        "fliprate.cli",
        "test",
        "--data",
        "non_existent_file_xyz_123.csv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "Error loading model or dataset" in proc.stderr


def test_cli_save_report_records_ci_metadata(sample_csv, tmp_path):
    out_json = str(tmp_path / "ci_report.json")
    cmd = [
        sys.executable,
        "-m",
        "fliprate.cli",
        "test",
        "--data",
        sample_csv,
        "-n",
        "2",
        "--max-flip-rate",
        "1.0",
        "--save",
        out_json,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert os.path.exists(out_json)

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data
    assert "ci_thresholds" in data["metadata"]
    assert data["metadata"]["ci_thresholds"]["max_flip_rate"] == 1.0
    assert data["metadata"]["ci_gate_passed"] is True
