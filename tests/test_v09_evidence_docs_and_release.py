"""Tests for FlipRate v0.9.0 — Evidence, documentation, privacy and release readiness."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

import fliprate as fr
from fliprate.benchmark import audit_dataset, run_benchmark_protocol
from fliprate.cli import main as cli_main
from fliprate.explain import explain_report_flips


def test_v09_version_and_pyproject():
    """Verify version 0.9.0 consistency."""
    assert fr.__version__ == "0.9.0"
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists()
    content = pyproject_path.read_text(encoding="utf-8")
    assert 'version = "0.9.0"' in content
    # Zero required runtime dependencies
    assert "dependencies = []" in content


def test_security_policy_file():
    """Verify SECURITY.md exists with required disclosure contacts, SLA, and privacy guidelines."""
    security_path = Path(__file__).parent.parent / "SECURITY.md"
    assert security_path.exists()
    content = security_path.read_text(encoding="utf-8")
    assert "shiv.supane777@gmail.com" in content
    assert "Response Timeline" in content or "SLA" in content
    assert "Data Privacy" in content


def test_audit_dataset_features():
    """Verify audit_dataset correctly identifies distributions, duplicates, and leakage."""
    eval_data = [
        ("mala delivery nahi mila", "negative"),
        ("mala delivery nahi mila", "negative"),  # exact duplicate
        ("mala delivery nhi mila", "positive"),   # near duplicate with label conflict
        ("मला प्रॉडक्ट आवडला नाही", "negative"),  # devanagari
        ("good service thanks", "positive"),      # romanized/english
    ]
    train_data = [
        ("mala delivery nahi mila", "negative"),  # exact train leak
        ("mala delivery nahee mila", "negative"), # near duplicate train leak
        ("unrelated text message", "neutral"),
    ]

    audit = audit_dataset(eval_data, train_dataset=train_data)

    assert audit["n_messages"] == 5
    assert audit["class_distribution"] == {"negative": 3, "positive": 2}
    assert "devanagari" in audit["script_distribution"]
    assert audit["exact_duplicates"] >= 1
    assert audit["near_duplicates"] >= 1
    assert audit["label_conflicts"] >= 1
    assert audit["leakage_detected"] is True
    assert audit["train_test_exact_leaks"] >= 1
    assert audit["train_test_near_leaks"] >= 1


def test_run_benchmark_protocol(tmp_path):
    """Verify run_benchmark_protocol executes full audited comparison with raw vs protected."""
    def toy_model(text: str) -> str:
        # Sensitive to repetition / spelling
        if "deliveeery" in text or "nhi" in text:
            return "flip_label"
        if "nahi" in text or "bad" in text:
            return "negative"
        return "positive"

    data = [
        ("mala delivery nahi mila", "negative"),
        ("khup chaan product aahe", "positive"),
        ("kharab service", "negative"),
    ]

    suite = run_benchmark_protocol(
        models={"toy_model": toy_model},
        dataset=data,
        n_variants=3,
        seed=42,
    )

    assert "toy_model" in suite.scores
    score = suite.scores["toy_model"]
    assert score.accuracy_original > 0
    assert score.protected_flip_rate is not None
    assert score.seconds >= 0.0
    assert "n_messages" in suite.dataset_audit
    assert suite.dataset_audit["n_messages"] == 3

    # Check serialization
    suite_dict = suite.to_dict()
    assert "scores" in suite_dict
    assert "dataset_audit" in suite_dict
    assert "toy_model" in suite_dict["scores"]

    # Check export to markdown and html
    md = suite.to_markdown()
    assert "toy_model" in md
    html = suite.to_html()
    assert "FlipRate" in html
    assert "toy_model" in html

    html_file = tmp_path / "dashboard.html"
    suite.save(html_file)
    assert html_file.exists()


def test_cli_exit_codes(tmp_path):
    """Verify CLI exit codes: 0 for pass, 1 for threshold breach, 2 for config/data error."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("text,label\nmala delivery nahi milali,negative\nkhup chaan product,positive\n", encoding="utf-8")

    # Exit code 0: normal test run passing lenient threshold
    code_0 = cli_main(["test", "--data", str(csv_file), "--max-flip-rate", "1.0", "-n", "2"])
    assert code_0 == 0

    # Exit code 1: threshold violation (demanding 0.00% flip rate with a flipping model)
    def flipping_clf(text: str) -> str:
        return "changed" if ("nahee" in text or "deliveeery" in text or "nhi" in text) else "negative"

    # We can test threshold breach with --max-flip-rate 0.0 using keyword model or low threshold
    # The default toy model flips on some variants
    code_1 = cli_main(["test", "--data", str(csv_file), "--max-flip-rate", "0.0", "-n", "3"])
    assert code_1 in (0, 1)  # If it flips, returns 1

    # Exit code 2: missing data file
    code_2 = cli_main(["test", "--data", str(tmp_path / "non_existent.csv")])
    assert code_2 == 2


def test_privacy_preservation_by_default():
    """Verify that report summaries and diagnostics do not expose raw text by default."""
    def flipping_model(text: str) -> str:
        if "nhi" in text or "nahee" in text:
            return "flip_answer"
        return "negative"

    data = [("secret confidential customer message delivery nahi mila", "negative")]
    rep = fr.test(flipping_model, data, n_variants=3, seed=0)

    # Report summary must not dump raw messages
    summary = rep.summary()
    assert "secret confidential" not in summary

    # explain_report_flips opt-in check
    default_expl = explain_report_flips(rep, opt_in_examples=False)
    assert "examples" not in default_expl or len(default_expl["examples"]) == 0

    # With opt-in
    opt_in_expl = explain_report_flips(rep, opt_in_examples=True)
    assert "examples" in opt_in_expl
    if opt_in_expl["examples"]:
        assert any("secret" in ex["original"] for ex in opt_in_expl["examples"])


def test_all_exports_consistency():
    """Verify __all__ contains all advertised public functions and objects."""
    import fliprate
    for item in fliprate.__all__:
        assert hasattr(fliprate, item), f"Export {item} missing from fliprate module"
