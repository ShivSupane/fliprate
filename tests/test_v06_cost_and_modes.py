import json
import pytest
import fliprate as fr
from fliprate.models import Predictor
from fliprate.evaluation import Report


def dummy_sensitive_model(text: str) -> str:
    """Flips to 'bad' if text has 'a' repeated or specific variants."""
    if "aa" in text or "nhi" in text:
        return "negative"
    return "positive"


import re

def dummy_stable_model(text: str) -> str:
    """Stable classifier that checks root words, collapsing character repetitions."""
    t = re.sub(r'(.)\1+', r'\1', text.lower())
    if "kharab" in t or "khrab" in t:
        return "negative"
    return "positive"


class CallCountingModel:
    def __init__(self, fn):
        self.fn = fn
        self.call_count = 0

    def __call__(self, text: str) -> str:
        self.call_count += 1
        return self.fn(text)


SAMPLE_CSV = """text,label
mala product avadla,positive
khup changla aahe,positive
kharab service hoti,negative
"""


class TestPredictorCostTracking:
    def test_predictor_accounting_and_cache(self):
        counter = CallCountingModel(dummy_stable_model)
        p = Predictor(counter, cache=True, batch_size=2)

        # 3 items, 1 duplicate in first call
        res = p.predict(["mala product avadla", "khup changla", "mala product avadla"])
        assert len(res) == 3
        assert p.total_lookups == 3
        assert p.cache_hits == 0
        assert p.calls == 2
        assert p.request_count == 1  # 2 unique items fit in 1 batch of size 2

        # Query same items again -> 100% cache hits
        res2 = p.predict(["mala product avadla", "khup changla"])
        assert p.total_lookups == 5
        assert p.cache_hits == 2
        assert p.calls == 2  # no new underlying model calls
        assert counter.call_count == 2
        assert 0.35 <= p.cache_hit_rate <= 0.45

        # Reset stats
        p.reset_stats()
        assert p.calls == 0
        assert p.request_count == 0
        assert p.cache_hits == 0
        assert p.total_lookups == 0

        # Invalidate cache
        p.invalidate_cache(reset_stats=True)
        res3 = p.predict(["mala product avadla"])
        assert p.calls == 1
        assert counter.call_count == 3


class TestGatingModes:
    def test_gate_fast_mode_evaluates_zero_variants(self):
        counter = CallCountingModel(dummy_stable_model)
        dec = fr.gate(counter, "mala product avadlaaa nahi", mode="fast")

        assert dec.mode == "fast"
        assert dec.n_variants == 0
        assert dec.n_variants_requested == 0
        assert dec.n_variants_evaluated == 0
        assert dec.raw_stability is None
        assert dec.evaluation_complete is True
        assert dec.stop_reason is None
        assert any("Robustness was not checked" in w for w in dec.warnings)
        # Model should only have been called for original text (1 call)
        assert counter.call_count == 1

    def test_gate_balanced_mode(self):
        dec = fr.gate(dummy_stable_model, "mala product avadlaaa nahi", mode="balanced", n_variants=8)
        assert dec.mode == "balanced"
        assert dec.n_variants_evaluated <= 3
        assert dec.evaluation_complete is True

    def test_gate_thorough_mode_default(self):
        dec = fr.gate(dummy_stable_model, "mala product avadlaaa nahi", n_variants=5)
        assert dec.mode == "thorough"
        assert dec.n_variants_evaluated >= 3
        assert dec.raw_stability is not None
        assert dec.evaluation_complete is True

    def test_gate_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            fr.gate(dummy_stable_model, "test text", mode="turbo")


class TestGatingEarlyStopping:
    def test_early_stopping_on_first_flip(self):
        dec = fr.gate(
            dummy_sensitive_model,
            "mala product avadla",  # variants include 'avadlaaa' which triggers flip in dummy_sensitive_model
            n_variants=8,
            early_stop=True,
        )
        assert dec.evaluation_complete is False
        assert dec.stop_reason in ("first_flip_detected", "threshold_unachievable")
        assert dec.raw_stability is None  # Must not claim exact full stability
        assert any("Early stop" in w for w in dec.warnings)
        assert dec.decision == "REVIEW"

    def test_early_stopping_on_stable_input(self):
        dec = fr.gate(
            dummy_stable_model,
            "mala product avadla",
            n_variants=5,
            early_stop=True,
        )
        assert dec.evaluation_complete is True
        assert dec.stop_reason is None
        assert dec.raw_stability == 1.0


class TestEvaluationModesAndReports:
    def test_test_fast_mode_evaluates_zero_variants(self):
        counter = CallCountingModel(dummy_stable_model)
        rep = fr.test(counter, SAMPLE_CSV, mode="fast")

        assert rep.mode == "fast"
        assert rep.n_variants == 0
        assert rep.n_variants_requested == 0
        assert rep.n_variants_evaluated == 0
        assert rep.flip_rate == 0.0
        assert rep.accuracy_original == 1.0
        assert rep.accuracy_variants == 1.0
        assert rep.metadata["robustness_evaluated"] is False
        assert "[fast mode]" in rep.summary()
        # Exactly 3 original rows were predicted
        assert counter.call_count == 3

    def test_test_balanced_mode(self):
        rep = fr.test(dummy_stable_model, SAMPLE_CSV, mode="balanced", n_variants=6)
        assert rep.mode == "balanced"
        assert rep.metadata["sampling_policy"] == "deterministic_sample"
        assert rep.n_variants_evaluated > 0
        assert rep.evaluation_complete is True

    def test_test_reproducibility(self):
        rep1 = fr.test(dummy_stable_model, SAMPLE_CSV, n_variants=4, seed=42)
        rep2 = fr.test(dummy_stable_model, SAMPLE_CSV, n_variants=4, seed=42)

        assert rep1.flip_rate == rep2.flip_rate
        assert rep1.accuracy_variants == rep2.accuracy_variants
        assert rep1.n_variants == rep2.n_variants
        assert rep1.metadata["dataset_sha256_16"] == rep2.metadata["dataset_sha256_16"]
        assert rep1.metadata["schema_version"] == "0.6.0"

    def test_report_dict_and_schema_version(self):
        rep = fr.test(dummy_stable_model, SAMPLE_CSV, n_variants=3, model_version="v1.0.0")
        d = rep.to_dict()
        assert d["schema_version"] == "0.6.0"
        assert d["mode"] == "thorough"
        assert d["evaluation_complete"] is True
        assert d["metadata"]["model_version"] == "v1.0.0"
        assert "request_count" in d["metadata"]
        assert "cache_hit_rate" in d["metadata"]


class TestEngineAndAssertStableModes:
    def test_fliprate_engine_fast_mode(self):
        engine = fr.FlipRate(model=dummy_stable_model, mode="fast")
        dec = engine.gate("mala product avadla")
        assert dec.mode == "fast"
        assert dec.n_variants == 0

        rep = engine.test(SAMPLE_CSV)
        assert rep.mode == "fast"
        assert rep.n_variants == 0

    def test_assert_stable_with_mode(self):
        rep = fr.assert_stable(
            dummy_stable_model,
            SAMPLE_CSV,
            max_flip_rate=0.05,
            min_accuracy=0.8,
            mode="balanced",
        )
        assert rep.mode == "balanced"
        assert rep.evaluation_complete is True


class TestCliModes:
    def test_cli_gate_fast(self, capsys):
        from fliprate.cli import main
        ret = main(["gate", "mala product avadla", "--mode", "fast"])
        assert ret == 0 or ret is None
        captured = capsys.readouterr()
        assert "PASS" in captured.out or "REVIEW" in captured.out
