"""Tests for v0.2.2 bug fixes and v0.3.0 features."""
import asyncio
import json
import pytest
import fliprate as fr
from fliprate.dataset import RuleSet
from fliprate.evaluation import load_dataset, read_csv_text
from fliprate import twitter_preprocess


def dummy_sentiment(text: str) -> str:
    # simple deterministic dummy
    return "neg" if "bekaar" in text or "kharab" in text else "pos"


def crashing_model(text: str) -> str:
    raise RuntimeError("Model inference server is down")


# --- 1. normalize() preserves URLs and entities byte-for-byte ---
def test_normalize_preserves_urls_and_entities():
    sample = "checkout https://example.com/test?a=1&b=2 and www.foo.bar/baz contact test.user@mail.com @shiv_01 #amazing 98765 nhi"
    res = fr.normalize(sample)
    # The URL, www, email, handle, hashtag, and numbers must remain byte-for-byte identical
    assert "https://example.com/test?a=1&b=2" in res.normalized
    assert "www.foo.bar/baz" in res.normalized
    assert "test.user@mail.com" in res.normalized
    assert "@shiv_01" in res.normalized
    assert "#amazing" in res.normalized
    assert "98765" in res.normalized
    # 'nhi' should be normalized to canonical 'nahi'
    assert "nahi" in res.normalized
    # Test specific example from prompt: normalize("http://x.com nahi") -> "http://x.com nahi"
    res_url = fr.normalize("http://x.com nahi")
    assert res_url.normalized == "http://x.com nahi"


# --- 2. gate() handles emoji without sending to REVIEW solely for script ---
def test_gate_emoji_not_review_for_script():
    # Latin text with emoji: previously was REVIEW because emoji failed Latin ratio
    res = fr.gate(dummy_sentiment, "achha 😡 hai", n_variants=3, min_variants=1)
    # Shouldn't fail due to "Mostly non-Latin script"
    assert "Mostly non-Latin script" not in res.reason
    assert "😡" in res.original_text


# --- 3. Argument validation on gate, perturb, test ---
def test_gate_argument_validation():
    with pytest.raises(ValueError, match="threshold"):
        fr.gate(dummy_sentiment, "achha hai", threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        fr.gate(dummy_sentiment, "achha hai", threshold=0.0)
    with pytest.raises(ValueError, match="n_variants"):
        fr.gate(dummy_sentiment, "achha hai", n_variants=0)
    with pytest.raises(ValueError, match="min_variants"):
        fr.gate(dummy_sentiment, "achha hai", min_variants=0)
    with pytest.raises(ValueError, match="on_error"):
        fr.gate(dummy_sentiment, "achha hai", on_error="invalid")


def test_perturb_argument_validation():
    with pytest.raises(ValueError, match="n must be >= 1"):
        fr.perturb("achha hai", n=0)


def test_test_argument_validation():
    with pytest.raises(ValueError, match="n_variants must be >= 1"):
        fr.test(dummy_sentiment, [("achha hai", "pos")], n_variants=0)
    with pytest.raises(ValueError, match="bootstrap must be >= 1"):
        fr.test(dummy_sentiment, [("achha hai", "pos")], bootstrap=0)


# --- 4. Bad input raises TypeError / ValueError with clear messages ---
def test_type_errors_on_bad_inputs():
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        fr.normalize(None)
    with pytest.raises(TypeError, match="text must be str, got int"):
        fr.normalize(123)
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        twitter_preprocess(None)
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        fr.perturb(None)
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        fr.gate(dummy_sentiment, None)
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        RuleSet().is_safe_variant(None, "test")
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        RuleSet().is_safe_variant("test", None)


def test_load_dataset_bad_row_message():
    bad_rows = [("good text", "pos"), ("only_text",)]
    with pytest.raises(ValueError, match="row 2 must be \\(text, label\\)"):
        load_dataset(bad_rows)

    bad_dict_rows = [{"text": "good", "label": "pos"}, {"wrong_key": "bad"}]
    with pytest.raises(ValueError, match="row 2 must be \\(text, label\\) or a dict with 'text' and 'label'"):
        load_dataset(bad_dict_rows)


# --- 5. gate() on_error handling ---
def test_gate_on_error_behavior():
    # on_error="raise" (default)
    with pytest.raises(RuntimeError, match="Model inference server is down"):
        fr.gate(crashing_model, "achha hai", on_error="raise")

    # on_error="review" returns GateResult
    res = fr.gate(crashing_model, "achha hai", on_error="review")
    assert res.decision == "REVIEW"
    assert "model error: RuntimeError: Model inference server is down" in res.reason


# --- 7. load_dataset / read_csv_text delimiter and custom column handling ---
def test_load_dataset_features():
    # Semicolon delimiter with custom column names and whitespace in header
    csv_data = "  message ; sentiment  \nkya haal ; pos\nbekaar hai ; neg\n"
    rows = read_csv_text(csv_data, text_col="message", label_col="sentiment")
    assert len(rows) == 2
    assert rows[0] == ("kya haal", "pos")
    assert rows[1] == ("bekaar hai", "neg")

    # Tab separated
    tsv_data = "text\tlabel\nkya haal\tpos\n"
    rows_tsv = read_csv_text(tsv_data)
    assert len(rows_tsv) == 1
    assert rows_tsv[0] == ("kya haal", "pos")


# --- 8. GateResult to_dict and to_json ---
def test_gate_result_serialization():
    res = fr.gate(dummy_sentiment, "achha hai", n_variants=3)
    d = res.to_dict()
    assert "decision" in d
    assert "prediction" in d
    assert "reason" in d
    assert "raw_stability" in d
    assert "normalized_stability" in d
    assert "warnings" in d
    assert "flipped_variants" in d

    j = res.to_json()
    parsed = json.loads(j)
    assert parsed["decision"] == res.decision
    assert parsed["flipped_variants"] == res.flipped_variants


# --- 9. async agate ---
def test_agate_execution():
    async def run_test():
        return await fr.agate(dummy_sentiment, "bahut achha hai", n_variants=3)

    res = asyncio.run(run_test())
    assert isinstance(res, fr.GateResult)
    assert res.prediction == "pos"


# --- 10. CLI report command ---
def test_cli_report(tmp_path):
    from fliprate.cli import main
    data_file = tmp_path / "data.csv"
    data_file.write_text("text,label\nkya haal hai,pos\nbekaar phone,neg\n", encoding="utf-8")
    report_file = tmp_path / "report.md"

    rc = main(["report", "--data", str(data_file), "--out", str(report_file), "-n", "2"])
    assert rc == 0
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "FlipRate Report" in content
    assert "Flip rate" in content
