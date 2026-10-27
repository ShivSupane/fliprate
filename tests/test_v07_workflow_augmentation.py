import pytest
from typing import List, Tuple

import fliprate as fr
from fliprate.adapters import FlipRateNormalizer
from fliprate.augmentation import AugmentationResult, augment, compare_augmentation
from fliprate.dataset import check_leakage, find_near_duplicates
from fliprate.evaluation import load_dataset


TRAIN_SAMPLE = """text,label
mala he product avadle,positive
khup changla phone aahe,positive
kharab service hoti,negative
service changli nahi hoti,negative
"""

TEST_SAMPLE = """text,label
product khup chaan aahe,positive
kharab anubhav aala,negative
service changli nhi hoti,negative
"""


class DummyDuckDataFrame:
    """Duck-typed table mimicking pandas DataFrame without requiring pandas."""

    def __init__(self, data: List[dict]):
        self._data = data
        self.columns = list(data[0].keys()) if data else []

    def to_dict(self, orient="records"):
        if orient == "records":
            return [dict(r) for r in self._data]
        raise NotImplementedError


class TestScikitLearnNormalizer:
    def test_normalizer_fit_and_transform_contract(self):
        norm = FlipRateNormalizer(lowercase=True)
        inputs = ["Mala Product Avadlaaa", "Khup Changla Phone", "Kharab Service Hoti"]
        
        # fit() must return self and do no data mutation
        fitted = norm.fit(inputs)
        assert fitted is norm

        # transform() preserves order and row count
        out = norm.transform(inputs)
        assert len(out) == len(inputs)
        assert out[0] == "mala product avadla"
        assert out[1] == "khup changla phone"
        assert out[2] == "kharab service hoti"

    def test_normalizer_sklearn_pipeline_and_clone(self):
        try:
            from sklearn.base import clone
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError:
            pytest.skip("scikit-learn is not installed in this environment")

        norm = FlipRateNormalizer(lowercase=True)
        
        # Verify clone() compatibility
        cloned = clone(norm)
        assert cloned is not norm
        assert cloned.lowercase == norm.lowercase
        assert cloned.rules == norm.rules

        # Verify pipeline compatibility
        pipe = Pipeline([
            ("normalizer", norm),
            ("vec", CountVectorizer()),
            ("clf", LogisticRegression()),
        ])

        X_train = ["mala product avadlaaa", "khup changla", "kharab service", "service bekaar hoti"]
        y_train = ["positive", "positive", "negative", "negative"]

        pipe.fit(X_train, y_train)
        preds = pipe.predict(["mala product avadla", "kharab service"])
        assert len(preds) == 2


class TestTrainingAugmentation:
    def test_augment_deterministic_and_preserves_originals(self):
        res1 = augment(TRAIN_SAMPLE, n_per_example=1, seed=42)
        res2 = augment(TRAIN_SAMPLE, n_per_example=1, seed=42)

        # Deterministic
        assert res1.to_rows() == res2.to_rows()
        assert res1.n_original == 4
        assert res1.n_augmented > 0
        assert res1.total_rows == res1.n_original + res1.n_augmented

        # Original rows preserved as first entries or original indices
        orig_texts = [r[0] for r in load_dataset(TRAIN_SAMPLE)]
        detailed_orig = [r.text for r in res1.detailed_rows if not r.is_augmented]
        assert detailed_orig == orig_texts

    def test_augment_without_originals(self):
        res = augment(TRAIN_SAMPLE, n_per_example=1, preserve_originals=False, seed=42)
        assert res.total_rows == res.n_augmented
        assert all(r.is_augmented for r in res.detailed_rows)

    def test_negation_invariance_in_augmentation(self):
        # Sample with explicit negation
        neg_data = """text,label
mala he product avadle nahi,negative
service changli nahi hoti,negative
"""
        res = augment(neg_data, n_per_example=2, seed=99)
        rows = res.to_rows()

        # Every augmented row for negative statements must still preserve negation
        for text, label in rows:
            assert label == "negative"
            t_lower = text.lower()
            assert any(neg in t_lower for neg in ("nahi", "nhi", "nahin", "naahi"))

    def test_augmentation_result_formats(self):
        res = augment(TRAIN_SAMPLE, n_per_example=1, seed=1)
        dicts = res.to_dicts()
        assert isinstance(dicts, list)
        assert len(dicts) == res.total_rows
        assert "text" in dicts[0] and "label" in dicts[0]
        assert "is_augmented" in dicts[0]
        assert "source_row" in dicts[0]

        d = res.to_dict()
        assert d["n_original"] == 4
        assert d["n_augmented"] > 0
        assert "by_kind" in d
        assert "summary" in dir(res)

    def test_compare_augmentation_workflow(self):
        class DummyLinearModel:
            def __init__(self):
                self.vocab = {}

            def fit(self, X, y):
                for text, label in zip(X, y):
                    for w in text.lower().split():
                        self.vocab[w] = label

            def __call__(self, text: str) -> str:
                for w in text.lower().split():
                    if w in self.vocab:
                        return self.vocab[w]
                return "positive"

        cmp_report = compare_augmentation(
            model_factory=lambda: DummyLinearModel(),
            train_data=TRAIN_SAMPLE,
            test_data=TEST_SAMPLE,
            n_per_example=1,
            seed=0,
        )

        assert "baseline" in cmp_report
        assert "augmented" in cmp_report
        assert "diff" in cmp_report
        assert cmp_report["augmented"]["train_size"] > cmp_report["baseline"]["train_size"]


class TestDatasetErgonomics:
    def test_load_dataset_duck_typed_dataframe(self):
        table = DummyDuckDataFrame([
            {"text": "mala phone avadla", "label": "positive"},
            {"text": "kharab ahe", "label": "negative"},
        ])
        rows = load_dataset(table)
        assert len(rows) == 2
        assert rows[0] == ("mala phone avadla", "positive")
        assert rows[1] == ("kharab ahe", "negative")

    def test_find_near_duplicates_with_row_references(self):
        csv_with_dups = """text,label
mala phone avadla,positive
khup changla phone,positive
mala phone avadla,positive
mala phone avadlaaa,positive
mala phone avadla,negative
"""
        dup_rep = find_near_duplicates(csv_with_dups)
        assert dup_rep.has_duplicates is True
        assert len(dup_rep.exact_duplicates) >= 2  # rows 1&3, 1&5
        assert len(dup_rep.near_duplicates) >= 1   # rows 1&4 (spelling variant)
        assert len(dup_rep.conflicting_pairs) >= 1 # row 5 has conflicting label 'negative'

        # Verify explainable row references
        first_exact = dup_rep.exact_duplicates[0]
        assert first_exact.row_a == 1
        assert first_exact.row_b == 3
        assert first_exact.match_type == "exact"

        summary = dup_rep.summary()
        assert "Exact duplicates found" in summary
        assert "Conflicting examples" in summary

    def test_check_leakage_diagnostics(self):
        train_csv = """text,label
mala he product avadla,positive
khup changla phone aahe,positive
kharab service hoti,negative
"""
        # test_csv contains exact leak (row 1) and near-variant leak (row 2: avadlaaa)
        test_csv = """text,label
mala he product avadla,positive
mala he product avadlaaa,positive
service changli hoti,positive
"""
        leak_rep = check_leakage(train_csv, test_csv)
        assert leak_rep.has_leakage is True
        assert len(leak_rep.exact_leaks) == 1
        assert len(leak_rep.near_leaks) >= 1

        exact = leak_rep.exact_leaks[0]
        assert exact.train_row == 1
        assert exact.test_row == 1
        assert exact.leak_type == "exact"

        near = leak_rep.near_leaks[0]
        assert near.train_row == 1
        assert near.test_row == 2
        assert near.leak_type == "near_variant"

        summary = leak_rep.summary()
        assert "Exact train/test leaks: 1" in summary
        assert "Near-variant spelling leaks" in summary


class TestFlipRateEngineAugment:
    def test_engine_augment_method(self):
        engine = fr.FlipRate(seed=42)
        res = engine.augment(TRAIN_SAMPLE, n_per_example=1)
        assert isinstance(res, AugmentationResult)
        assert res.n_augmented > 0
        assert res.seed == 42
