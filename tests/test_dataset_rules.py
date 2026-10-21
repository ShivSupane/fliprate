import os
import tempfile
import unittest

import fliprate as fr
from fliprate.dataset import check_dataset, mine_variants, read_sentimix, split_dataset, write_candidates
from fliprate.rules import RuleSet


class RuleSetTests(unittest.TestCase):
    def test_extend_adds_new_group(self):
        r = fr.DEFAULT_RULES.extended(groups=[("paisa", "paise", "pese")])
        self.assertIsNotNone(r.group_of("pese"))
        self.assertIsNone(fr.DEFAULT_RULES.group_of("pese"))
        self.assertNotEqual(r.fingerprint(), fr.DEFAULT_RULES.fingerprint())

    def test_overlap_merges_into_existing_group(self):
        r = fr.DEFAULT_RULES.extended(groups=[("nahi", "nahii", "naheen")])
        self.assertEqual(r.group_of("naheen"), r.group_of("nahi"))

    def test_group_joining_two_groups_is_refused(self):
        with self.assertRaises(ValueError):
            fr.DEFAULT_RULES.extended(groups=[("nahi", "bahut")])

    def test_ambiguous_word_cannot_become_safe(self):
        with self.assertRaises(ValueError):
            fr.DEFAULT_RULES.extended(groups=[("nahi", "nai")])

    def test_custom_rules_change_perturbation(self):
        r = fr.DEFAULT_RULES.extended(groups=[("paisa", "pese")])
        vs = fr.perturb("mala pese pahije", n=6, kinds=["spelling"], rules=r)
        self.assertTrue(any("paisa" in v.text for v in vs))
        self.assertEqual(fr.perturb("mala pese pahije", n=6, kinds=["spelling"]), [])

    def test_from_csv_only_approved(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rules.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("spellings,status\npaisa|pese,approve\nmala|mila,reject\nkal|kl,\n")
            r = RuleSet.from_csv(p)
            self.assertIsNotNone(r.group_of("pese"))
            self.assertIsNone(r.group_of("mila"))
            self.assertIsNone(r.group_of("kl"))

    def test_fingerprint_stable(self):
        self.assertEqual(RuleSet().fingerprint(), RuleSet().fingerprint())


class DatasetToolTests(unittest.TestCase):
    ROWS = [("mala order nahi milala", "delivery"), ("mala order nhi milala", "delivery"),
            ("MALA ORDER NAHIII MILALA!!", "delivery"), ("payment fail ho gaya", "payment"),
            ("paymnt fail ho gaya", "payment"), ("refund chahiye", "payment"), ("late delivery", "delivery"),
            ("agent se baat karao", "human"), ("manager chahiye", "human")]

    def test_check_finds_near_duplicates_and_conflicts(self):
        c = check_dataset(self.ROWS)
        self.assertEqual(c.near_duplicate_groups, 2)
        self.assertEqual(c.conflicting_groups, 0)
        bad = self.ROWS + [("mala order nahi milala", "payment")]
        self.assertEqual(check_dataset(bad).conflicting_groups, 1)

    def test_split_has_no_leakage_and_is_deterministic(self):
        dev, test = split_dataset(self.ROWS, test_frac=0.5, seed=1)
        keys = lambda rows: {fr.DEFAULT_RULES.canonical_tokens(t) for t, _ in rows}
        self.assertFalse(keys(dev) & keys(test))
        self.assertEqual(len(dev) + len(test), len(self.ROWS))
        self.assertEqual(split_dataset(self.ROWS, 0.5, 1), (dev, test))

    def test_split_validates_fraction(self):
        with self.assertRaises(ValueError):
            split_dataset(self.ROWS, test_frac=1.5)

    def test_read_sentimix_synthetic_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("meta\t1\tpositive\nall\tEng\nthe\tEng\nbest\tEng\n\nmeta\t2\tNegative\nbakwas\tHin\nfilm\tEng\n")
            self.assertEqual(read_sentimix(p), [("all the best", "positive"), ("bakwas film", "negative")])

    def test_miner_proposes_and_flags_short_words(self):
        texts = ["mala bhaji", "mla bhaji", "mala chai", "mla chai", "nahi nhi nahin"] * 2
        cands = mine_variants(texts, min_count=2)
        spells = [set(c.spellings) for c in cands]
        self.assertIn({"mala", "mla"}, spells)
        self.assertTrue(all("nahi" not in s for s in spells), "already-known groups must not be re-proposed")
        c = next(c for c in cands if set(c.spellings) == {"mala", "mla"})
        self.assertIn("short", c.risk)

    def test_candidates_roundtrip_requires_human_approval(self):
        cands = mine_variants(["mala bhaji", "mla bhaji"] * 2, min_count=2)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.csv")
            write_candidates(cands, p)
            self.assertIsNone(RuleSet.from_csv(p).group_of("mla"))  # unapproved -> ignored


class SentimixCleanTests(unittest.TestCase):
    def _write(self, d, body):
        p = os.path.join(d, "s.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        return p

    def test_mentions_urls_hashtags_cleaned(self):
        body = ("meta\t173\tpositive\n@\tO\nBeingSalmanKhan\tEng\nIt\tEng\nsidhi\tHin\n#\tO\nbest\tEng\n"
                "https\tO\n\nmeta\t2\tneutral\n@\tO\nSomeone\tEng\n")
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, body)
            self.assertEqual(read_sentimix(p), [("It sidhi best", "positive")])   # 2nd tweet empty -> dropped
            self.assertEqual(read_sentimix(p, clean=False)[0][0], "@ BeingSalmanKhan It sidhi # best https")


class OverlapTests(unittest.TestCase):
    def test_cross_split_overlap(self):
        from fliprate.dataset import cross_split_overlap
        train = [("mala order nahi milala", "neg"), ("bahut accha", "pos")]
        ev = [("MALA ORDER NHI MILALA", "neg"), ("bahut acha hai", "pos"), ("mala order nahi milala", "pos")]
        r = cross_split_overlap(train, ev)
        self.assertEqual(r["overlapping"], 2)
        self.assertEqual(r["overlapping_with_different_label"], 1)


class AuditTests(unittest.TestCase):
    def test_audit_sample_and_summary(self):
        from fliprate.dataset import make_audit_sample, summarize_audit, _wilson
        rows = [("mala order ajun milala nahi", "x"), ("bahut accha product hai", "y")] * 5
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.csv")
            sample = make_audit_sample(rows, n=10, path=p)
            self.assertEqual(len(sample), 10)
            import csv
            with open(p, encoding="utf-8", newline="") as f:
                judged = list(csv.DictReader(f))
            for i, r in enumerate(judged):
                r["meaning_preserved"] = "yes" if i % 5 else "no"
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(judged[0])); w.writeheader(); w.writerows(judged)
            s = summarize_audit(p)
            self.assertEqual(s["ALL"]["judged"], 10)
            self.assertAlmostEqual(s["ALL"]["precision"], 0.8)
            self.assertLess(s["ALL"]["ci_low"], 0.8 < s["ALL"]["ci_high"])
        lo, hi = _wilson(0, 0); self.assertEqual((lo, hi), (0.0, 0.0))
        lo, hi = _wilson(10, 10); self.assertEqual(hi, 1.0)


if __name__ == "__main__":
    unittest.main()
