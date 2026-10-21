"""Dataset tools: validate, split without leakage, read SentiMix, and mine candidate spelling variants."""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .evaluation import Row, load_dataset
from .rules import DEFAULT_RULES, WORD_RE, RuleSet, word_key


def write_csv(rows: Sequence[Row], path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)


# ------------------------------------------------------------------------------ checks
@dataclass
class DataCheck:
    n: int
    label_counts: Dict[str, int]
    exact_duplicates: int
    near_duplicate_groups: int
    conflicting_groups: int
    empty: int
    non_latin: int
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        lines = [f"Rows: {self.n}", "Labels: " + ", ".join(f"{k}={v}" for k, v in sorted(self.label_counts.items())),
                 f"Exact duplicates: {self.exact_duplicates} | near-duplicate groups: {self.near_duplicate_groups} "
                 f"| groups with conflicting labels: {self.conflicting_groups}"]
        lines += [f"WARNING: {w}" for w in self.warnings] or ["No problems found."]
        return "\n".join(lines)


def _groups(rows: Sequence[Row], rules: RuleSet) -> Dict[Tuple[str, ...], List[int]]:
    g: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i, (text, _) in enumerate(rows):
        g[rules.canonical_tokens(text)].append(i)
    return g


def check_dataset(source, rules: Optional[RuleSet] = None, min_per_label: int = 20) -> DataCheck:
    """Find problems that would make results misleading: duplicates, label conflicts, tiny classes."""
    rules = rules or DEFAULT_RULES
    rows = load_dataset(source)
    counts = Counter(l for _, l in rows)
    exact = len(rows) - len({t.strip().lower() for t, _ in rows})
    groups = _groups(rows, rules)
    near = sum(1 for idx in groups.values() if len(idx) > 1)
    conflicts = sum(1 for idx in groups.values() if len({rows[i][1] for i in idx}) > 1)
    empty = sum(1 for t, _ in rows if len(t.split()) < 1)
    non_latin = sum(1 for t, _ in rows if sum(c.isalpha() and not c.isascii() for c in t) > 0.5 * max(1, sum(c.isalpha() for c in t)))
    w = []
    if conflicts:
        w.append(f"{conflicts} group(s) of near-identical messages have different labels; fix the labels")
    if near:
        w.append(f"{near} group(s) of near-duplicate messages (same words, different spelling); keep them in ONE split")
    small = [k for k, v in counts.items() if v < min_per_label]
    if small:
        w.append(f"labels with fewer than {min_per_label} examples: {', '.join(sorted(small))}; metrics will be unstable")
    if non_latin:
        w.append(f"{non_latin} message(s) are mostly non-Latin script; FlipRate v0.1 targets romanized text")
    if len(rows) < 100:
        w.append(f"only {len(rows)} rows; fine for a demo, too small for published claims")
    return DataCheck(len(rows), dict(counts), exact, near, conflicts, empty, non_latin, w)


def split_dataset(source, test_frac: float = 0.6, seed: int = 0, rules: Optional[RuleSet] = None) -> Tuple[List[Row], List[Row]]:
    """Split into (dev, test) with NO leakage: messages that are spelling variants of each other stay together.
    Stratified by label and deterministic for a given seed."""
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    rules = rules or DEFAULT_RULES
    rows = load_dataset(source)
    groups = _groups(rows, rules)
    by_label: Dict[str, List[Tuple[str, List[int]]]] = defaultdict(list)
    for key, idx in groups.items():
        label = Counter(rows[i][1] for i in idx).most_common(1)[0][0]
        h = hashlib.sha256(f"{seed}|{' '.join(key)}".encode()).hexdigest()
        by_label[label].append((h, idx))
    dev, test = [], []
    for label, items in by_label.items():
        items.sort()
        n_test = max(1, round(len(items) * test_frac)) if len(items) > 1 else 0
        for pos, (_, idx) in enumerate(items):
            (test if pos < n_test else dev).extend(rows[i] for i in idx)
    return dev, test


# ---------------------------------------------------------------------------- SentiMix
def _clean_tokens(tokens: List[str]) -> List[str]:
    """Drop @mentions and URLs (as other SentiMix papers do), keep hashtag words."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t == "@":                                   # "@" "UserName" are two tokens in SentiMix
            i += 2
        elif t.startswith("@") and len(t) > 1:
            i += 1
        elif t == "#":
            i += 1
        elif t.startswith("#") and len(t) > 1:
            out.append(t[1:]); i += 1
        elif t.lower().startswith(("http", "www.")):
            i += 1
        else:
            out.append(t); i += 1
    return out


def read_sentimix(path, clean: bool = True) -> List[Row]:
    """Read a SemEval-2020 Task 9 (SentiMix) file: blocks of `meta <id> <sentiment>` followed by one
    `token <language-tag>` line per word (CoNLL style, per the task papers).

    clean=True drops @mentions and URL tokens and keeps hashtag words. How URLs are split into tokens
    in the real files is unverified, so print a few rows and compare with your copy.
    """
    rows: List[Row] = []
    tokens: List[str] = []
    label: Optional[str] = None

    def flush():
        nonlocal tokens, label
        toks = _clean_tokens(tokens) if clean else tokens
        if toks and label:
            rows.append((" ".join(toks), label))
        tokens, label = [], None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                flush()
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if parts[0] == "meta":
                flush()
                label = parts[2].strip().lower() if len(parts) > 2 else None
            else:
                tokens.append(parts[0])
    flush()
    return rows


def cross_split_overlap(train, evaluation, rules: Optional[RuleSet] = None) -> Dict[str, float]:
    """How many evaluation messages (up to spelling variation) also appear in the training data?
    Overlap inflates accuracy, and flip-rate results are not trustworthy if it is large."""
    rules = rules or DEFAULT_RULES
    tr, ev = load_dataset(train), load_dataset(evaluation)
    seen: Dict[Tuple[str, ...], set] = defaultdict(set)
    for text, label in tr:
        seen[rules.canonical_tokens(text)].add(label)
    overlap = conflict = 0
    for text, label in ev:
        labels = seen.get(rules.canonical_tokens(text))
        if labels:
            overlap += 1
            conflict += label not in labels
    return {"eval_rows": len(ev), "overlapping": overlap, "overlap_fraction": overlap / len(ev),
            "overlapping_with_different_label": conflict}


# ---------------------------------------------------------------------------- mining
def _skeleton(key: str) -> str:
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiou]", "", key))


def _edit_distance(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Candidate:
    spellings: List[str]          # most frequent first
    counts: List[int]
    risk: str                     # "", or why a human must look twice

    @property
    def total(self) -> int:
        return sum(self.counts)


def mine_variants(texts: Iterable[str], min_count: int = 2, max_edit: int = 2,
                  rules: Optional[RuleSet] = None) -> List[Candidate]:
    """Propose spelling groups from real messages for a HUMAN to approve. Never applied automatically.

    Words with the same consonant skeleton, the same first letter and a small edit distance are grouped.
    Short words collide easily (Marathi "mala" vs "mila"), so they are flagged.
    """
    rules = rules or DEFAULT_RULES
    raw_counts: Counter = Counter()
    for t in texts:
        for w in WORD_RE.findall(t):
            if w.isascii() and w.isalpha() and len(w) >= 3:
                raw_counts[w.lower()] += 1
    key_counts: Counter = Counter()
    key_forms: Dict[str, Counter] = defaultdict(Counter)
    for w, c in raw_counts.items():
        k = word_key(w)
        key_counts[k] += c
        key_forms[k][w] += c

    buckets: Dict[str, List[str]] = defaultdict(list)
    for k in key_counts:
        sk = _skeleton(k)
        if len(sk) >= 2:
            buckets[sk].append(k)

    out: List[Candidate] = []
    for keys in buckets.values():
        parent = {k: k for k in keys}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if a[0] == b[0] and _edit_distance(a, b, max_edit) <= max_edit:
                    parent[find(a)] = find(b)
        clusters: Dict[str, List[str]] = defaultdict(list)
        for k in keys:
            clusters[find(k)].append(k)
        for members in clusters.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda k: -key_counts[k])
            if sum(key_counts[k] for k in members) < min_count:
                continue
            if len({rules.key_to_group.get(k) for k in members}) == 1 and rules.key_to_group.get(members[0]) is not None:
                continue  # already covered by an existing group
            spell = [key_forms[k].most_common(1)[0][0] for k in members]
            risks = []
            if any(len(k) <= 4 for k in members):
                risks.append("short words collide easily")
            if any(s in rules.negations or s in rules.ambiguous for s in spell):
                risks.append("contains a negation or ambiguous word")
            out.append(Candidate(spell, [key_counts[k] for k in members], "; ".join(risks)))
    out.sort(key=lambda c: -c.total)
    return out


def write_candidates(cands: Sequence[Candidate], path) -> None:
    """Write candidates for review. A human fills `status` with approve/reject, then RuleSet.from_csv uses it."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spellings", "counts", "risk", "status"])
        for c in cands:
            w.writerow(["|".join(c.spellings), "|".join(map(str, c.counts)), c.risk, ""])


# ------------------------------------------------------------------------------ audit
def make_audit_sample(source, n: int = 100, seed: int = 0, rules: Optional[RuleSet] = None, path=None) -> List[Dict[str, str]]:
    """Sample generated variants for a fluent speaker to judge. Fill `meaning_preserved` with yes/no."""
    import random
    from .perturbation import perturb
    rules = rules or DEFAULT_RULES
    rows = load_dataset(source)
    pool = []
    for i, (text, _) in enumerate(rows):
        for v in perturb(text, 4, seed=seed + i, rules=rules):
            pool.append({"original": text, "variant": v.text, "kind": v.kind, "meaning_preserved": ""})
    random.Random(seed).shuffle(pool)
    sample = pool[:n]
    if path:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["original", "variant", "kind", "meaning_preserved"])
            w.writeheader()
            w.writerows(sample)
    return sample


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_audit(path) -> Dict[str, Dict[str, float]]:
    """Precision of the "safe variant" claim per change type, with 95% Wilson intervals.
    Rows whose `meaning_preserved` is blank are ignored."""
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            a = (r.get("meaning_preserved") or "").strip().lower()
            if a not in ("yes", "y", "no", "n"):
                continue
            for key in (r["kind"], "ALL"):
                stats[key][1] += 1
                stats[key][0] += a in ("yes", "y")
    out = {}
    for k, (yes, n) in stats.items():
        lo, hi = _wilson(yes, n)
        out[k] = {"judged": n, "precision": yes / n, "ci_low": lo, "ci_high": hi}
    return out
