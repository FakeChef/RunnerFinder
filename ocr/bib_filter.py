"""
Bib candidate filter. Runs AFTER OCR on stored candidates; OCR itself is unchanged.

V1.3.1 (rule set 2, universal bib formats; docs: BIB_FORMAT_RULES_V1_3_1.md). Every candidate gets a
score from the rules in config/bib_rules.yaml and one of three statuses:

    accepted    score >= accept_threshold: a normal search result
    uncertain   no evidence of noise, but not enough for a bib (3 digits, a lone year, a number inside a
                text line): searchable, shown after the confirmed results and labelled
    noise       specific evidence (distance, bag-check line, excluded text, 1-2 digits, a year inside
                other text): hidden unless "Show filtered candidates"

The result is stored with the candidate, never instead of it:

    evidence_json["filter"] = {"accepted": false, "status": "noise", "score": 0,
                               "reasons": ["likely_storage_number"], "rules": "<hash>"}

"accepted" is kept with its V0.3.1 meaning, so V1.3.0 code still reads a re-scored index correctly
(it hides uncertain rows). Rows scored before V1.3.1 have no "status"; see status_of().
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RULES = {
    "rules_version": 2,
    "event_type": "any",
    # format eligibility: length of the numeric core (leading zeros count), optional letter prefix
    "common_lengths": [4, 5, 6], "common_length_score": 5,
    "short_lengths": [3], "short_length_score": 1,          # searchable, never confirmed by length alone
    "long_length_score": 1,                                  # 7+ digits: chip codes, phone numbers, ...
    "min_digits": 3, "short_score": -4,                      # 1-2 digits: distances, waves, labels
    "max_prefix_letters": 3, "clean_form_score": 1,          # "A1076", "UT1234", "60914"
    "high_confidence": 0.90, "high_confidence_score": 2,
    # race data: only when authoritative race data is provided (empty = universal rules)
    "bib_patterns": [], "pattern_score": 3,
    # evidence-based noise
    "exclude_patterns": ["KM", "km", "号", "存包", "WAVE", "组"], "exclude_score": -5,
    "distance_pattern": r"\d+(?:[.,]\d+)?\s*K\b", "distance_score": -5,
    "years": ["2024", "2025", "2026"], "year_in_text_score": -8, "possible_year_score": -4,
    "text_context_min_letters": 4, "text_context_score": -4,
    "penalty_patterns": [{"pattern": r"^\s*\d{1,2}\s*-\s*\d", "score": -8,
                          "reason": "likely_storage_number"}],
    "non_person_repeat_min_photos": 3, "non_person_repeat_score": -3,
    # the bib number is the largest text on a bib: text at most this fraction of the height of another
    # (non-noise) candidate read in the same person crop is secondary print (bag zone, sponsor, title)
    "small_text_ratio": 0.5, "small_text_score": -4,
    "accept_threshold": 5,
}

from search.filter_status import ACCEPTED, NOISE, SOFT_REASONS, UNCERTAIN, status_of  # noqa: F401,E402

NON_PERSON_SOURCES = ("full_image", "center_crop")
_DIGITS = re.compile(r"\d+")
_PREFIX = re.compile(r"^([A-Z]+)\d")
_WORD_SPLIT = re.compile(r"\s+")
_SEPARATORS = re.compile(r"[\s\-_.#·:/]+")


@dataclass(frozen=True)
class FilterResult:
    accepted: bool
    score: int
    reasons: tuple[str, ...]          # rules that spoke against the candidate ("low_score" if none)
    rules: str                        # short hash of the rules used
    status: str = ""                  # accepted | uncertain | noise

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "status": self.status, "score": self.score,
                "reasons": list(self.reasons), "rules": self.rules}


_ROTATED = {"0": "0", "1": "1", "2": "2", "5": "5", "6": "9", "8": "8", "9": "6"}


def _upside_down(digits) -> str | None:
    """The digits as read after a 180 degree rotation ("2026" -> "9202"); None if a digit has no such reading."""
    s = str(digits)
    return "".join(_ROTATED[ch] for ch in reversed(s)) if all(ch in _ROTATED for ch in s) else None


def _words(raw: str, bib: str) -> list[str]:
    """Letter words of a raw OCR line besides the bib itself ("2026 PARENT-CHILD" -> ["PARENTCHILD"])."""
    out = []
    for token in _WORD_SPLIT.split((raw or "").strip()):
        norm = _SEPARATORS.sub("", token.upper())
        if not norm or norm == bib:
            continue
        letters = "".join(ch for ch in norm if ch.isalpha())
        if len(letters) >= 2 and not (len(letters) <= 3 and norm[len(letters):].isdigit()
                                      and norm.startswith(letters)):
            out.append(letters)
    return out


@dataclass
class BibCandidateFilter:
    rules: dict = field(default_factory=lambda: dict(DEFAULT_RULES))

    def __post_init__(self):
        r = {**DEFAULT_RULES, **(self.rules or {})}
        self.rules = r
        self._patterns = [re.compile(p) for p in r["bib_patterns"]]
        self._distance = re.compile(r["distance_pattern"]) if r["distance_pattern"] else None
        self._penalties = [(re.compile(p["pattern"]), int(p["score"]), p.get("reason", "penalty_pattern"))
                           for p in r["penalty_patterns"]]
        # a printed year is also read upside down (wristbands, ribbons): "2026" rotated 180 deg is "9202"
        self._years = {str(y) for y in r["years"]} | {u for u in map(_upside_down, r["years"]) if u}
        blob = json.dumps(r, sort_keys=True, ensure_ascii=False)
        self.rules_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def load(cls, path: Path | str | None) -> "BibCandidateFilter":
        """Rules from a YAML file; built-in defaults if the file does not exist."""
        if path is None or not Path(path).is_file():
            return cls()
        import yaml
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        unknown = set(data) - set(DEFAULT_RULES)
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)} (rule set "
                             f"{DEFAULT_RULES['rules_version']}: see config/bib_rules.yaml)")
        return cls(data)

    def evaluate(self, text: str | None, numeric_identifier: str | None = None,
                 confidence: float | None = None, evidence: list[str] | None = None,
                 source: str | None = None, non_person_repeats: int = 0, *,
                 larger_text_ratio: float | None = None) -> FilterResult:
        """
        text: the full candidate bib ("A1076"); numeric_identifier: its digit part ("1076");
        evidence: raw OCR texts behind it; non_person_repeats: in how many photos
        the same number was read outside a person crop (event-level, optional);
        larger_text_ratio: its text height / the tallest other non-noise candidate in the same
        person crop (photo-level, see evaluate_photo; None = no such candidate).
        """
        r = self.rules
        bib = (text or "").strip().upper()
        raws = [t for t in (evidence or []) if t] or [bib]
        digits = numeric_identifier or "".join(_DIGITS.findall(bib))
        m = _PREFIX.match(bib)
        prefix = m.group(1) if m else ""
        score, reasons, noise = 0, [], []

        # format eligibility
        if len(digits) in r["common_lengths"]:
            score += r["common_length_score"]
        elif len(digits) in r["short_lengths"]:
            score += r["short_length_score"]
            reasons.append("short_number")
        elif len(digits) > max(r["common_lengths"]):
            score += r["long_length_score"]
            reasons.append("long_number")
        elif 0 < len(digits) < r["min_digits"]:
            score += r["short_score"]
            noise.append("too_few_digits")
        if re.fullmatch(rf"[A-Z]{{0,{int(r['max_prefix_letters'])}}}\d+", bib):
            score += r["clean_form_score"]
        if confidence is not None and confidence >= r["high_confidence"]:
            score += r["high_confidence_score"]
        if any(p.fullmatch(bib) for p in self._patterns):
            score += r["pattern_score"]

        # evidence-based noise
        if any(x in t for x in r["exclude_patterns"] for t in [bib, *raws]):
            score += r["exclude_score"]
            noise.append("excluded_text")
        if self._distance and any(self._distance.search(t.upper()) for t in raws):
            score += r["distance_score"]
            noise.append("distance")
        for pattern, value, reason in self._penalties:
            if any(pattern.search(t) for t in raws):
                score += value
                noise.append(reason)
        if digits in self._years and not prefix:
            if any(_words(t, bib) for t in raws):
                score += r["year_in_text_score"]
                noise.append("year")
            else:
                score += r["possible_year_score"]
                reasons.append("possible_year")
        if all(any(len(w) >= r["text_context_min_letters"] for w in _words(t, bib)) for t in raws):
            score += r["text_context_score"]
            reasons.append("text_context")
        if larger_text_ratio is not None and larger_text_ratio <= r["small_text_ratio"]:
            score += r["small_text_score"]
            reasons.append("small_text")
        if (source in NON_PERSON_SOURCES
                and non_person_repeats >= r["non_person_repeat_min_photos"]):
            score += r["non_person_repeat_score"]
            noise.append("repeated_outside_person")

        accepted = score >= r["accept_threshold"]
        if accepted:
            status = ACCEPTED
        elif noise:
            status = NOISE
        else:
            status = UNCERTAIN
            if not reasons:
                reasons.append("low_score")
        return FilterResult(accepted, score, tuple(noise + reasons), self.rules_hash, status)


@dataclass(frozen=True)
class PhotoCandidate:
    """One candidate of a photo (or video frame) with the photo-level context the filter needs."""
    bib: str | None
    numeric: str | None
    confidence: float | None
    raws: tuple[str, ...]
    source: str | None
    height: float | None                 # text line height in pixels
    regions: tuple[str, ...]             # person-crop / region ids it was read in
    non_person_repeats: int = 0


def evaluate_photo(bib_filter: BibCandidateFilter, cands: list[PhotoCandidate]) -> list[FilterResult]:
    """Filter results for all candidates of one photo. Pass 1 without size context; pass 2 with the text
    height relative to the tallest OTHER candidate that shares a region and is not noise in pass 1."""
    def run(c, ratio=None):
        return bib_filter.evaluate(c.bib, c.numeric, c.confidence, list(c.raws), c.source,
                                   c.non_person_repeats, larger_text_ratio=ratio)
    first = [run(c) for c in cands]
    out = []
    for i, c in enumerate(cands):
        refs = [o.height for j, o in enumerate(cands)
                if j != i and o.height and first[j].status != NOISE and set(o.regions) & set(c.regions)]
        ratio = c.height / max(refs) if refs and c.height else None
        out.append(run(c, ratio) if ratio is not None else first[i])
    return out


def evidence_texts(evidence: dict, ocr_text: str | None) -> list[str]:
    """Raw OCR texts behind a stored candidate (accepted readings, else ocr_text)."""
    texts = [r.get("raw_text") for r in (evidence.get("readings") or []) if r.get("raw_text")]
    return texts or ([ocr_text] if ocr_text else [])


def filtered_evidence_json(bib_filter: BibCandidateFilter, evidence_json: str | None,
                           bib: str | None, numeric: str | None, confidence: float | None,
                           ocr_text: str | None, source: str | None,
                           non_person_repeats: int = 0) -> tuple[str, FilterResult]:
    """evidence_json of a stored row with evidence["filter"] (re)computed; all other keys kept."""
    evidence = json.loads(evidence_json) if evidence_json else {}
    res = bib_filter.evaluate(bib, numeric, confidence, evidence_texts(evidence, ocr_text),
                              source, non_person_repeats)
    evidence["filter"] = res.as_dict()
    return json.dumps(evidence, ensure_ascii=False), res


_default: tuple[str, BibCandidateFilter] | None = None


def default_filter() -> BibCandidateFilter:
    """Filter with the rules file from config (reloaded if the path changes)."""
    global _default
    from config import get_config
    path = str(get_config().paths.bib_rules_file)
    if _default is None or _default[0] != path:
        _default = (path, BibCandidateFilter.load(path))
    return _default[1]
