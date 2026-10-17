"""
Filter status of a stored bib candidate (V1.3.1). Standard library only: the GUI process reads it
without importing the OCR package (ocr/bib_filter.py computes the results and re-exports these names).
"""
from __future__ import annotations

ACCEPTED, UNCERTAIN, NOISE = "accepted", "uncertain", "noise"
# reasons that only say "not enough evidence": the candidate stays searchable (uncertain)
SOFT_REASONS = frozenset({"low_score", "short_number", "long_number", "possible_year", "text_context",
                          "small_text"})


def status_of(filter_result: dict | None) -> str:
    """Status of a stored evidence_json["filter"]. No filter result: accepted (V0.2 behaviour).
    Results stored before V1.3.1 have no status: accepted stays accepted, a rejection whose only reason
    is "low_score" (no evidence of noise, e.g. a 4-digit bib under rule set 1) is uncertain, anything
    else is noise. Reading rule only: nothing is re-scored."""
    if not filter_result:
        return ACCEPTED
    if filter_result.get("status") in (ACCEPTED, UNCERTAIN, NOISE):
        return filter_result["status"]
    if filter_result.get("accepted", True):
        return ACCEPTED
    reasons = set(filter_result.get("reasons") or ())
    return UNCERTAIN if reasons <= {"low_score"} else NOISE
