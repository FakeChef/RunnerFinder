"""
Adaptive S5->S7 video bib decision (Track B, approved in B2; positions, thresholds and escalation
condition are copied unchanged from the validated implementation, video_bib_feasibility_B1/scripts).

    stage 1  S5 positions (i + 0.5) / 5 = 10, 30, 50, 70, 90 % of the duration
             -> VERIFIED if exactly one key has >= 3 votes
    stage 2  only if stage 1 is not VERIFIED and the best key has >= 2 votes:
             add the S7 positions 1.5/7 and 5.5/7 (21.4 %, 78.6 %) and re-vote the 7 frames
             -> VERIFIED if exactly one key has >= 4 votes
    else     not indexed: AMBIGUOUS if >= 2 keys each have >= 2 votes, or a conflict (several keys
             at the threshold); UNINDEXED otherwise. Neither asserts a runner identity.

Votes (B1/B2 rule):
  * candidates = the frame's deduplicated bibs (PhotoOCR) ACCEPTED by the noise filter
    (config/bib_rules.yaml) - "filtered" in B2;
  * vote key = numeric_bib (the digit run, leading zeros kept; what search matches); a bib
    without a single digit run uses its full form;
  * one frame gives at most one vote per key.
Safeguard (B2 recommendation, tightening only): keys containing one of the rule file's years
(e.g. "2026", "42026", "02026" - printed race year read as a bib) never vote.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STRATEGY = "adaptive_s5_s7/1"
S5_POSITIONS = tuple(round((i + 0.5) / 5, 6) for i in range(5))           # 0.1 .. 0.9
S7_POSITIONS = tuple(round((i + 0.5) / 7, 6) for i in range(7))
ESCALATION_POSITIONS = (S7_POSITIONS[1], S7_POSITIONS[5])                  # 0.214286, 0.785714
S5_THRESHOLD = 3
S7_THRESHOLD = 4
ESCALATE_MIN_VOTES = 2
AMBIGUOUS_MIN_VOTES = 2

VERIFIED, AMBIGUOUS, UNINDEXED = "VERIFIED", "AMBIGUOUS", "UNINDEXED"


@dataclass(frozen=True)
class FrameBib:
    bib: str                 # full normalised bib (e.g. "A15293", "15293")
    numeric: str | None      # numeric_bib
    confidence: float
    filter_accepted: bool
    filter_reasons: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return self.numeric or self.bib


def year_key(key: str, years: tuple[str, ...]) -> bool:
    digits = "".join(ch for ch in key if ch.isdigit())
    return any(y and y in digits for y in years)


def frame_keys(bibs: list[FrameBib], years: tuple[str, ...]) -> dict[str, float]:
    """Vote keys of one frame -> best confidence (one vote per key per frame)."""
    keys: dict[str, float] = {}
    for b in bibs:
        if not b.filter_accepted or year_key(b.key, years):
            continue
        keys[b.key] = max(keys.get(b.key, 0.0), float(b.confidence or 0.0))
    return keys


@dataclass
class Tally:
    votes: dict[str, int] = field(default_factory=dict)
    confs: dict[str, list[float]] = field(default_factory=dict)
    forms: dict[str, set[str]] = field(default_factory=dict)

    def ranked(self) -> list[tuple[str, int]]:
        return sorted(self.votes.items(), key=lambda kv: (-kv[1], kv[0]))


def tally(frames: list[list[FrameBib]], years: tuple[str, ...]) -> Tally:
    t = Tally()
    for bibs in frames:
        for k, c in frame_keys(bibs, years).items():
            t.votes[k] = t.votes.get(k, 0) + 1
            t.confs.setdefault(k, []).append(c)
        for b in bibs:
            if b.filter_accepted and not year_key(b.key, years):
                t.forms.setdefault(b.key, set()).add(b.bib)
    return t


def decide(t: Tally, threshold: int) -> tuple[str, str | None]:
    """(VERIFIED, key) when exactly one key reaches the threshold; else (status, None)."""
    passing = [k for k, v in t.ranked() if v >= threshold]
    if len(passing) == 1:
        return VERIFIED, passing[0]
    if len(passing) > 1:
        return AMBIGUOUS, None                          # conflict: never accepted
    multi = [k for k, v in t.votes.items() if v >= AMBIGUOUS_MIN_VOTES]
    return (AMBIGUOUS if len(multi) >= 2 else UNINDEXED), None


def should_escalate(status: str, t: Tally) -> bool:
    best = t.ranked()[0][1] if t.votes else 0
    return status != VERIFIED and best >= ESCALATE_MIN_VOTES


@dataclass
class Decision:
    status: str
    key: str | None
    escalated: bool
    tally: Tally
    frames_used: int

    def full_bib(self) -> str | None:
        """Best full form for the accepted key (prefixed form preferred when it was read)."""
        if self.key is None:
            return None
        forms = sorted(self.tally.forms.get(self.key, {self.key}), key=lambda f: (-len(f), f))
        return forms[0]


def decide_adaptive(s5_frames: list[list[FrameBib]], extra_frames_fn, years: tuple[str, ...]) -> Decision:
    """s5_frames: bibs of the 5 S5 frames (positions S5_POSITIONS order).
    extra_frames_fn(): bibs of the 2 escalation frames; called only when escalating."""
    t5 = tally(s5_frames, years)
    status, key = decide(t5, S5_THRESHOLD)
    if not should_escalate(status, t5):
        return Decision(status, key, False, t5, len(s5_frames))
    frames7 = list(s5_frames) + list(extra_frames_fn())
    t7 = tally(frames7, years)
    status, key = decide(t7, S7_THRESHOLD)
    return Decision(status, key, True, t7, len(frames7))
