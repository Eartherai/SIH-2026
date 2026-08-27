"""Operational priority scoring.

Confidence and priority answer different questions and must never be conflated:

    CONFIDENCE  "How likely is it that this detection is a real man-made object?"
                -> a (calibrated) property of the evidence.

    PRIORITY    "How much should an operator care about it?"
                -> a policy decision that also depends on what the object is,
                   how big it is, and whether we can actually navigate to it.

A 55%-confidence ghost net that is 12 m long and well-geolocated is a more useful
tasking than a 95%-confidence 30 cm object with no position fix.

IMPORTANT SCOPE NOTE
--------------------
The weights below are an AQUA-SHIELD product convention chosen to be transparent
and adjustable. They are NOT an official marine-hazard standard; no such
standard was found for derelict-gear triage during this project's research, and
we do not claim one. Operators can retune `PriorityWeights` per campaign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class PriorityWeights:
    confidence: float = 0.35        # is it real?
    hazard_class: float = 0.25      # how damaging is this object type?
    size: float = 0.15              # bigger gear entangles more
    persistence: float = 0.15       # seen in many pings -> a real, sizeable object
    actionability: float = 0.10     # can a vessel actually be sent to it?


# Relative harm of each level-2 class. Derived from the PS 26057 background text:
# ghost fishing gear is called out as the most destructive category because it
# keeps killing after it is lost; wrecks and pipes are hazards to navigation and
# to trawling gear but are static and usually already charted.
CLASS_HAZARD = {
    "ghost_fishing_gear": 1.00,
    "shipwreck_structure": 0.70,
    "pipe_cylinder": 0.65,
    "mine_like_object": 0.85,
    "other_man_made": 0.55,
    "unknown_anomaly": 0.40,
    "bottom_object_uncertain": 0.30,
}


@dataclass
class PriorityResult:
    score: float                 # 0..100
    band: str                    # ROUTINE | ELEVATED | HIGH | URGENT
    components: dict             # each normalised factor, for audit
    weights: dict
    explanation: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 1)
        return d


def _band(score: float) -> str:
    if score >= 75:
        return "URGENT"
    if score >= 55:
        return "HIGH"
    if score >= 35:
        return "ELEVATED"
    return "ROUTINE"


def score_priority(*, confidence_pct: float, level2_class: str,
                   estimated_length_m: float | None, observation_count: int,
                   survey_quality: float, geolocated: bool,
                   geoloc_uncertainty_m: float | None,
                   weights: PriorityWeights | None = None) -> PriorityResult:
    w = weights or PriorityWeights()

    f_conf = max(0.0, min(confidence_pct / 100.0, 1.0))
    f_class = CLASS_HAZARD.get(level2_class, 0.4)

    # Size: log-scaled between 0.5 m and 20 m. Unknown size scores neutral (0.5)
    # rather than zero -- absence of a measurement is not evidence of smallness.
    if estimated_length_m is None or estimated_length_m <= 0:
        f_size = 0.5
        size_note = "unknown (no ground sample distance) - scored neutral"
    else:
        f_size = min(1.0, max(0.0, math.log10(max(estimated_length_m, 0.5) / 0.5)
                              / math.log10(20.0 / 0.5)))
        size_note = f"{estimated_length_m:.2f} m"

    # Persistence saturates: 8+ sightings adds no further evidence of realness.
    f_persist = min(1.0, math.log1p(max(observation_count, 1)) / math.log1p(8))

    if not geolocated:
        f_act = 0.15
        act_note = "no position fix - cannot be tasked directly"
    elif geoloc_uncertainty_m is None:
        f_act = 0.6
        act_note = "position known, uncertainty unreported"
    else:
        # 5 m -> ~1.0, 50 m -> ~0.35, 200 m -> ~0.1
        f_act = float(max(0.1, min(1.0, 5.0 / max(geoloc_uncertainty_m, 5.0)) ** 0.5))
        act_note = f"+/-{geoloc_uncertainty_m:.0f} m"

    q = max(0.0, min(survey_quality, 1.0))

    raw = (w.confidence * f_conf + w.hazard_class * f_class + w.size * f_size
           + w.persistence * f_persist + w.actionability * f_act)
    total_w = sum(asdict(w).values())
    # Poor data quality damps the whole score, but never below half: a hazard in
    # a noisy frame is still a hazard.
    score = 100.0 * (raw / total_w) * (0.5 + 0.5 * q)

    comps = {"confidence": round(f_conf, 3), "hazard_class": round(f_class, 3),
             "size": round(f_size, 3), "persistence": round(f_persist, 3),
             "actionability": round(f_act, 3), "survey_quality": round(q, 3)}
    expl = (f"class={level2_class} (harm {f_class:.2f}); confidence {confidence_pct:.0f}%; "
            f"size {size_note}; seen {observation_count}x; {act_note}; "
            f"frame quality {q:.2f}")
    return PriorityResult(score, _band(score), comps, asdict(w), expl)
