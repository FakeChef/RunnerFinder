"""Deterministic ranking of detected runners for Lite modes."""
from __future__ import annotations

import math


def rank_persons(persons, image_size: tuple[int, int],
                 area_weight: float = 0.45, center_weight: float = 0.35,
                 confidence_weight: float = 0.20):
    """Rank kept persons using normalized area, center proximity and confidence."""
    kept = [p for p in persons if p.kept]
    if not kept:
        return []
    width, height = image_size
    max_area = max(p.box.area for p in kept) or 1.0
    cx, cy = width / 2.0, height / 2.0
    max_distance = math.hypot(cx, cy) or 1.0
    ranked = []
    for p in kept:
        pcx = (p.box.x1 + p.box.x2) / 2.0
        pcy = (p.box.y1 + p.box.y2) / 2.0
        center = max(0.0, 1.0 - math.hypot(pcx - cx, pcy - cy) / max_distance)
        score = (area_weight * (p.box.area / max_area)
                 + center_weight * center
                 + confidence_weight * max(0.0, min(1.0, p.confidence)))
        ranked.append((p, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0].index))
