"""
Shared winding configuration for TractionMachinePM2D.
"""
from __future__ import annotations

from typing import List

# Default per-slot phase assignment for one 48s/8p sector (6 slots).
DEFAULT_PHASE_MAP_SECTOR = ("A+", "A+", "C-", "C-", "B+", "B+")

# Decode phase string -> (phase_index, sign)
PHASE_DECODE: dict[str, tuple[int, int]] = {
    "A+": (0, +1), "A-": (0, -1),
    "B+": (1, +1), "B-": (1, -1),
    "C+": (2, +1), "C-": (2, -1),
}


def winding_phase_map(ns: int) -> List[str]:
    """
    Return a phase-map list of length ns for a 3-phase q=ns//3 winding.

    Each phase occupies q consecutive slots before rotating:
      q=1: A+ C- B+  (ns=3)
      q=2: A+ A+ C- C- B+ B+  (ns=6, the 48s/8p default)
    Requires ns divisible by 3.
    """
    if ns % 3 != 0:
        raise ValueError(f"ns={ns} must be divisible by 3")
    q = ns // 3
    phases = ["A+", "C-", "B+"]
    return [phases[i // q] for i in range(ns)]


def phase_map_for_slots(ns: int) -> List[str]:
    """
    Return a phase-map list of length ns.

    For ns != 6, repeat the canonical sector pattern.
    """
    if ns <= 0:
        return []
    base = DEFAULT_PHASE_MAP_SECTOR
    return [base[i % len(base)] for i in range(ns)]
