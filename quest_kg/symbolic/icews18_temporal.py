"""Temporal symbolic constraints for ICEWS18.

Constraints (per paper/method.md §3.5):
  1. Temporal consistency: timestamps along a path must be monotonically non-decreasing
     (events build on prior events, not future ones).
  2. Actor-event compatibility: a pair (actor_type, event_code) must be admissible
     under the ICEWS code book (passed as set in constructor).
"""
from __future__ import annotations

from quest_kg.core.types import Path


class ICEWS18Checker:
    def __init__(self, actor_event_compat: set[tuple[str, str]] | None = None):
        """
        Args:
            actor_event_compat: set of admissible (actor_type, event_code) pairs.
                                If empty, the constraint is permissive.
        """
        self.actor_event_compat = actor_event_compat or set()

    def violates(self, path: Path) -> bool:
        # 1. Monotone timestamps
        last_ts: int | None = None
        for t in path.triples:
            if t.timestamp is not None:
                if last_ts is not None and t.timestamp < last_ts:
                    return True
                last_ts = t.timestamp
        # 2. Actor-event compatibility
        if self.actor_event_compat:
            for t in path.triples:
                if (t.s_type, t.r) not in self.actor_event_compat:
                    return True
        return False

    def violation_reason(self, path: Path) -> str | None:
        last_ts: int | None = None
        for i, t in enumerate(path.triples):
            if t.timestamp is not None:
                if last_ts is not None and t.timestamp < last_ts:
                    return f"non-monotone timestamp at edge {i}: {t.timestamp} < {last_ts}"
                last_ts = t.timestamp
        if self.actor_event_compat:
            for i, t in enumerate(path.triples):
                if (t.s_type, t.r) not in self.actor_event_compat:
                    return f"actor-event mismatch at edge {i}: ({t.s_type}, {t.r})"
        return None
