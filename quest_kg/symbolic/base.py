"""Base interface for per-task symbolic constraint checkers.

Each checker implements `violates(path)` returning True if the path violates any
constraint specific to that task. The exact rule sets live in the per-task files
in this package; the inference pipeline only ever sees the abstract interface.
"""
from __future__ import annotations

from typing import Protocol

from quest_kg.core.types import Path


class SymbolicChecker(Protocol):
    """Protocol for a per-task constraint checker.

    Implementations should be deterministic and side-effect-free given fixed
    schema / rule definitions; this matters for reproducibility.
    """

    def violates(self, path: Path) -> bool: ...
    def violation_reason(self, path: Path) -> str | None: ...
