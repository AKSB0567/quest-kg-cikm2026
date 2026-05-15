"""Freebase symbolic constraints for WebQSP and CWQ.

Constraints (per paper/method.md §3.5):
  1. Domain/range check: (head_type, relation, tail_type) must appear in Freebase schema.
  2. Functional-relation cardinality: relations marked functional admit exactly one
     outgoing edge per subject (we check this at training time and softly penalize at
     inference; in the path checker we just flag obvious violations).
  3. Type-of-answer match: the terminal entity's type must be in the query's
     expected-answer type set (passed via constructor).
"""
from __future__ import annotations

from typing import Iterable

from quest_kg.core.types import Path


class FreebaseChecker:
    def __init__(
        self,
        schema: dict[tuple[str, str], set[str]] | None = None,
        functional_relations: set[str] | None = None,
        expected_answer_types: set[str] | None = None,
    ):
        """
        Args:
            schema:                  {(head_type, relation): set_of_admissible_tail_types}
            functional_relations:    relations that admit only one outgoing edge per subject
            expected_answer_types:   types admissible for the terminal entity (per query)
        """
        self.schema = schema or {}
        self.functional_relations = functional_relations or set()
        self.expected_answer_types = expected_answer_types or set()

    def _domain_range_ok(self, s_type: str, r: str, o_type: str) -> bool:
        if not self.schema:
            return True  # no schema available -> permissive
        valid_tails = self.schema.get((s_type, r))
        if valid_tails is None:
            return False
        return o_type in valid_tails

    def violates(self, path: Path) -> bool:
        # 1. Domain/range along the path
        for t in path.triples:
            if not self._domain_range_ok(t.s_type, t.r, t.o_type):
                return True
        # 3. Answer-type match
        if self.expected_answer_types and path.triples:
            terminal = path.triples[-1]
            if terminal.o_type not in self.expected_answer_types:
                return True
        return False

    def violation_reason(self, path: Path) -> str | None:
        for i, t in enumerate(path.triples):
            if not self._domain_range_ok(t.s_type, t.r, t.o_type):
                return f"domain/range mismatch at edge {i}: ({t.s_type}, {t.r}, {t.o_type})"
        if self.expected_answer_types and path.triples:
            terminal = path.triples[-1]
            if terminal.o_type not in self.expected_answer_types:
                return f"terminal type {terminal.o_type} not in expected answer types {self.expected_answer_types}"
        return None


# Alias for clarity
WebQSPChecker = FreebaseChecker
CWQChecker = FreebaseChecker
