"""Datalog constraints for the OrgAccess dynamic policy benchmark.

For Phase 2/3 the constraint set runs in two modes:
  - "strict"  (default off):  require all R1-R6 rules in a single path
  - "permissive" (default on): only flag clear violations (deny edges, revocation,
    context mismatch). Allows short paths to pass through inference.

The permissive mode is the right default for end-to-end QUEST-KG inference at
k=2 retrieval — strict mode is used only in the §4 ablation where we sweep
retrieval depth.
"""
from __future__ import annotations

from quest_kg.core.types import Path

# Required relation labels in OrgAccess
HAS_ROLE       = "has_role"
GRANTS         = "grants"
GOVERNED_BY    = "governed_by"
VALID_IN       = "valid_in"
REVOKED_AT     = "revoked_at"
DENIED_FOR     = "denied_for"


class OrgAccessChecker:
    def __init__(
        self,
        active_contexts_at_t: dict[int, str] | None = None,
        strict: bool = False,
    ):
        """
        Args:
            active_contexts_at_t: {timestamp: active_context_id} schedule.
            strict:               if True, require all of {has_role, grants, governed_by, valid_in}
                                  to appear in the path (Datalog R1). Default False for
                                  end-to-end inference -- short paths are allowed.
        """
        self.active_contexts_at_t = active_contexts_at_t or {}
        self.strict = strict
        self.required_relations = {HAS_ROLE, GRANTS, GOVERNED_BY, VALID_IN}

    def violates(self, path: Path) -> bool:
        if not path.triples:
            return True

        relations_in_path = {t.r for t in path.triples}

        # R5: deny edges override (always enforced)
        if DENIED_FOR in relations_in_path:
            return True

        # R1: strict-mode required-relation check
        if self.strict and not self.required_relations.issubset(relations_in_path):
            return True

        # R2: revocation (always enforced when revoke edge is in the path)
        revoke_events = [t for t in path.triples if t.r == REVOKED_AT and t.timestamp is not None]
        if revoke_events:
            max_path_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
            if max_path_ts is not None and any(rv.timestamp <= max_path_ts for rv in revoke_events):
                return True

        # R3: context mismatch (only checked when both context schedule + valid_in edges present)
        if self.active_contexts_at_t:
            valid_in_edges = [t for t in path.triples if t.r == VALID_IN]
            if valid_in_edges:
                query_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
                if query_ts is not None and query_ts in self.active_contexts_at_t:
                    active_ctx = self.active_contexts_at_t[query_ts]
                    if not any(t.o == active_ctx for t in valid_in_edges):
                        return True

        return False

    def violation_reason(self, path: Path) -> str | None:
        if not path.triples:
            return "empty path"
        relations_in_path = {t.r for t in path.triples}
        if DENIED_FOR in relations_in_path:
            return "deny-list rule fires (R5)"
        if self.strict and not self.required_relations.issubset(relations_in_path):
            return f"missing required relations (R1): {self.required_relations - relations_in_path}"
        revoke_events = [t for t in path.triples if t.r == REVOKED_AT and t.timestamp is not None]
        if revoke_events:
            max_path_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
            if max_path_ts is not None and any(rv.timestamp <= max_path_ts for rv in revoke_events):
                return "role revoked before query time (R2)"
        if self.active_contexts_at_t:
            valid_in_edges = [t for t in path.triples if t.r == VALID_IN]
            if valid_in_edges:
                query_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
                if query_ts is not None and query_ts in self.active_contexts_at_t:
                    active_ctx = self.active_contexts_at_t[query_ts]
                    if not any(t.o == active_ctx for t in valid_in_edges):
                        return f"policy not valid in active context {active_ctx} (R3)"
        return None
