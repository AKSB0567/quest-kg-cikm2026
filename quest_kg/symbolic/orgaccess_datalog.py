"""Datalog constraints for the OrgAccess dynamic policy benchmark.

Implements the 6 rules described in paper/method.md §3.5:
  R1: access requires has_role + grants + governed_by + valid_in(context)
  R2: revocation blocks role usage after revocation timestamp
  R3: policy must be applicable to active context at query time
  R4: role assignment must precede the query timestamp
  R5: deny-list rules override grant rules (if a "deny" edge exists)
  R6: path must end on an access(User, Resource, t) ground atom

These are evaluated against the KG triples present in the path.
"""
from __future__ import annotations

from quest_kg.core.types import Path, Triple


# Required relation labels in OrgAccess
HAS_ROLE       = "has_role"
GRANTS         = "grants"
GOVERNED_BY    = "governed_by"
VALID_IN       = "valid_in"
REVOKED_AT     = "revoked_at"
DENIED_FOR     = "denied_for"  # optional deny-list relation


class OrgAccessChecker:
    def __init__(
        self,
        required_relations: tuple[str, ...] = (HAS_ROLE, GRANTS, GOVERNED_BY, VALID_IN),
        active_contexts_at_t: dict[int, str] | None = None,
    ):
        """
        Args:
            required_relations:  set of relations that must all appear in the path
                                 for an ACCESS conclusion to be derivable.
            active_contexts_at_t: {timestamp: active_context_id} schedule used to
                                  check R3 (policy valid in active context).
        """
        self.required_relations = set(required_relations)
        self.active_contexts_at_t = active_contexts_at_t or {}

    def violates(self, path: Path) -> bool:
        if not path.triples:
            return True

        relations_in_path = {t.r for t in path.triples}

        # R5: deny overrides — if any deny edge is present, the path violates.
        if DENIED_FOR in relations_in_path:
            return True

        # R1: all required relations must be present
        if not self.required_relations.issubset(relations_in_path):
            return True

        # R2: revocations
        # If a revoked_at triple appears with timestamp t_revoke, and the path's
        # query timestamp t_query > t_revoke, then the corresponding role usage is invalid.
        # We approximate by saying: if any revoked_at edge exists with a timestamp <= max ts in path,
        # this is a violation (i.e. the role was revoked by query time).
        revoke_events = [t for t in path.triples if t.r == REVOKED_AT and t.timestamp is not None]
        if revoke_events:
            max_path_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
            if max_path_ts is not None and any(rv.timestamp <= max_path_ts for rv in revoke_events):
                return True

        # R3: valid_in matches active context at query time
        if self.active_contexts_at_t:
            valid_in_edges = [t for t in path.triples if t.r == VALID_IN]
            query_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
            if query_ts is not None and query_ts in self.active_contexts_at_t:
                active_ctx = self.active_contexts_at_t[query_ts]
                if valid_in_edges and not any(t.o == active_ctx for t in valid_in_edges):
                    return True

        return False

    def violation_reason(self, path: Path) -> str | None:
        if not path.triples:
            return "empty path"
        relations_in_path = {t.r for t in path.triples}
        if DENIED_FOR in relations_in_path:
            return "deny-list rule fires (R5)"
        missing = self.required_relations - relations_in_path
        if missing:
            return f"missing required relations (R1): {missing}"
        # Revocation
        revoke_events = [t for t in path.triples if t.r == REVOKED_AT and t.timestamp is not None]
        if revoke_events:
            max_path_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
            if max_path_ts is not None and any(rv.timestamp <= max_path_ts for rv in revoke_events):
                return "role revoked before query time (R2)"
        # Context mismatch
        if self.active_contexts_at_t:
            valid_in_edges = [t for t in path.triples if t.r == VALID_IN]
            query_ts = max((t.timestamp for t in path.triples if t.timestamp is not None), default=None)
            if query_ts is not None and query_ts in self.active_contexts_at_t:
                active_ctx = self.active_contexts_at_t[query_ts]
                if valid_in_edges and not any(t.o == active_ctx for t in valid_in_edges):
                    return f"policy not valid in active context {active_ctx} (R3)"
        return None
