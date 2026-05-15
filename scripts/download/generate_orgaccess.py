"""Generate OrgAccess: a synthetic enterprise access-control benchmark.

This is a from-scratch generator that produces a temporally evolving KG of
{User, Role, Resource, Policy, Context} entities + relations, with queries that
test dynamic authorization under changing policy state. Designed to be released
with the paper to ensure reproducibility (CIKM reviewers value this).

Schema:
  Entities:    User, Role, Resource, Policy, Context
  Relations:   has_role(User -> Role), grants(Role -> Resource), governed_by(Resource -> Policy),
               revoked_at(Role -> timestamp), valid_in(Policy -> Context),
               assigned_at(User -> Role -> timestamp), active_context(Context -> bool)

Queries:
  Each query (u, r, t) asks: at time `t` in current context, can `u` access `r`?
  Ground-truth labels computed by deterministic rule evaluation over the timestamped graph.

Usage:
    python -m scripts.download.generate_orgaccess --root /content/drive/MyDrive/quest_kg/data \
        --n_users 500 --n_resources 200 --n_policies 50 --n_contexts 8 \
        --n_queries 5000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import safe_target, write_marker


@dataclass
class Triple:
    s: str
    r: str
    o: str
    t: int = 0  # timestamp (event index); 0 means time-invariant


def generate(seed: int, n_users: int, n_resources: int, n_policies: int,
             n_contexts: int, n_queries: int, T_horizon: int = 200) -> dict:
    rng = random.Random(seed)

    users = [f"u_{i:04d}" for i in range(n_users)]
    roles = [f"r_{i:02d}" for i in range(max(8, n_users // 50))]
    resources = [f"res_{i:04d}" for i in range(n_resources)]
    policies = [f"pol_{i:03d}" for i in range(n_policies)]
    contexts = [f"ctx_{i:02d}" for i in range(n_contexts)]

    triples: list[Triple] = []

    # Static structure: role -> resource grants
    for role in roles:
        for res in rng.sample(resources, k=max(1, n_resources // 10)):
            triples.append(Triple(role, "grants", res))

    # Each resource governed by 1-2 policies
    for res in resources:
        for pol in rng.sample(policies, k=rng.choice([1, 1, 2])):
            triples.append(Triple(res, "governed_by", pol))

    # Each policy valid in some contexts
    for pol in policies:
        for ctx in rng.sample(contexts, k=rng.choice([1, 2, 3])):
            triples.append(Triple(pol, "valid_in", ctx))

    # Temporal: users get assigned and revoked from roles over time
    events: list[Triple] = []
    for t in range(1, T_horizon + 1):
        # 1-3 assignments per timestep
        for _ in range(rng.choice([1, 2, 3])):
            u, ro = rng.choice(users), rng.choice(roles)
            events.append(Triple(u, "has_role", ro, t))
        # occasional revocations
        if t > 20 and rng.random() < 0.3:
            ro = rng.choice(roles)
            events.append(Triple(ro, "revoked_at", str(t), t))

    triples.extend(events)

    # Active context per timestep (deterministic schedule)
    ctx_schedule = [contexts[t % n_contexts] for t in range(1, T_horizon + 1)]

    # Ground-truth oracle
    def can_access(u: str, res: str, t_query: int) -> bool:
        # what roles does u have at t_query?
        u_roles = set()
        for e in events:
            if e.r == "has_role" and e.s == u and 0 < e.t <= t_query:
                u_roles.add(e.o)
        # remove revoked roles
        revoked = {e.s for e in events if e.r == "revoked_at" and 0 < e.t <= t_query}
        u_roles -= revoked
        if not u_roles:
            return False
        # resource's grants
        granting_roles = {tr.s for tr in triples if tr.r == "grants" and tr.o == res}
        if not (u_roles & granting_roles):
            return False
        # policy must be valid in current context
        active_ctx = ctx_schedule[t_query - 1]
        govern_pols = {tr.o for tr in triples if tr.r == "governed_by" and tr.s == res}
        for pol in govern_pols:
            valid_ctxs = {tr.o for tr in triples if tr.r == "valid_in" and tr.s == pol}
            if active_ctx in valid_ctxs:
                return True
        return False

    # Generate queries
    queries = []
    for i in range(n_queries):
        u = rng.choice(users)
        res = rng.choice(resources)
        t_q = rng.randint(50, T_horizon)
        label = can_access(u, res, t_q)
        queries.append({
            "qid": i,
            "user": u,
            "resource": res,
            "timestamp": t_q,
            "context": ctx_schedule[t_q - 1],
            "label": 1 if label else 0,
            "question": f"At time {t_q} in context {ctx_schedule[t_q - 1]}, can {u} access {res}?",
        })

    return {
        "triples": [asdict(t) for t in triples],
        "context_schedule": ctx_schedule,
        "queries": queries,
        "stats": {
            "n_users": n_users,
            "n_roles": len(roles),
            "n_resources": n_resources,
            "n_policies": n_policies,
            "n_contexts": n_contexts,
            "n_triples": len(triples),
            "n_queries": len(queries),
            "T_horizon": T_horizon,
            "positive_rate": sum(q["label"] for q in queries) / len(queries),
        },
    }


def main(root: str, **kw) -> None:
    name = "orgaccess"
    out = safe_target(root, name)
    data = generate(**kw)
    # Splits 70/15/15
    rng = random.Random(kw["seed"])
    qids = list(range(len(data["queries"])))
    rng.shuffle(qids)
    n = len(qids)
    tr_idx = set(qids[: int(0.7 * n)])
    va_idx = set(qids[int(0.7 * n): int(0.85 * n)])
    train = [q for q in data["queries"] if q["qid"] in tr_idx]
    valid = [q for q in data["queries"] if q["qid"] in va_idx]
    test = [q for q in data["queries"] if q["qid"] not in tr_idx and q["qid"] not in va_idx]

    (out / "triples.json").write_text(json.dumps(data["triples"], indent=2))
    (out / "context_schedule.json").write_text(json.dumps(data["context_schedule"], indent=2))
    (out / "train.json").write_text(json.dumps(train, indent=2))
    (out / "valid.json").write_text(json.dumps(valid, indent=2))
    (out / "test.json").write_text(json.dumps(test, indent=2))
    (out / "stats.json").write_text(json.dumps(data["stats"], indent=2))
    write_marker(root, name)
    print(f"[orgaccess] OK -> {out} ({data['stats']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--n_resources", type=int, default=200)
    ap.add_argument("--n_policies", type=int, default=50)
    ap.add_argument("--n_contexts", type=int, default=8)
    ap.add_argument("--n_queries", type=int, default=5000)
    ap.add_argument("--T_horizon", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(
        root=args.root,
        n_users=args.n_users,
        n_resources=args.n_resources,
        n_policies=args.n_policies,
        n_contexts=args.n_contexts,
        n_queries=args.n_queries,
        T_horizon=args.T_horizon,
        seed=args.seed,
    )
