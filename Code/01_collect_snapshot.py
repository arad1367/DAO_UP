import json
import random
import sys
import time
from datetime import datetime, timezone

import requests

import config as cfg

SPACE_Q = """
query ($ids: [String]) {
  spaces(first: 100, where: {id_in: $ids}) {
    id name proposalsCount followersCount network
  }
}"""

PROP_Q = """
query ($space: String!, $gt: Int!, $lt: Int!) {
  proposals(first: 1000, where: {space: $space, state: "closed", created_gt: $gt, created_lt: $lt},
            orderBy: "created", orderDirection: asc) {
    id title body choices start end snapshot state author created type quorum
    scores scores_total scores_state votes link discussion flagged
    space { id name }
  }
}"""

PROP_Q_SAFE = PROP_Q.replace(" scores_state", "").replace(" flagged", "")

VOTE_Q = """
query ($id: String!, $n: Int!) {
  votes(first: $n, where: {proposal: $id}, orderBy: "vp", orderDirection: desc) {
    voter vp choice created
  }
}"""


def ts(d):
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


class GraphQLError(Exception):
    pass


def gql(query, variables):
    headers = {"Content-Type": "application/json"}
    if cfg.SNAPSHOT_API_KEY:
        headers["x-api-key"] = cfg.SNAPSHOT_API_KEY
    for attempt in range(8):
        try:
            r = requests.post(cfg.SNAPSHOT_URL, json={"query": query, "variables": variables},
                              headers=headers, timeout=60)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            out = r.json()
            if "errors" in out:
                raise GraphQLError(out["errors"])
            time.sleep(cfg.REQUEST_PAUSE if not cfg.SNAPSHOT_API_KEY else 0.3)
            return out["data"]
        except GraphQLError:
            raise
        except Exception as e:
            wait = 5 * 2 ** attempt
            print(f"  error: {e} -> retry in {wait}s")
            time.sleep(wait)
    raise SystemExit("Snapshot API not reachable after retries")


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def collect_spaces():
    data = gql(SPACE_Q, {"ids": cfg.CANDIDATE_SPACES})
    spaces = data["spaces"] or []
    found = {s["id"] for s in spaces}
    for s in cfg.CANDIDATE_SPACES:
        if s not in found:
            print(f"  space not found: {s}")
    with open(cfg.RAW / "spaces.json", "w", encoding="utf-8") as f:
        json.dump(spaces, f, indent=2)
    return spaces


def collect_proposals(space_id):
    out, gt, lt = [], ts(cfg.START_DATE), ts(cfg.END_DATE)
    while True:
        try:
            batch = gql(PROP_Q, {"space": space_id, "gt": gt, "lt": lt})["proposals"]
        except GraphQLError as e:
            print(f"  full query rejected ({e}); using reduced field set")
            batch = gql(PROP_Q_SAFE, {"space": space_id, "gt": gt, "lt": lt})["proposals"]
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        gt = batch[-1]["created"]
    return out


def main():
    spaces = collect_spaces()
    print(f"{len(spaces)} spaces found")
    prop_path = cfg.RAW / "proposals.jsonl"
    done_spaces = {p["space"]["id"] for p in load_jsonl(prop_path)}
    for s in spaces:
        if s["id"] in done_spaces:
            continue
        props = collect_proposals(s["id"])
        print(f"  {s['id']}: {len(props)} closed proposals")
        with open(prop_path, "a", encoding="utf-8") as f:
            for p in props:
                f.write(json.dumps(p) + "\n")
        if not props:
            with open(prop_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": f"__empty__{s['id']}", "space": {"id": s["id"]}, "votes": 0}) + "\n")

    proposals = [p for p in load_jsonl(prop_path) if not p["id"].startswith("__empty__")]
    counts = {}
    for p in proposals:
        counts[p["space"]["id"]] = counts.get(p["space"]["id"], 0) + 1
    keep_spaces = {k for k, v in counts.items() if v >= cfg.MIN_PROPOSALS_PER_SPACE}
    eligible = [p for p in proposals
                if p["space"]["id"] in keep_spaces and p.get("type") in cfg.VOTE_TYPES
                and (p.get("votes") or 0) >= cfg.MIN_VOTERS and not p.get("flagged")
                and (p.get("scores_total") or 0) > 0]
    rng = random.Random(cfg.SEED)
    grouped = {}
    for p in sorted(eligible, key=lambda x: x["created"]):
        grouped.setdefault(p["space"]["id"], []).append(p)
    by_space = {k: (v if len(v) <= cfg.MAX_PROPOSALS_PER_SPACE else rng.sample(v, cfg.MAX_PROPOSALS_PER_SPACE))
                for k, v in grouped.items()}
    sample = [p for v in by_space.values() for p in v]
    fixed = cfg.RAW / "sample_ids.json"
    if fixed.exists():
        keep = set(json.load(open(fixed, encoding="utf-8")))
        sample = [p for p in proposals if p["id"] in keep]
        print(f"using existing sample definition: {len(keep)} proposal ids, {len(sample)} found")
    print(f"vote collection for {len(sample)} proposals in {len(by_space)} spaces")

    vote_path = cfg.RAW / "votes_top.jsonl"
    done = {r["proposal"] for r in load_jsonl(vote_path)}
    todo = [p for p in sample if p["id"] not in done]
    for i, p in enumerate(todo):
        votes = gql(VOTE_Q, {"id": p["id"], "n": cfg.TOP_VOTES_PER_PROPOSAL})["votes"]
        with open(vote_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"proposal": p["id"], "space": p["space"]["id"], "votes": votes}) + "\n")
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"  votes {i + 1}/{len(todo)}")
    if not fixed.exists():
        with open(fixed, "w", encoding="utf-8") as f:
            json.dump([p["id"] for p in sample], f)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
