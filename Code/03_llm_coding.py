import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config as cfg

CATEGORIES = [
    "treasury_funding", "grants_ecosystem", "parameter_risk", "technical_upgrade",
    "deployment_integration", "governance_process", "tokenomics_incentives",
    "elections_personnel", "partnership", "other",
]

SYSTEM = """You are a careful research assistant coding DAO governance proposals for an academic content analysis.
Apply the codebook strictly and return only valid JSON."""

CODEBOOK = f"""CODEBOOK

1. category (choose exactly one): {", ".join(CATEGORIES)}
   treasury_funding: spending, budgets, treasury management or diversification
   grants_ecosystem: grants, funding programmes, public goods, ecosystem funds
   parameter_risk: changes to fees, rates, collateral, caps, risk parameters
   technical_upgrade: smart contract, protocol or infrastructure upgrades and new technical features
   deployment_integration: deploying on new chains, integrations, listings of new assets
   governance_process: rules of governance itself (voting, quorum, delegation, constitution, councils)
   tokenomics_incentives: token supply, emissions, staking, rewards, incentive programmes
   elections_personnel: elections, appointments, compensation of contributors or delegates
   partnership: agreements with external organisations
   other: anything else

2. innovation_type (choose exactly one):
   radical: introduces something fundamentally new for this DAO, such as a new product, new market,
            new business model, new organisational or governance structure, or a clear break from existing routines
   incremental: refines, adjusts, extends, renews or continues something that already exists
   none: no innovation content (e.g. routine elections, routine payments, ratifications, signalling only)

3. novelty_score (integer 1-5): 1 = routine continuation, 3 = moderate change, 5 = fundamentally new

4. scope (choose exactly one): technical, financial, organisational, ecosystem

5. requests_funding (true/false) and funding_usd (number in USD if a monetary amount is requested and stated, otherwise null;
   convert token amounts to USD only if the proposal states the USD value)

6. complexity (integer 1-5): how much expertise is needed to evaluate the proposal

7. confidence (number 0-1): your confidence in the innovation_type code

Return JSON with keys: category, innovation_type, novelty_score, scope, requests_funding, funding_usd,
complexity, confidence, rationale (max 30 words)."""


def prompt(row):
    return (f"{CODEBOOK}\n\nDAO: {row['space']}\nTitle: {row['title']}\nVoting options: {row['choices']}\n"
            f"Proposal text:\n<<<\n{row['body']}\n>>>")


class Coder:
    def __init__(self, model, path):
        from openai import OpenAI
        self.client = OpenAI()
        self.model, self.path, self.lock = model, path, threading.Lock()
        self.done = {}
        if path.exists():
            for l in open(path, encoding="utf-8"):
                r = json.loads(l)
                self.done[r["proposal_id"]] = r

    def code(self, row):
        if row["proposal_id"] in self.done:
            return self.done[row["proposal_id"]]
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=400,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(row)}])
                out = json.loads(r.choices[0].message.content)
                out.update(proposal_id=row["proposal_id"], model=self.model,
                           prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens)
                with self.lock:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(out) + "\n")
                    self.done[row["proposal_id"]] = out
                return out
            except Exception as e:
                time.sleep(2 ** attempt)
                err = e
        print(f"  failed {row['proposal_id']}: {err}")
        return None


def run(coder, rows, label):
    todo = [r for r in rows if r["proposal_id"] not in coder.done]
    print(f"{label}: {len(rows)} proposals, {len(todo)} to code with {coder.model}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(coder.code, r) for r in todo]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 100 == 0 or i + 1 == len(futs):
                print(f"  {i + 1}/{len(futs)}")


def normalise(df):
    df = df.copy()
    df["category"] = df["category"].where(df["category"].isin(CATEGORIES), "other")
    df["innovation_type"] = df["innovation_type"].astype(str).str.lower().str.strip()
    df["innovation_type"] = df["innovation_type"].where(df["innovation_type"].isin(["radical", "incremental", "none"]), "none")
    for c in ["novelty_score", "complexity", "confidence", "funding_usd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["requests_funding"] = df["requests_funding"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    return df


def main():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first")
    texts = pd.read_json(cfg.PROCESSED / "proposals_text.jsonl", lines=True)
    rows = texts.to_dict("records")
    est = sum(len(prompt(r)) for r in rows) / 4 / 1e6
    print(f"approx. input tokens for coder A: {est:.1f}M")

    a = Coder(cfg.CODER_A, cfg.CODING / "coder_A.jsonl")
    run(a, rows, "Coder A (all proposals)")
    rng = random.Random(cfg.SEED)
    sub = rows if len(rows) <= cfg.CODER_B_SAMPLE else rng.sample(rows, cfg.CODER_B_SAMPLE)
    b = Coder(cfg.CODER_B, cfg.CODING / "coder_B.jsonl")
    run(b, sub, "Coder B (validation subsample)")

    A = normalise(pd.DataFrame([a.done[r["proposal_id"]] for r in rows if r["proposal_id"] in a.done]))
    B = normalise(pd.DataFrame([b.done[r["proposal_id"]] for r in sub if r["proposal_id"] in b.done]))
    A.to_csv(cfg.PROCESSED / "coding_A.csv", index=False)
    B.to_csv(cfg.PROCESSED / "coding_B.csv", index=False)

    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import spearmanr
    m = A.merge(B, on="proposal_id", suffixes=("_A", "_B"))
    rel = []
    for c in ["category", "innovation_type", "scope", "requests_funding"]:
        rel.append(dict(variable=c, n=len(m), agreement=(m[c + "_A"] == m[c + "_B"]).mean(),
                        cohen_kappa=cohen_kappa_score(m[c + "_A"].astype(str), m[c + "_B"].astype(str))))
    rad_a, rad_b = (m["innovation_type_A"] == "radical").astype(int), (m["innovation_type_B"] == "radical").astype(int)
    rel.append(dict(variable="radical_vs_rest", n=len(m), agreement=(rad_a == rad_b).mean(),
                    cohen_kappa=cohen_kappa_score(rad_a, rad_b)))
    for c in ["novelty_score", "complexity"]:
        x = m[[c + "_A", c + "_B"]].dropna()
        rel.append(dict(variable=c, n=len(x), agreement=(x[c + "_A"] == x[c + "_B"]).mean(),
                        cohen_kappa=cohen_kappa_score(x[c + "_A"].round().astype(int), x[c + "_B"].round().astype(int),
                                                      weights="quadratic"),
                        spearman=spearmanr(x[c + "_A"], x[c + "_B"]).correlation))
    rel = pd.DataFrame(rel)
    rel.to_csv(cfg.TABLES / "T_coding_reliability.csv", index=False)
    print(rel.round(3).to_string(index=False))
    cost = A[["prompt_tokens", "completion_tokens"]].sum().to_dict()
    print(f"coder A tokens: {cost}")


if __name__ == "__main__":
    main()
