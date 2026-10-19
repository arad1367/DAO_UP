import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg

APPROVE = re.compile(r"^\W*(for|yes|yae|yay|aye|approve|approved|accept|in favou?r|support|ratify|adopt|execute)\b", re.I)
REJECT = re.compile(r"^\W*(against|no|nay|reject|deny|oppose|do not|don't|decline)\b", re.I)
ABSTAIN = re.compile(r"^\W*(abstain|abstention|neutral)\b", re.I)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def classify_choices(choices):
    a = [i for i, c in enumerate(choices) if APPROVE.search(str(c))]
    r = [i for i, c in enumerate(choices) if REJECT.search(str(c))]
    ab = [i for i, c in enumerate(choices) if ABSTAIN.search(str(c))]
    other = set(range(len(choices))) - set(a) - set(r) - set(ab)
    if len(a) == 1 and len(r) == 1 and not other:
        return a[0], r[0]
    return None, None


def passed(scores, ai, ri, total, quorum):
    return int(scores[ai] > scores[ri] and (not quorum or total >= quorum))


def proposal_measures(p, votes):
    scores = [float(x or 0) for x in (p.get("scores") or [])]
    total = float(p.get("scores_total") or 0)
    n_votes = int(p.get("votes") or 0)
    choices = p.get("choices") or []
    vp = np.array([float(v.get("vp") or 0) for v in votes])
    vp_sorted = np.sort(vp)[::-1]
    share = vp_sorted / total if total > 0 else vp_sorted * np.nan
    cum = np.cumsum(share)
    over = np.where(cum > 0.5)[0]
    nakamoto = int(over[0] + 1) if len(over) else np.nan
    covered = float(cum[-1]) if len(cum) else np.nan
    sorted_scores = sorted(scores, reverse=True)
    margin_all = (sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0)) / total if total else np.nan

    m = dict(
        proposal_id=p["id"], space=p["space"]["id"], space_name=p["space"].get("name") or p["space"]["id"],
        title=p.get("title") or "", created=p["created"],
        date=datetime.fromtimestamp(p["created"], tz=timezone.utc).strftime("%Y-%m-%d"),
        year=datetime.fromtimestamp(p["created"], tz=timezone.utc).year,
        duration_days=((p.get("end") or 0) - (p.get("start") or 0)) / 86400,
        vote_type=p.get("type"), n_choices=len(choices), voters=n_votes, vp_total=total,
        quorum=float(p.get("quorum") or 0), has_discussion=int(bool(p.get("discussion"))),
        body_words=len((p.get("body") or "").split()),
        top1_share=float(share[:1].sum()) if len(share) else np.nan,
        top5_share=float(share[:5].sum()) if len(share) else np.nan,
        top10_share=float(share[:10].sum()) if len(share) else np.nan,
        hhi=float((share ** 2).sum()) if len(share) else np.nan,
        nakamoto=nakamoto, top_votes_coverage=covered, margin_all=margin_all,
    )

    ai, ri = classify_choices(choices)
    m["binary"] = int(ai is not None)
    if ai is None or len(scores) <= max(ai, ri):
        return m
    m["approve_share"] = scores[ai] / total if total else np.nan
    m["margin"] = abs(scores[ai] - scores[ri]) / (scores[ai] + scores[ri]) if scores[ai] + scores[ri] else np.nan
    m["passed"] = passed(scores, ai, ri, total, m["quorum"])
    m["quorum_met"] = int(not m["quorum"] or total >= m["quorum"])

    order = np.argsort(-vp)
    for k in [1, 3, 5, 10]:
        s, t = list(scores), total
        for idx in order[:k]:
            ch = votes[idx].get("choice")
            w = float(votes[idx].get("vp") or 0)
            if isinstance(ch, int) and 1 <= ch <= len(s):
                s[ch - 1] -= w
                t -= w
        m[f"pivotal_top{k}"] = int(passed(s, ai, ri, t, m["quorum"]) != m["passed"])

    s, t, flip_k = list(scores), total, np.nan
    for j, idx in enumerate(order):
        ch = votes[idx].get("choice")
        w = float(votes[idx].get("vp") or 0)
        if isinstance(ch, int) and 1 <= ch <= len(s):
            s[ch - 1] -= w
            t -= w
        if passed(s, ai, ri, t, m["quorum"]) != m["passed"]:
            flip_k = j + 1
            break
    m["min_whales_to_flip"] = flip_k
    win = ai if m["passed"] else ri
    top10_win = sum(float(votes[i].get("vp") or 0) for i in order[:10] if votes[i].get("choice") == win + 1)
    m["top10_winning_side_share"] = top10_win / scores[win] if scores[win] else np.nan
    return m


def main():
    props = {p["id"]: p for p in load_jsonl(cfg.RAW / "proposals.jsonl") if not p["id"].startswith("__empty__")}
    sample = json.load(open(cfg.RAW / "sample_ids.json", encoding="utf-8"))
    votes = {r["proposal"]: r["votes"] for r in load_jsonl(cfg.RAW / "votes_top.jsonl")}
    rows = [proposal_measures(props[i], votes.get(i, [])) for i in sample if i in props and i in votes]
    df = pd.DataFrame(rows).sort_values(["space", "created"])
    df["log_voters"] = np.log1p(df["voters"])
    df["log_body_words"] = np.log1p(df["body_words"])
    df["log_vp_total"] = np.log1p(df["vp_total"])
    df["space_age_days"] = df.groupby("space")["created"].transform(lambda x: (x - x.min()) / 86400)
    df["prior_proposals"] = df.groupby("space").cumcount()
    df.to_csv(cfg.PROCESSED / "proposals_measures.csv", index=False)

    texts = []
    for i in df["proposal_id"]:
        p = props[i]
        texts.append(dict(proposal_id=i, space=p["space"]["id"], title=p.get("title") or "",
                          body=(p.get("body") or "")[:cfg.MAX_BODY_CHARS], choices=json.dumps(p.get("choices") or [])))
    pd.DataFrame(texts).to_json(cfg.PROCESSED / "proposals_text.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"proposals: {len(df)}  binary: {int(df['binary'].sum())}  spaces: {df['space'].nunique()}")
    print(df.groupby("space").agg(n=("proposal_id", "size"), binary=("binary", "sum"),
                                  median_voters=("voters", "median"),
                                  median_nakamoto=("nakamoto", "median")).to_string())


if __name__ == "__main__":
    main()
