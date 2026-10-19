import json
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config as cfg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, GREY, ORANGE = "#2f6db5", "#9aa5b1", "#d9822b"
REPORT = []
FULL = []

CONTROLS = "log_body_words + duration_days + requests_funding + complexity + has_discussion + C(category)"
FE = "C(space) + C(year)"


def load():
    m = pd.read_csv(cfg.PROCESSED / "proposals_measures.csv")
    a = pd.read_csv(cfg.PROCESSED / "coding_A.csv")
    keep = ["proposal_id", "category", "innovation_type", "novelty_score", "scope", "requests_funding",
            "funding_usd", "complexity", "confidence"]
    df = m.merge(a[keep], on="proposal_id", how="inner")
    df["radical"] = (df["innovation_type"] == "radical").astype(int)
    df["no_innovation"] = (df["innovation_type"] == "none").astype(int)
    df["log_funding_usd"] = np.log1p(df["funding_usd"].fillna(0))
    df["complexity"] = df["complexity"].fillna(df["complexity"].median())
    df["duration_days"] = df["duration_days"].clip(0, 60)
    return df


def save(df, name, caption):
    df.to_csv(cfg.TABLES / f"{name}.csv", index=False)
    REPORT.append(f"\n## {name}: {caption}\n\n" + df.to_markdown(index=False, floatfmt=".4g") + "\n")


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def fit(formula, data, kind, label, terms):
    cols = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)) & set(data.columns))
    d = data.dropna(subset=cols).copy()
    dv = formula.split("~")[0].strip()
    if kind in ("logit", "lpm"):
        var = d.groupby("space")[dv].transform("std").fillna(0)
        d = d[var > 0]
    if len(d) < 30 or d["space"].nunique() < 2:
        return None
    groups = pd.Categorical(d["space"]).codes
    ck = dict(cov_type="cluster", cov_kwds={"groups": groups})
    try:
        if kind == "logit":
            res = smf.logit(formula, d).fit(disp=0, maxiter=200, method="bfgs", **ck)
            fitstat = res.prsquared
        elif kind == "poisson":
            import statsmodels.api as sm
            res = smf.glm(formula, d, family=sm.families.Poisson()).fit(**ck)
            fitstat = 1 - res.deviance / res.null_deviance
        else:
            res = smf.ols(formula, d).fit(**ck)
            fitstat = res.rsquared
    except Exception as e:
        print(f"  {label} failed: {e}")
        return None
    FULL.append(f"\n===== {label} ({kind}) =====\n{res.summary()}\n")
    out = dict(model=label, kind=kind, n=int(res.nobs), daos=d["space"].nunique(), fit=fitstat)
    for t in terms:
        if t in res.params:
            out[f"{t}_b"] = res.params[t]
            out[f"{t}_se"] = res.bse[t]
            out[f"{t}_p"] = res.pvalues[t]
            out[f"{t}_txt"] = f"{res.params[t]:.3f}{stars(res.pvalues[t])} ({res.bse[t]:.3f})"
            if kind == "logit":
                out[f"{t}_oddsratio"] = float(np.exp(res.params[t]))
    if kind == "logit":
        try:
            me = res.get_margeff(at="overall").summary_frame()
            for t in terms:
                if t in me.index:
                    out[f"{t}_AME"] = me.loc[t, "dy/dx"]
                    out[f"{t}_AME_p"] = me.loc[t, "Pr(>|z|)"]
        except Exception:
            pass
    return out


def descriptives(df):
    b = df[df["binary"] == 1]
    t1 = df.groupby("space_name").agg(
        proposals=("proposal_id", "size"), first=("date", "min"), last=("date", "max"),
        binary=("binary", "sum"), median_voters=("voters", "median"),
        median_top1_share=("top1_share", "median"), median_nakamoto=("nakamoto", "median"),
        radical_pct=("radical", "mean"),
    ).reset_index()
    t1 = t1.merge(b.groupby("space_name").agg(pass_rate=("passed", "mean"),
                                              pivotal_top5_pct=("pivotal_top5", "mean")).reset_index(),
                  on="space_name", how="left")
    total = pd.DataFrame([dict(space_name="All", proposals=len(df), first=df["date"].min(), last=df["date"].max(),
                               binary=int(df["binary"].sum()), median_voters=df["voters"].median(),
                               median_top1_share=df["top1_share"].median(), median_nakamoto=df["nakamoto"].median(),
                               radical_pct=df["radical"].mean(), pass_rate=b["passed"].mean(),
                               pivotal_top5_pct=b["pivotal_top5"].mean())])
    save(pd.concat([t1.sort_values("proposals", ascending=False), total]), "T1_sample_by_dao", "Sample by DAO")

    cols = ["voters", "log_voters", "vp_total", "top1_share", "top5_share", "top10_share", "hhi", "nakamoto",
            "margin", "passed", "pivotal_top1", "pivotal_top5", "pivotal_top10", "min_whales_to_flip", "radical",
            "no_innovation", "novelty_score", "complexity", "requests_funding", "body_words", "duration_days"]
    cols = [c for c in cols if c in df]
    d = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index().rename(columns={"index": "variable"})
    save(d, "T2_descriptives", "Descriptive statistics")

    ct = pd.crosstab(df["category"], df["innovation_type"], margins=True).reset_index()
    save(ct, "T3_category_by_innovation", "Proposal category by innovation type (coder A)")


def concentration(df):
    b = df[df["binary"] == 1]
    rows = []
    for k in [1, 3, 5, 10]:
        c = f"pivotal_top{k}"
        rows.append(dict(scenario=f"remove top {k} voter(s)", share_outcomes_flip=b[c].mean(),
                         n_flip=int(b[c].sum()), n=len(b)))
    f = b["min_whales_to_flip"]
    rows.append(dict(scenario="median voters to remove for a flip", share_outcomes_flip=f.median(), n=int(f.notna().sum())))
    rows.append(dict(scenario="share decided by top-10 alone (top10 share of winning side > 0.5)",
                     share_outcomes_flip=(b["top10_winning_side_share"] > 0.5).mean(), n=len(b)))
    save(pd.DataFrame(rows), "T4_pivotality", "Counterfactual pivotality of the largest voters (binary proposals)")

    yr = df.groupby("year").agg(proposals=("proposal_id", "size"), median_voters=("voters", "median"),
                                median_top1=("top1_share", "median"), median_top10=("top10_share", "median"),
                                median_nakamoto=("nakamoto", "median")).reset_index()
    yr = yr.merge(b.groupby("year").agg(pivotal_top5=("pivotal_top5", "mean"), pass_rate=("passed", "mean")).reset_index(),
                  on="year", how="left")
    save(yr, "T5_trends_by_year", "Participation and concentration by year")

    order = list(df.groupby("space_name")["top1_share"].median().sort_values().index)
    n = len(order)
    fig, ax = plt.subplots(1, 2, figsize=(11, max(4, 0.32 * n + 1.5)), sharey=True,
                           gridspec_kw={"wspace": 0.08})
    data = [df.loc[df["space_name"] == s, "top1_share"].dropna() for s in order]
    ax[0].boxplot(data, positions=np.arange(n), vert=False, showfliers=False, patch_artist=True, widths=0.6,
                  boxprops=dict(facecolor="#dbe6f4", color=BLUE), medianprops=dict(color=BLUE, linewidth=1.5),
                  whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
    ax[0].set_yticks(np.arange(n), order, fontsize=9)
    ax[0].set_ylim(-0.7, n - 0.3)
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("Voting-power share of the largest voter")
    ax[0].set_title("A. Largest-voter share by DAO", loc="left")
    ax[0].grid(axis="x", color="#e5e5e5", linewidth=0.6)
    pv = b.groupby("space_name")[["pivotal_top1", "pivotal_top5", "pivotal_top10"]].mean().reindex(order)
    y = np.arange(n)
    h = 0.26
    for i, (c, col) in enumerate(zip(pv.columns, [GREY, BLUE, ORANGE])):
        ax[1].barh(y + (i - 1) * h, pv[c].fillna(0), h, color=col, label=c.replace("pivotal_top", "Top "))
    for j, s in enumerate(order):
        if pd.isna(pv.loc[s, "pivotal_top1"]):
            ax[1].text(0.01, j, "no approve/reject votes", va="center", fontsize=8, color="#777777")
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("Share of outcomes that flip without these voters")
    ax[1].set_title("B. Pivotality of the largest voters", loc="left")
    ax[1].grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax[1].tick_params(axis="y", length=0)
    ax[1].legend(frameon=False, title="Voters removed", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07 * 26 / n))
    plt.tight_layout()
    plt.savefig(cfg.FIGURES / "F1_concentration_pivotality.png")
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(yr["year"], yr["median_nakamoto"], "-o", color=BLUE)
    ax[0].set_ylabel("Median Nakamoto coefficient")
    ax[0].set_ylim(0, max(5, float(yr["median_nakamoto"].max()) + 1))
    ax[0].yaxis.get_major_locator().set_params(integer=True)
    ax[0].set_title("A. Voters needed for a majority", loc="left")
    ax[1].plot(yr["year"], yr["median_top10"], "-o", color=BLUE, label="Top-10 share")
    ax[1].plot(yr["year"], yr["pivotal_top5"], "-s", color=ORANGE, label="Outcomes flipped by top-5")
    ax[1].set_ylim(0, 1)
    ax[1].legend(frameon=False)
    ax[1].set_title("B. Concentration and pivotality over time", loc="left")
    for a in ax:
        a.set_xticks(yr["year"])
    plt.tight_layout()
    plt.savefig(cfg.FIGURES / "F2_trends.png")
    plt.close()


def innovation_figure(df):
    b = df[df["binary"] == 1]
    order = ["none", "incremental", "radical"]
    stats = []
    for t in order:
        for var, d in [("log_voters", df), ("margin", b), ("passed", b), ("pivotal_top5", b)]:
            x = d.loc[d["innovation_type"] == t, var].dropna()
            stats.append(dict(innovation_type=t, variable=var, mean=x.mean(), ci=1.96 * x.std(ddof=1) / np.sqrt(len(x))
                              if len(x) > 1 else np.nan, n=len(x)))
    s = pd.DataFrame(stats)
    save(s, "T6_outcomes_by_innovation", "Unadjusted means by innovation type")
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    titles = {"log_voters": "Participation (log voters)", "margin": "Vote margin", "passed": "Approval rate",
              "pivotal_top5": "Flipped by top-5"}
    for a, v in zip(ax, titles):
        q = s[s["variable"] == v].set_index("innovation_type").loc[order]
        a.bar(order, q["mean"], yerr=q["ci"], color=[GREY, "#7fa6d6", BLUE], capsize=3)
        a.set_title(titles[v], loc="left")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES / "F3_outcomes_by_innovation.png")
    plt.close()


def regressions(df):
    b = df[df["binary"] == 1].copy()
    terms = ["radical", "no_innovation"]
    rhs = f"radical + no_innovation + {CONTROLS} + {FE}"
    models = [
        fit(f"voters ~ {rhs}", df, "poisson", "M1 Participation (voters, PPML)", terms),
        fit(f"margin ~ {rhs}", b, "ols", "M2 Contestation (margin, OLS)", terms),
        fit(f"passed ~ {rhs}", b, "logit", "M3 Approval (logit)", terms),
        fit(f"passed ~ {rhs}", b, "lpm", "M3b Approval (linear probability)", terms),
        fit(f"pivotal_top5 ~ {rhs}", b, "logit", "M4 Whale-decided outcome (top-5 pivotal, logit)", terms),
        fit(f"top10_share ~ {rhs}", df, "ols", "M5 Concentration (top-10 share, OLS)", terms),
    ]
    tm = ["radical", "top10_share", "radical:top10_share", "no_innovation"]
    models.append(fit(f"passed ~ radical * top10_share + no_innovation + {CONTROLS} + {FE}", b, "logit",
                      "M6 Approval x concentration (logit)", tm))
    models.append(fit(f"passed ~ radical * top10_share + no_innovation + {CONTROLS} + {FE}", b, "lpm",
                      "M6b Approval x concentration (LPM)", tm))
    t = pd.DataFrame([m for m in models if m])
    save(t, "T7_main_regressions", "Main regressions (DAO and year fixed effects, SE clustered by DAO)")
    return b


def robustness(df):
    b = df[df["binary"] == 1].copy()
    rows = []
    rhs_nov = f"novelty_score + {CONTROLS} + {FE}"
    for dv, d, kind in [("voters", df, "poisson"), ("margin", b, "ols"), ("passed", b, "logit"), ("pivotal_top5", b, "logit")]:
        rows.append(fit(f"{dv} ~ {rhs_nov}", d, kind, f"R1 novelty score: {dv}", ["novelty_score"]))

    rhs = f"radical + no_innovation + {CONTROLS} + {FE}"
    big = df["space"].value_counts().index[0]
    for name, sub in [
        ("R2 excluding largest DAO", df[df["space"] != big]),
        ("R3 from 2023 onwards", df[df["year"] >= 2023]),
        ("R4 high-confidence codes (>=0.7)", df[df["confidence"] >= 0.7]),
        ("R5 basic For/Against/Abstain type only", df[df["vote_type"] == "basic"]),
        ("R6 votes cover >=90% of voting power", df[df["top_votes_coverage"] >= 0.9]),
    ]:
        sb = sub[sub["binary"] == 1]
        rows.append(fit(f"voters ~ {rhs}", sub, "poisson", f"{name}: voters", ["radical"]))
        rows.append(fit(f"passed ~ {rhs}", sb, "logit", f"{name}: passed", ["radical"]))
        rows.append(fit(f"pivotal_top5 ~ {rhs}", sb, "logit", f"{name}: pivotal_top5", ["radical"]))

    bpath = cfg.PROCESSED / "coding_B.csv"
    if bpath.exists():
        B = pd.read_csv(bpath)[["proposal_id", "innovation_type", "category"]]
        d2 = df.drop(columns=["innovation_type", "category", "radical", "no_innovation"]).merge(B, on="proposal_id")
        d2["radical"] = (d2["innovation_type"] == "radical").astype(int)
        d2["no_innovation"] = (d2["innovation_type"] == "none").astype(int)
        sb = d2[d2["binary"] == 1]
        rows.append(fit(f"voters ~ {rhs}", d2, "poisson", "R7 coder B labels: voters", ["radical"]))
        rows.append(fit(f"passed ~ {rhs}", sb, "logit", "R7 coder B labels: passed", ["radical"]))
        rows.append(fit(f"pivotal_top5 ~ {rhs}", sb, "logit", "R7 coder B labels: pivotal_top5", ["radical"]))
    t = pd.DataFrame([r for r in rows if r])
    save(t, "T8_robustness", "Robustness checks")


def main():
    df = load()
    df.to_csv(cfg.PROCESSED / "analysis_dataset.csv", index=False)
    print(f"analysis dataset: {len(df)} proposals, {df['space'].nunique()} DAOs, {int(df['binary'].sum())} binary")
    descriptives(df)
    concentration(df)
    innovation_figure(df)
    regressions(df)
    robustness(df)
    rel = cfg.TABLES / "T_coding_reliability.csv"
    if rel.exists():
        REPORT.insert(0, "\n## T0_coding_reliability: Agreement between coder A and coder B\n\n" +
                      pd.read_csv(rel).to_markdown(index=False, floatfmt=".3f") + "\n")
    head = (f"# Results report\n\nProposals: {len(df)}; DAOs: {df['space'].nunique()}; "
            f"binary approve/reject proposals: {int(df['binary'].sum())}; period: {df['date'].min()} to {df['date'].max()}\n"
            f"Coder A: {cfg.CODER_A}; Coder B: {cfg.CODER_B}\n")
    (cfg.RESULTS / "REPORT.md").write_text(head + "".join(REPORT), encoding="utf-8")
    (cfg.RESULTS / "regression_full_output.txt").write_text("".join(FULL), encoding="utf-8")
    print(f"wrote {cfg.RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
