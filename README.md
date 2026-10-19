# DAO_UP: Voting-power concentration and radical proposals in DAOs

Replication code and data for the research paper **"Decentralized in name? Voting-power concentration, pivotal voters and the fate of radical proposals in DAOs"** (under review).

The project builds a dataset of governance proposals from 26 major decentralized autonomous organizations (DAOs) on [Snapshot](https://snapshot.org), covering January 2021 to June 2026. It then:

1. measures how concentrated voting power is, and how often the largest voters decide outcomes (counterfactual pivotality);
2. codes each proposal's category and innovation type (radical / incremental / none) with off-the-shelf large language models (no fine-tuning), validated by a second, independent model;
3. estimates how innovation type and concentration relate to participation, contestation, approval and large-voter influence.

> Results, tables and figures are not included while the paper is under review. They are produced locally by running the pipeline below and will be added after publication.

## Repository structure

```
DAO_UP/
├── Code/
│   ├── config.py               paths, DAO list, sampling rules, model names
│   ├── 01_collect_snapshot.py  downloads proposals and the 1,000 largest votes per proposal
│   ├── 02_build_measures.py    participation, concentration, margin, approval, pivotality
│   ├── 03_llm_coding.py        LLM-assisted content coding (coder A: all, coder B: validation subsample)
│   ├── 04_analysis.py          descriptive statistics, figures, regressions, robustness checks
│   ├── 05_package_results.py   zips Results/ for sharing
│   ├── run_all.sh              runs steps 01–05 in order
│   └── requirements.txt
├── Data/
│   ├── raw/
│   │   ├── spaces.json         Snapshot spaces found and included
│   │   └── sample_ids.json     IDs of the 4,084 sampled proposals (fixes the sample)
│   ├── processed/
│   │   ├── proposals_measures.csv   proposal-level measures
│   │   ├── proposals_text.jsonl     proposal title, options and text used for coding
│   │   ├── coding_A.csv / coding_B.csv  normalized LLM codes
│   │   └── analysis_dataset.csv     merged dataset used in the regressions
│   └── llm_coding/
│       ├── coder_A.jsonl       raw coder A outputs (gpt-4.1-mini-2025-04-14)
│       └── coder_B.jsonl       raw coder B outputs (gpt-4o-2024-08-06)
└
```

The large raw downloads (`Data/raw/proposals.jsonl` and `Data/raw/votes_top.jsonl`) are not stored in the repository because they exceed GitHub's file-size limit. Step 01 recreates them from the public Snapshot API. Because `sample_ids.json` is included, step 01 rebuilds exactly the same proposal sample.

## Requirements

- Python 3.10 or newer
- An OpenAI API key, needed only to re-run step 03. The coded outputs are included, so steps 02 and 04 run without a key.
- Internet access to `hub.snapshot.org` for step 01

## Setup

Windows (Git Bash):

```bash
cd /c/path/to/DAO_UP
python -m venv dao_env
source dao_env/Scripts/activate
pip install -r Code/requirements.txt
export PYTHONUTF8=1
```

macOS / Linux:

```bash
cd path/to/DAO_UP
python3 -m venv dao_env
source dao_env/bin/activate
pip install -r Code/requirements.txt
```

Set your OpenAI key as an environment variable. Never write it into the code.

```bash
export OPENAI_API_KEY="sk-..."
```

An optional Snapshot API key (`SNAPSHOT_API_KEY`) speeds up collection.

## Running the pipeline

```bash
cd Code
python 01_collect_snapshot.py   # 2–4 hours (Snapshot rate limits); resumes if interrupted
python 02_build_measures.py     # seconds
python 03_llm_coding.py         # uses cached codes in Data/llm_coding; only uncoded proposals are sent to the API
python 04_analysis.py           # about 1 minute; writes Results/tables, Results/figures, Results/REPORT.md
```

To reproduce the analysis from the included processed data only, without downloading or calling any API, run:

```bash
cd Code
python 04_analysis.py
```

## Key settings (`Code/config.py`)

| Setting                      | Value                                                      |
| ---------------------------- | ---------------------------------------------------------- |
| Period                       | 2021-01-01 to 2026-06-30 (proposal creation date)          |
| Vote types                   | single-choice and basic (For/Against/Abstain)              |
| Minimum voters per proposal  | 10                                                         |
| Maximum proposals per DAO    | 400 (random sample, seed 42)                               |
| Votes retrieved per proposal | the 1,000 largest by voting power                          |
| Coder A / Coder B            | gpt-4.1-mini-2025-04-14 / gpt-4o-2024-08-06, temperature 0 |
| Coder B validation sample    | 600 proposals                                              |

## Data source and license

Proposal and vote data come from the public Snapshot GraphQL API and are pseudonymous blockchain records. Code is released under the MIT License (see `LICENSE`). Derived data in `Data/` are released under CC BY 4.0.

## Citation

The paper is under review. Citation details will be added after publication.

## Contact

Pejman Ebrahimi, University of Liechtenstein: pejman.ebrahimi@uni.li
