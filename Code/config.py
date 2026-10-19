import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CODING = DATA / "llm_coding"
RESULTS = ROOT / "Results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

for p in [RAW, PROCESSED, CODING, TABLES, FIGURES]:
    p.mkdir(parents=True, exist_ok=True)

SNAPSHOT_URL = "https://hub.snapshot.org/graphql"
SNAPSHOT_API_KEY = os.getenv("SNAPSHOT_API_KEY", "")
REQUEST_PAUSE = 1.1

CANDIDATE_SPACES = [
    "uniswapgovernance.eth",
    "aavedao.eth",
    "aave.eth",
    "arbitrumfoundation.eth",
    "opcollective.eth",
    "ens.eth",
    "lido-snapshot.eth",
    "gitcoindao.eth",
    "balancer.eth",
    "safe.eth",
    "cow.eth",
    "1inch.eth",
    "sushigov.eth",
    "curve.eth",
    "apecoin.eth",
    "gmx.io",
    "stgdao.eth",
    "frax.eth",
    "olympusdao.eth",
    "snapshot.dcl.eth",
    "shapeshiftdao.eth",
    "hop.eth",
    "pooltogether.eth",
    "beethovenxfi.eth",
    "metislayer2.eth",
    "radiantcapital.eth",
    "venus-xvs.eth",
    "pancake",
    "kleros.eth",
    "dydxgov.eth",
]

START_DATE = "2021-01-01"
END_DATE = "2026-06-30"
MIN_PROPOSALS_PER_SPACE = 30
MIN_VOTERS = 10
MAX_PROPOSALS_PER_SPACE = 400
TOP_VOTES_PER_PROPOSAL = 1000
VOTE_TYPES = ["single-choice", "basic"]

CODER_A = os.getenv("CODER_A", "gpt-4.1-mini-2025-04-14")
CODER_B = os.getenv("CODER_B", "gpt-4o-2024-08-06")
CODER_B_SAMPLE = int(os.getenv("CODER_B_SAMPLE", "600"))
MAX_BODY_CHARS = 6000
SEED = 42
