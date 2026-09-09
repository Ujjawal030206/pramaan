"""Central configuration. Everything tunable lives here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORPUS_DIR = ROOT / "corpus"
RAW_DIR = CORPUS_DIR / "raw"
SOURCES_FILE = CORPUS_DIR / "sources.yaml"
INDEX_DIR = ROOT / "index"

# ---------------------------------------------------------------- retrieval
# Multilingual by default: several source documents contain Hindi passages and
# users are told they may ask in Hindi, so an English-only encoder would quietly
# fail on exactly the users this is meant to serve. Set PRAMAAN_EMBED_MODEL to
# sentence-transformers/all-MiniLM-L6-v2 for a smaller English-only footprint.
EMBED_MODEL = os.getenv(
    "PRAMAAN_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# 10 rather than 6: on the probe set in scripts/bench_retrieval.py the last two
# governing clauses land at ranks 9 and 11, and passing extra clauses to the
# gate is close to free in accuracy terms -- an irrelevant clause simply fails
# to entail anything. The cost is latency, since the verifier runs one forward
# pass per (clause, sentence) pair.
TOP_K = int(os.getenv("PRAMAAN_TOP_K", "10"))

# Hybrid retrieval. Eligibility turns on literal tokens -- "income tax", "6000",
# "18 to 40" -- and a dense-only retriever benchmarked here ranked the PM-KISAN
# income-tax exclusion 35th for a question that quoted the phrase. BM25 fixes
# exactly that class of miss, so we fuse both rankings with RRF.
HYBRID_RETRIEVAL = os.getenv("PRAMAAN_HYBRID", "1") not in ("0", "false", "False")
BM25_WEIGHT = float(os.getenv("PRAMAAN_BM25_WEIGHT", "1.0"))
RRF_DEPTH_MULTIPLIER = int(os.getenv("PRAMAAN_RRF_DEPTH", "10"))

# Clause chunking. Government documents are clause-structured, so we chunk on
# paragraph boundaries and keep chunks small enough that a citation points at
# something a human can actually read on screen.
# Below MIN we keep accumulating rather than starting a new chunk at a clause
# number -- that case is normally a bare section heading, which belongs with the
# clause it introduces.
CHUNK_MIN_CHARS = 120
CHUNK_MAX_CHARS = 700

# ---------------------------------------------------------------- drafting
DRAFT_MODEL = os.getenv("PRAMAAN_DRAFT_MODEL", "claude-sonnet-5")
DRAFT_MAX_TOKENS = 700

# ---------------------------------------------------------------- verification
# The gate. A sentence survives only if some retrieved clause entails it with
# at least this probability. Raise it to be stricter (more abstention, fewer
# unsupported claims); lower it to be chattier. 0.5 was chosen on the eval set
# in eval/questions.yaml -- see README for how to re-tune.
ENTAILMENT_THRESHOLD = float(os.getenv("PRAMAAN_ENTAILMENT_THRESHOLD", "0.5"))

# "nli"  -> local cross-encoder, no API calls, what we demo and measure
# "llm"  -> entailment judged by the API, for hosts too small for torch
# "auto" -> nli if torch is importable, else llm
VERIFIER_BACKEND = os.getenv("PRAMAAN_VERIFIER_BACKEND", "auto")
NLI_MODEL = os.getenv("PRAMAAN_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# If fewer than this fraction of drafted sentences survive, we treat the whole
# answer as unsupported and abstain rather than handing back a shredded reply.
MIN_SURVIVING_FRACTION = float(os.getenv("PRAMAAN_MIN_SURVIVING_FRACTION", "0.34"))

# Clauses the gate retrieves for itself, per sentence, on top of whatever was
# given to the drafter. Set to 0 to make the gate trust the drafting retrieval.
VERIFY_TIME_K = int(os.getenv("PRAMAAN_VERIFY_TIME_K", "5"))

# ---------------------------------------------------------------- speech
ASR_MODEL = os.getenv("PRAMAAN_ASR_MODEL", "base")
ENABLE_ASR = os.getenv("PRAMAAN_ENABLE_ASR", "1") not in ("0", "false", "False")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
