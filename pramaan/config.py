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
# "hybrid" dense+BM25 (best), "dense", or "lexical" -- BM25 only, which needs
# no torch and no encoder at all. Lexical mode exists so the app can be hosted
# on a free tier too small to hold torch plus two transformer models; it is
# measurably worse, and the sidebar says so.
RETRIEVAL_MODE = os.getenv("PRAMAAN_RETRIEVAL_MODE", "hybrid")
HYBRID_RETRIEVAL = RETRIEVAL_MODE == "hybrid"
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
# Which LLM writes the draft. "auto" picks whichever provider key is present:
# groq / gemini (both have free tiers, no card) / openai-compatible /
# anthropic, falling back to "extractive" if there is no key at all.
# The gate does not care which model this is -- that is rather the point.
DRAFT_PROVIDER = os.getenv("PRAMAAN_DRAFT_PROVIDER", "auto")
# Empty means "use that provider's default", see drafter.DEFAULT_MODELS.
DRAFT_MODEL = os.getenv("PRAMAAN_DRAFT_MODEL", "")
DRAFT_MAX_TOKENS = 700

# For any OpenAI-compatible endpoint: OpenRouter, a local Ollama, LM Studio.
OPENAI_BASE_URL = os.getenv("PRAMAAN_OPENAI_BASE_URL", "https://api.openai.com/v1")

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
# Measured, see scripts/bench_verifier.py. distilroberta is 25x faster and
# looked fine on clean hand-written premises (true >=0.766, false <=0.026), but
# on the real retrieved clauses -- longer, OCR-scarred, full of clause numbering
# -- it dropped below threshold on questions DeBERTa answers correctly, so the
# app abstained on valid questions. A gate that refuses good questions is worse
# than a slow one. Speed comes from NLI_CANDIDATES instead.
NLI_MODEL = os.getenv("PRAMAAN_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# If fewer than this fraction of drafted sentences survive, we treat the whole
# answer as unsupported and abstain rather than handing back a shredded reply.
MIN_SURVIVING_FRACTION = float(os.getenv("PRAMAAN_MIN_SURVIVING_FRACTION", "0.34"))

# Clauses the gate retrieves for itself, per sentence, on top of whatever was
# given to the drafter. Set to 0 to make the gate trust the drafting retrieval.
VERIFY_TIME_K = int(os.getenv("PRAMAAN_VERIFY_TIME_K", "5"))

# The cross-encoder costs ~2-3s per premise on CPU, so running it over every
# retrieved clause made a single answer take 30-45s. Cheap embedding similarity
# picks the few premises actually worth checking; the correct clause is nearly
# always among the top handful, and anything that is not is not going to entail
# the sentence anyway.
NLI_CANDIDATES = int(os.getenv("PRAMAAN_NLI_CANDIDATES", "4"))
# Clauses are <=700 chars (~175 tokens), so 512 just pays for padding.
NLI_MAX_LENGTH = int(os.getenv("PRAMAAN_NLI_MAX_LENGTH", "256"))

# Score each clause sentence-by-sentence instead of as one 550-char block.
# This is the granularity finding from the SummaC paper we cite: a claim is
# entailed by one sentence, and burying that sentence in a paragraph of clause
# numbering and cross-references dilutes the signal. It is also markedly
# cheaper, because attention cost grows with the square of sequence length.
SENTENCE_LEVEL_PREMISES = os.getenv("PRAMAAN_SENTENCE_PREMISES", "0") not in ("0", "false", "False")

# ---------------------------------------------------------------- speech
ASR_MODEL = os.getenv("PRAMAAN_ASR_MODEL", "base")
ENABLE_ASR = os.getenv("PRAMAAN_ENABLE_ASR", "1") not in ("0", "false", "False")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
