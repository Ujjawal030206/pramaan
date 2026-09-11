---
title: PRAMAAN
emoji: 🪔
colorFrom: purple
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
short_description: The AI advisor that refuses to guess.
---

# PRAMAAN

**The AI advisor that refuses to guess.**

Proof-grounded eligibility guidance for Indian government welfare schemes.

Built for **CodeArambh 2.0** · Domain: AI/ML
Team Pramaan — Ujjawal Srivastava (lead), Amey Dongre, Tejas Gupta

---

## The problem we are actually solving

India runs thousands of central and state welfare schemes. The rules that decide
who qualifies sit inside long PDFs, written in English, spread across dozens of
portals. Most people who are entitled to a benefit never find out.

The obvious fix — put a chatbot in front of it — makes things worse. A language
model will answer an eligibility question fluently and confidently with no idea
which clause governs the answer, and no way to signal that it is guessing. In
this setting a hallucination is not a bug, it is harm: a wrong *yes* costs
someone a day's wage, a long journey and the wrong documents at the counter; a
wrong *no* silently denies a benefit they were owed.

So the hard part is not answering the question. It is **proving the answer**.

## What PRAMAAN does differently

Most retrieval-augmented systems put their trust in the prompt: *"only use the
provided context, do not make things up."* That is a request. Models comply
most of the time, which is exactly what makes the remaining cases dangerous.

PRAMAAN does not ask. After the model drafts an answer, a **separate classifier
checks every sentence against the retrieved clauses**. A sentence survives only
if some clause actually entails it. Everything else is deleted before the user
sees it. If too little survives, the system abstains and routes the user to a
human.

```
question
   ↓
retrieve      dense search over verbatim clauses from official PDFs
   ↓
draft         LLM writes an answer from those clauses only
   ↓
VERIFY        every sentence entailment-checked against its evidence   ← the gate
   ↓
deliver       supported sentences + the exact clause, page and date
   or
abstain       "I could not verify this" + where to go instead
```

The asymmetry is the point. A prompt can be talked out of. A threshold cannot.

## What is actually novel here

Not the RAG. The RAG is table stakes. Three things:

1. **The gate is post-generation and mechanical.** The component that decides
   what reaches the user is not the component that wrote it, and it has one
   job: entailment, yes or no. See [`pramaan/verifier.py`](pramaan/verifier.py).
2. **Abstention is a first-class output**, not an error path. `Outcome.abstained`
   is a normal result with its own UI, its own reason string, and its own
   next-step routing.
3. **The failure mode is silence, not error.** When retrieval misses, nothing
   is entailed, so nothing survives, so we abstain. Degrading gracefully falls
   out of the architecture rather than being bolted on.

## We publish our own failure rate

`eval/run_eval.py` reports, on every run:

| Metric | What it means |
|---|---|
| **Verification catch-rate** | share of drafted sentences the gate blocked |
| **Abstention rate** | share of questions we declined to answer |
| **Citation coverage** | share of surviving sentences carrying a clause |

Catch-rate and abstention rate are meaningless alone — a system that abstains on
everything scores perfectly and is useless. So the eval set is split into
questions the corpus *does* settle and questions it *does not*, and we also
report:

| Metric | What it means |
|---|---|
| **Answer rate** | on answerable questions — are we actually useful? |
| **Abstention precision** | on unanswerable questions — do we decline the right ones? |

Citation coverage is 1.0 by construction; it is reported as an invariant check,
so if it ever drops we know the gate has a bug.

## The corpus

Seven official documents, five schemes, downloaded directly from government
domains — nothing written or paraphrased by us:

| Scheme | Documents |
|---|---|
| PM-KISAN | Operational Guidelines, Revised FAQ, Clause 3.3/3.4 Clarifications |
| PM-KMY (Kisan Maan Dhan) | Operational Guidelines |
| Atal Pension Yojana | Details of the Scheme |
| PMJJBY | Rules |
| PMSBY | Rules |

Sources and retrieval dates are in [`corpus/sources.yaml`](corpus/sources.yaml).
Every citation shown to a user names the document, the page, the issuing
authority and the date we fetched it — because a circular that was current in
September may not be current in March.

**To add a scheme:** add an entry to `sources.yaml`, then re-run
`fetch_corpus.py` and `build_index.py`. No retraining. That is the scaling story.

## Getting an LLM key (free)

The drafter is a commodity and the gate does not care which model wrote the
sentences it checks, so PRAMAAN takes whichever provider you can get:

| Provider | Where | Cost |
|---|---|---|
| **Groq** (recommended) | console.groq.com/keys | free, no card, very fast |
| **Google Gemini** | aistudio.google.com/apikey | free, no card |
| OpenAI-compatible | OpenRouter free models, local Ollama, LM Studio | free |
| Anthropic | console.anthropic.com | paid |

Put one in a file called `.env` in this folder (same level as `app.py`):

```
GROQ_API_KEY=gsk_your_key_here
```

`pramaan/__init__.py` loads that file into the environment before anything reads
config, so nothing else needs configuring. On Streamlit Cloud there is no `.env`
-- put the same line in **Settings -> Secrets** instead, as TOML, and `app.py`
copies it across. The sidebar shows which provider it detected, so you can tell
at a glance whether the key was picked up. Retrieval, citations and the
verification gate are **all local and cost nothing** -- only drafting needs a
provider.

With no key at all the drafter falls back to `extractive`, answering with
sentences copied verbatim from retrieved clauses. The app still runs and still
cites, but be honest in a demo: an extractive draft is made of clause text, so
the gate passes it by construction and is not being exercised.

## Running it

Requires Python 3.11–3.13 (3.14 has no torch wheels yet).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

cp .env.example .env            # add a free GROQ_API_KEY (see below)

python scripts/fetch_corpus.py  # downloads the 7 official documents
python scripts/build_index.py   # extracts clauses, builds the index
streamlit run app.py
```

First run downloads the encoder (~470 MB) and the NLI model (~700 MB).

### Evaluate

```bash
python eval/run_eval.py
```

### Test

```bash
python tests/test_verifier.py    # gate logic, no torch or API key needed
python tests/test_gate_nli.py    # real NLI gate against real clauses, no API key
```

## Configuration

Everything is in [`pramaan/config.py`](pramaan/config.py) and overridable by
environment variable.

| Variable | Default | Effect |
|---|---|---|
| `PRAMAAN_ENTAILMENT_THRESHOLD` | `0.5` | Higher = stricter gate, more abstention |
| `PRAMAAN_VERIFIER_BACKEND` | `auto` | `nli` (local, measured) / `llm` (for small hosts) |
| `PRAMAAN_MIN_SURVIVING_FRACTION` | `0.34` | Below this share surviving, abstain entirely |
| `PRAMAAN_TOP_K` | `10` | Clauses retrieved per question |
| `PRAMAAN_DRAFT_PROVIDER` | `auto` | `groq` / `gemini` / `openai` / `anthropic` / `extractive` |
| `PRAMAAN_EMBED_MODEL` | multilingual MiniLM | Set to `all-MiniLM-L6-v2` for a smaller English-only build |

### Tuning the threshold

The threshold trades abstention against coverage. Sweep it against the eval set:

```bash
for t in 0.3 0.4 0.5 0.6 0.7; do
  PRAMAAN_ENTAILMENT_THRESHOLD=$t python eval/run_eval.py --out eval/results_$t.json
done
```

Pick the value that maximises abstention precision without collapsing the
answer rate. Do not tune it on the demo questions you plan to show judges.

## Deploying

The app holds two transformer models resident and needs **~1.7 GB of RAM** once
it has answered a question (measured, not estimated). That number rules most
free hosts out:

| Host | Free RAM | Runs Streamlit | Verdict |
|---|---|---|---|
| Vercel | n/a | no -- serverless, no persistent websocket | won't run at all |
| Render | 512 MB | yes | 3x short of RAM |
| Streamlit Community Cloud | ~1 GB | yes, natively | likely OOMs on boot |
| **Hugging Face Spaces** | **16 GB** | yes, natively | **use this** |

Vercel is doubly out: its Hobby serverless bundle caps at 250 MB against our
1.3 GB of dependencies, and its 10s function timeout is shorter than our 12.7s
median answer.

### Hugging Face Spaces (recommended)

1. huggingface.co -> **New Space** -> SDK **Streamlit**, hardware **CPU basic (free)**.
2. Push this repo to the Space remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/pramaan
   git push space main
   ```
3. Space **Settings -> Variables and secrets -> New secret**: `GROQ_API_KEY`.
4. First build takes ~10 minutes (torch is a large download) and the first
   question downloads ~850 MB of models. Both are cached afterwards.

The YAML block at the top of this README is the Space configuration; leave it
in place.

Free Spaces sleep after ~48h idle and take ~30s to wake, so open the link once
before a demo.

### Streamlit Community Cloud (fallback)

Works the same way -- point it at `app.py`, put `GROQ_API_KEY` in
**Settings -> Secrets** as TOML. If it is killed on boot, swap
`requirements-cloud.txt` in for `requirements.txt` and set:
```toml
PRAMAAN_RETRIEVAL_MODE = "lexical"
PRAMAAN_VERIFIER_BACKEND = "llm"
```
That fits in 1 GB but runs the degraded stack -- BM25-only retrieval and an
LLM-judged gate, which the eval scores at 53.8% against 92.3%. Prefer Spaces.

Do **not** just point `PRAMAAN_EMBED_MODEL` at a smaller encoder: the committed
vectors were built with one specific model and mixing them silently corrupts
retrieval. The retriever refuses to start if you try.

## Layout

```
app.py                  Streamlit UI; shows the trace, not just the answer
pramaan/
  config.py             all tunables
  ingest.py             PDF -> verbatim clauses + citation metadata
  retriever.py          dense retrieval (FAISS, numpy fallback)
  drafter.py            the LLM step, constrained to retrieved clauses
  verifier.py           THE GATE - entailment check, strip, abstain
  pipeline.py           orchestration; Outcome is the public result type
  asr.py                optional speech input
scripts/
  fetch_corpus.py       download official documents
  build_index.py        extract clauses, embed, index
eval/
  questions.yaml        answerable + must-abstain questions
  run_eval.py           the metrics above
tests/
```

## What we measured, and what it cost us

Three findings from building this, all of which changed the design:

**1. Chunking decided everything.** Our first version chunked page by page and
merged paragraphs freely. That glued clause 10.1 ("The financial benefit of
Rs.6000/- per year ... in three installments") onto the end of a paragraph about
nodal officers submitting lists. The resulting embedding was about nodal
officers, so the benefit clause ranked **29th of 246** for "How much money do I
get under PM-KISAN?" — effectively unfindable. Chunking on clause boundaries
instead fixed it.

**2. List items are meaningless without their stem.** "All Persons who paid
Income Tax in last assessment year" is a noun phrase, not a rule. The sentence
that makes it a rule — "4.1 The following categories ... shall not be eligible"
— sits above the list and, in this document, on the previous page. Chunking per
page stranded the item; attaching the stem took its entailment score from
**0.026 to 0.872**.

**3. Dense retrieval alone misses literal terms.** Eligibility turns on exact
tokens. A dense-only retriever ranked the income-tax exclusion **35th** for a
question that quoted the phrase "income tax". Adding BM25 and fusing with RRF:

| retriever | recall@6 | MRR |
|---|---|---|
| dense only | 70% | 0.485 |
| dense + BM25 (RRF) | **80%** | **0.663** |

Reproduce with `python scripts/bench_retrieval.py`. The two remaining probe
misses land at ranks 9 and 10, which is why `TOP_K` is 10 rather than 6.

The gate itself, measured on real clause text by `tests/test_gate_nli.py`:
supported claims score **0.95–0.99**, fabricated ones **0.00–0.01**.

**4. A fast gate is a trap.** Judging entailment with the LLM instead of the
local classifier is 20x faster and looks fine question by question. Across the
whole eval set it collapses:

| gate | overall accuracy | answers the 7 it should | refuses the 6 it should | median |
|---|---|---|---|---|
| local NLI (default) | **92.3%** | 6 / 7 | 6 / 6 | 11.3s |
| LLM-judged | 53.8% | **1 / 7** | 6 / 6 | 0.6s |

The LLM judge abstains on 92% of everything -- perfect on abstention precision,
useless in practice. That degenerate strategy is exactly what the split
answerable/unanswerable eval set exists to expose, and why catch-rate is never
reported alone.

Measured on the 13-question set with the local NLI gate:

| Metric | Value |
|---|---|
| Verification catch-rate | 40.0% |
| Abstention rate | 53.8% |
| Citation coverage | 100% |
| Answer rate (answerable) | 85.7% -- 6 of 7 |
| Abstention precision (unanswerable) | 100% -- 6 of 6 |
| Overall accuracy | 92.3% |
| Median latency | 11.3s |

**5. The gate is sensitive to phrasing, not only meaning.** Our sharpest
limitation, and the cause of the single eval miss. Against the PM-KISAN
definition clause, four true and equivalent sentences score:

| Sentence (all true, all equivalent) | Entailment |
|---|---|
| "The family covered by PM-KISAN is husband, wife and minor children." | 0.793 |
| "...is a family comprising husband, wife and minor children..." | 0.548 |
| "A farmer's family means husband, wife and minor children." | 0.218 |
| "Husband, wife and minor children are covered by the definition of family." | 0.020 |

A true sentence can be deleted for how it happens to be worded.

Two fixes considered and rejected on evidence. Scoring both directions does not
help -- reverse entailment is ~0.005 on all of the above. And lowering the
threshold is unsafe: an *inverted* claim ("persons who paid income tax **are**
eligible", against the clause saying they are not) scores 0.370, so dropping the
bar to rescue a 0.46 true sentence would start admitting false ones. 0.5 stays.

We prefer this failure direction. A deleted true sentence costs coverage; an
admitted false one costs someone a wasted day and a journey to a government
office.

**6. Refutation must outrank support -- but only relevant refutation.** The
side-by-side comparison exposed a false positive: asked about income tax, a plain
chatbot wrote "paying income tax does not automatically disqualify you from
PM-KISAN". The gate scored it 0.952 *supported* -- entailed by an unrelated
PM-KMY enrolment clause -- while the actual income-tax exclusion contradicted it
at 0.902. The gate was reading entailment and ignoring contradiction.

A first veto (any clause contradicting at 0.85+ deletes the sentence) caught it,
but also deleted true sentences: off-topic clauses from other schemes "contradicted"
correct PM-KISAN facts, and one eval question failed outright. The fix that held:
**only a clause at least as relevant as the supporting clause may veto it.** The
false claim is refuted by the most relevant clause there is, so it is still caught;
cross-scheme noise can no longer veto. On the 13-question eval the bounded veto
fires on zero correct answers.

**7. The eval is noisier than one number suggests.** Groq's drafts vary from run to
run even at temperature 0, so the same gate has scored 92.3%, 100%, 92.3% and 76.9%
across four runs of the 13-question set (mean about 90%). Refusal is the stable
part -- **6 of 6 unanswerable questions refused on every run**. The variance is on
the answering side, where a differently-worded draft can fall just under the
threshold. Treat any single figure as one sample.

## Honest limitations

- **Retrieval is the ceiling.** If the right clause is not retrieved, we abstain.
  That is the safe failure, but it is still a failure, and our abstention rate
  is dominated by retrieval misses rather than genuine unanswerability.
- **Entailment is not correctness.** A clause can entail a sentence that is
  nonetheless a bad answer to the question asked. The gate catches fabrication,
  not irrelevance.
- **The gate is not infallible.** Probing it, we found a PM-KMY clause about
  social-security overlap scoring 0.557 for a claim about institutional land
  holders — a false positive above threshold. It lost to the correct clause at
  0.958 and so did no harm, but it is there. The threshold is a dial, not a
  proof.
- **Source PDF quality varies.** The PM-KISAN FAQ is scanned and OCR-damaged
  ("lncome Tax", "su pe ran n uated"). We do not repair source text, because
  editing a document we are citing would defeat the point; those passages
  simply retrieve less well.
- **English-dominant corpus.** Query-side Hindi works via the multilingual
  encoder, but most source text is English, so Hindi answers are weaker.
- **Five schemes.** This is a prototype corpus, not coverage of the thousands of
  schemes that exist. The ingestion path is the product; the corpus is a demo.
- **Not an official service.** Nothing here should be relied on for a real
  benefit decision without confirming against the source document we cite.

## Licence

MIT for our code. The corpus documents remain the property of their issuing
ministries and are redistributed here only by reference — `fetch_corpus.py`
downloads them at build time rather than vendoring them into the repo.
