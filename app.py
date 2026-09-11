"""PRAMAAN -- web interface.

Streamlit underneath, so it still deploys free, but restyled end to end: the
framework's own chrome is hidden and every result is rendered as our own HTML.
The layout exists to put the gate on screen -- the sentences it proved, the ones
it deleted, and the clause behind each -- instead of tucking it into a collapsed
panel nobody opens.
"""
from __future__ import annotations

import html
import os
import re
import time
from collections import defaultdict
from dataclasses import replace

import streamlit as st

# Streamlit Community Cloud supplies secrets through st.secrets, not the
# environment, so copy them across BEFORE pramaan is imported -- config.py
# reads os.getenv at import time. Existing env vars win.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from pramaan.config import ENTAILMENT_THRESHOLD  # noqa: E402
from pramaan.drafter import detect_provider, model_for  # noqa: E402
from pramaan.pipeline import Pramaan  # noqa: E402

st.set_page_config(
    page_title="PRAMAAN — the AI advisor that refuses to guess",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REPO_URL = "https://github.com/Ujjawal030206/pramaan"

# From eval/run_eval.py on the 13-question set. Update both together whenever
# the corpus, model or threshold changes: the site must never advertise a
# number the repository cannot reproduce.
MEASURED_ACCURACY = "92.3%"
MEASURED_DETAIL = "answers 6 of 7 it should · refuses 6 of 6 it should"

# An answer that failed because the model could not be reached. Never cached,
# and shown with a note that it says nothing about the question itself.
PROVIDER_FAILURE = "drafting step unavailable"

EXAMPLES = [
    ("Farmer with 1 hectare — PM-KISAN?",
     "I am a farmer with 1 hectare of land. Can I get PM-KISAN?"),
    ("Paid income tax — still eligible?",
     "I paid income tax last year. Am I still eligible for PM-KISAN?"),
    ("Age 42 — Atal Pension Yojana?",
     "I am 42 years old. Can I join Atal Pension Yojana?"),
    ("PMJJBY — cost and payout",
     "How much does PMJJBY cost per year and what does it pay out?"),
    ("PM-KMY — entry age",
     "What is the entry age for PM Kisan Maan Dhan Yojana?"),
    # Deliberately unanswerable, and labelled so: the refusal is the demo.
    ("Trap: tomorrow's weather",
     "Will PRAMAAN tell me tomorrow's weather?"),
]

STEPS = [
    ("retrieve", "Retrieve", "Searching the clause index"),
    ("draft", "Draft", "Writing from the clauses only"),
    ("verify", "Verify", "Checking every sentence"),
    ("done", "Deliver", "Answer — or refuse"),
]

E = html.escape

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0B0D12;--surface:#141821;--surface2:#10141C;--line:#232938;--line2:#2E3547;
  --text:#E6E8EE;--muted:#8A91A3;--faint:#5C6477;
  --violet:#7C5CFF;--teal:#3DD9C5;--ok:#34D399;--bad:#F87171;--warn:#FBBF24;
  --grad:linear-gradient(135deg,#7C5CFF 0%,#3DD9C5 100%);
  --font:'Inter',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Consolas,monospace;
}
.stApp{
  font-family:var(--font);color:var(--text);
  background:
    radial-gradient(900px 520px at 10% -8%,rgba(124,92,255,.20),transparent 60%),
    radial-gradient(760px 460px at 96% -4%,rgba(61,217,197,.13),transparent 60%),
    var(--bg);
}
.stApp p,.stApp label,.stApp button,.stApp textarea{font-family:var(--font)!important}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],[data-testid="stHeaderActionElements"],
[data-testid="InputInstructions"],#MainMenu,footer{display:none!important}
[data-testid="stMainBlockContainer"],.block-container{max-width:1120px!important;padding:1.3rem 2rem 4rem!important}

/* nav */
.pm-nav{display:flex;align-items:center;justify-content:space-between;padding:.2rem 0 1.4rem}
.pm-brand{display:flex;align-items:center;gap:.65rem;font-weight:800;letter-spacing:.16em;font-size:.98rem}
.pm-logo{width:30px;height:30px;border-radius:9px;background:var(--grad);display:grid;place-items:center;
  color:#0B0D12;font-size:.95rem;box-shadow:0 0 26px rgba(124,92,255,.5)}
.pm-navr{display:flex;gap:.6rem;align-items:center}
.pm-chip{display:inline-flex;align-items:center;gap:.45rem;padding:.38rem .8rem;border:1px solid var(--line2);
  border-radius:999px;font-size:.78rem;color:var(--muted);background:rgba(20,24,33,.7)}
.pm-dot{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 10px var(--ok)}
.pm-link,.pm-flink{color:var(--text)!important;text-decoration:none!important;font-size:.8rem;
  border:1px solid var(--line2);padding:.38rem .85rem;border-radius:999px;transition:border-color .15s}
.pm-link:hover,.pm-flink:hover{border-color:var(--violet)}

/* hero */
.pm-hero{padding:1.4rem 0 1.6rem}
.pm-eyebrow{font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;color:var(--teal);font-weight:600;margin-bottom:1rem}
.pm-hero h1{font-size:clamp(2.3rem,5.4vw,4rem)!important;line-height:1.03!important;font-weight:800!important;
  letter-spacing:-.035em;margin:0 0 1.05rem!important;padding:0!important;color:var(--text)!important}
.pm-grad{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.pm-lede{font-size:1.08rem;line-height:1.65;color:var(--muted);max-width:720px;margin:0 0 1.6rem}
.pm-stats{display:flex;flex-wrap:wrap;gap:.6rem}
.pm-stat{background:rgba(20,24,33,.8);border:1px solid var(--line);border-radius:12px;padding:.62rem 1rem;
  display:flex;align-items:baseline;gap:.5rem}
.pm-stat b{font-size:1.2rem;font-weight:750;color:var(--text)}
.pm-stat span{font-size:.8rem;color:var(--muted)}

/* ask card */
.st-key-askcard{background:linear-gradient(180deg,rgba(26,31,43,.92),rgba(20,24,33,.92));
  border:1px solid var(--line2);border-radius:18px;padding:1.25rem 1.35rem 1.15rem;
  box-shadow:0 24px 70px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.03)}
.pm-label{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:.1rem}
.pm-gap{margin-top:.8rem}
[data-testid="stTextArea"] [data-baseweb]{border:0!important;background:transparent!important}
[data-testid="stTextArea"] textarea{background:#0E1219!important;border:1px solid var(--line2)!important;
  border-radius:12px!important;color:var(--text)!important;font-size:1.03rem!important;line-height:1.5!important;
  padding:.9rem 1rem!important;transition:border-color .15s,box-shadow .15s}
[data-testid="stTextArea"] textarea:focus{border-color:var(--violet)!important;box-shadow:0 0 0 3px rgba(124,92,255,.22)!important}
[data-testid="stTextArea"] textarea::placeholder{color:var(--faint)!important}
[data-testid="stBaseButton-primary"]{background:var(--grad)!important;border:0!important;border-radius:12px!important;
  padding:.66rem 1.2rem!important;box-shadow:0 10px 30px rgba(124,92,255,.35)!important;
  transition:transform .15s ease,box-shadow .15s ease}
[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 14px 36px rgba(61,217,197,.32)!important}
[data-testid="stBaseButton-primary"] p{color:#0B0D12!important;font-weight:700!important;font-size:.95rem!important}
[class*="st-key-ex"] button{background:rgba(14,18,25,.85)!important;border:1px solid var(--line2)!important;
  border-radius:999px!important;min-height:0!important;padding:.42rem .9rem!important;transition:all .15s}
[class*="st-key-ex"] button p{color:var(--muted)!important;font-size:.83rem!important}
[class*="st-key-ex"] button:hover{border-color:var(--violet)!important;background:rgba(124,92,255,.09)!important}
[class*="st-key-ex"] button:hover p{color:var(--text)!important}
.st-key-ex5 button{border-color:rgba(251,191,36,.38)!important}
.st-key-ex5 button p{color:#FCD34D!important}

/* live pipeline */
.pm-steps{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:1.2rem 0 .3rem}
.pm-step{position:relative;overflow:hidden;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:.8rem .95rem}
.pm-step .n{font-family:var(--mono);font-size:.68rem;color:var(--faint);letter-spacing:.06em}
.pm-step .t{font-weight:650;font-size:.95rem;margin-top:.2rem}
.pm-step .d{font-size:.78rem;color:var(--muted);margin-top:.2rem;min-height:1.15em}
.pm-step.pending{opacity:.55}
.pm-step.active{border-color:rgba(124,92,255,.65);box-shadow:0 0 0 1px rgba(124,92,255,.25),0 0 30px rgba(124,92,255,.2)}
.pm-step.active::after{content:"";position:absolute;bottom:0;height:2px;width:40%;background:var(--grad);animation:pmscan 1.1s linear infinite}
.pm-step.active .n{color:var(--violet)}
.pm-step.done{border-color:rgba(52,211,153,.32)}
.pm-step.done .n{color:var(--ok)}
.pm-step.refused{border-color:rgba(251,191,36,.5)}
.pm-step.refused .n{color:var(--warn)}
.pm-step.skipped{opacity:.5;border-style:dashed}
@keyframes pmscan{0%{left:-40%}100%{left:100%}}
.pm-pending{display:flex;align-items:center;gap:.7rem;padding:.8rem 1rem;border:1px dashed var(--line2);
  border-radius:12px;color:var(--muted);font-size:.88rem;margin:.6rem 0}
.pm-spin{width:14px;height:14px;border-radius:50%;border:2px solid var(--line2);border-top-color:var(--teal);
  animation:pmspin .8s linear infinite;flex:none}
@keyframes pmspin{to{transform:rotate(360deg)}}

/* cards */
.pm-card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1.2rem 1.35rem;margin:.9rem 0}
.pm-card h3,.pm-h h3,.pm-cmph h3{font-size:.74rem!important;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted)!important;font-weight:600!important;margin:0!important;padding:0!important}
.pm-card h3{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin:0 0 .85rem!important}
.pm-badges{display:flex;gap:.4rem;flex-wrap:wrap}
.pm-badge{font-size:.72rem;padding:.2rem .62rem;border-radius:999px;letter-spacing:0;text-transform:none;font-weight:600;white-space:nowrap}
.pm-badge.ok{color:var(--ok);background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3)}
.pm-badge.bad{color:var(--bad);background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3)}
.pm-badge.warn{color:var(--warn);background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3)}
.pm-badge.cache{color:#B9A8FF;background:rgba(124,92,255,.12);border:1px solid rgba(124,92,255,.35)}
.pm-answer{font-size:1.14rem;line-height:1.78;color:var(--text)}
.pm-cite{font-family:var(--mono)!important;font-size:.66rem;color:var(--teal);background:rgba(61,217,197,.1);
  border:1px solid rgba(61,217,197,.28);border-radius:5px;padding:0 .3rem;margin:0 .15rem 0 .2rem}
.pm-sub{font-size:.84rem;color:var(--muted);margin:-.3rem 0 .95rem;line-height:1.55}
.pm-small{font-size:.82rem;color:var(--muted);margin:.6rem 0 0}

/* the gate */
.pm-rows{display:flex;flex-direction:column;gap:.55rem}
.pm-row{display:grid;grid-template-columns:26px 1fr;gap:.75rem;padding:.8rem .9rem;border-radius:12px;
  border:1px solid var(--line);background:var(--surface2)}
.pm-row.kept{border-color:rgba(52,211,153,.24)}
.pm-row.del{border-color:rgba(248,113,113,.32);background:rgba(248,113,113,.045)}
.pm-ic{width:24px;height:24px;border-radius:7px;display:grid;place-items:center;font-size:.8rem;font-weight:800}
.kept .pm-ic{background:rgba(52,211,153,.15);color:var(--ok)}
.del .pm-ic{background:rgba(248,113,113,.15);color:var(--bad)}
.pm-st{font-size:.98rem;line-height:1.55}
.del .pm-st{color:#CDB0B0;text-decoration:line-through;text-decoration-color:rgba(248,113,113,.75);text-decoration-thickness:2px}
.pm-meta{display:flex;flex-wrap:wrap;align-items:center;gap:.7rem;margin-top:.5rem;font-size:.76rem;color:var(--muted)}
.pm-bar{position:relative;display:inline-block;width:120px;height:6px;border-radius:99px;background:#252B3A}
.pm-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:99px}
.kept .pm-bar i{background:var(--ok)}
.del .pm-bar i{background:var(--bad)}
.pm-bar b{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--text);opacity:.55;border-radius:2px}
.pm-score{font-family:var(--mono)!important;color:var(--text)}
.pm-num{font-family:var(--mono)!important;font-size:.7rem;color:var(--teal);border:1px solid rgba(61,217,197,.35);
  border-radius:5px;padding:0 .35rem;font-weight:500}

/* tiles */
.pm-tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:.9rem 0}
.pm-tile{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:.85rem 1rem}
.pm-tile b{display:block;font-size:1.65rem;font-weight:750;letter-spacing:-.02em;line-height:1.2}
.pm-tile span{font-size:.76rem;color:var(--muted)}
.pm-tile.ok b{color:var(--ok)}
.pm-tile.bad b{color:var(--bad)}

/* sources */
.pm-h{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;margin:1.5rem 0 .7rem}
.pm-h span{font-size:.78rem;color:var(--faint)}
.pm-srcs{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:.7rem}
.pm-src{background:var(--surface2);border:1px solid var(--line);border-radius:14px;padding:1rem 1.05rem;
  display:flex;flex-direction:column;gap:.5rem}
.pm-src .top{display:flex;gap:.55rem;align-items:center}
.pm-scheme{font-weight:650;font-size:.92rem}
.pm-doc{font-size:.76rem;color:var(--muted);line-height:1.45}
.pm-quote{font-size:.86rem;line-height:1.6;color:#C7CBD6;border-left:2px solid var(--violet);padding-left:.75rem;
  display:-webkit-box;-webkit-line-clamp:6;-webkit-box-orient:vertical;overflow:hidden}
.pm-srcf{display:flex;justify-content:space-between;align-items:center;gap:.6rem;font-size:.74rem;color:var(--faint);margin-top:.15rem}
.pm-srcf a{color:var(--teal)!important;text-decoration:none!important;font-size:.78rem}
.pm-ret{padding:.75rem 0;border-bottom:1px solid var(--line)}
.pm-ret .pm-score{margin-right:.6rem;font-size:.76rem}

/* refusal + notes */
.pm-refuse{background:linear-gradient(180deg,rgba(251,191,36,.08),rgba(20,24,33,.92));border:1px solid rgba(251,191,36,.38);
  border-radius:16px;padding:1.25rem 1.35rem;margin:.9rem 0}
.pm-refuse .h{display:flex;align-items:center;gap:.65rem;font-weight:700;font-size:1.12rem;color:#FCD34D}
.pm-rico{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;background:rgba(251,191,36,.18);
  color:var(--warn);font-size:.85rem;font-weight:800}
.pm-refuse p{color:var(--text);line-height:1.65;margin:.7rem 0 0}
.pm-refuse .why{font-size:.78rem;color:var(--muted);font-family:var(--mono)!important;margin-top:.55rem}
.pm-next{margin-top:.95rem;padding:.85rem .95rem;border-radius:12px;background:rgba(61,217,197,.06);
  border:1px solid rgba(61,217,197,.22);font-size:.9rem;color:#BFEFE8;line-height:1.6}
.pm-note{margin:.3rem 0 1.1rem;padding:.8rem 1rem;border-radius:12px;border:1px solid rgba(251,191,36,.35);
  background:rgba(251,191,36,.06);color:#FDE68A;font-size:.9rem;line-height:1.55}
.pm-note code{color:#FDE68A;background:rgba(0,0,0,.25);padding:.05rem .3rem;border-radius:4px}

/* comparison */
.pm-cmph{margin:1.3rem 0 .2rem}
.pm-cmph span{display:block;font-size:.84rem;color:var(--muted);margin-top:.4rem;max-width:780px;line-height:1.55}
.pm-cmp{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin:.8rem 0}
.pm-side{border-radius:16px;padding:1.15rem 1.25rem;border:1px solid var(--line2);background:var(--surface)}
.pm-side.ours{border-color:rgba(124,92,255,.5);box-shadow:0 0 0 1px rgba(124,92,255,.14),0 0 44px rgba(124,92,255,.12)}
.pm-sideh{display:flex;justify-content:space-between;align-items:center;gap:.6rem}
.pm-side .k{font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;color:var(--text);font-weight:700}
.pm-side .s{font-size:.78rem;color:var(--faint);margin:.3rem 0 .85rem}
.pm-side .body{font-size:.96rem;line-height:1.75}
.pm-hl{border-radius:4px;padding:.05rem .2rem}
.pm-hl.yes{background:rgba(52,211,153,.12);box-shadow:inset 0 -2px 0 rgba(52,211,153,.55)}
.pm-hl.no{border-bottom:2px dashed rgba(248,113,113,.55)}
.pm-hl.con{background:rgba(248,113,113,.2);box-shadow:inset 0 -2px 0 var(--bad);color:#FECACA}
.pm-legend{display:flex;flex-wrap:wrap;gap:1rem;font-size:.74rem;color:var(--muted);margin-top:.9rem}
.pm-legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:.35rem;vertical-align:-1px}
.pm-legend i.g{background:var(--ok)}
.pm-legend i.r{background:transparent;border-bottom:2px dashed var(--bad);border-radius:0;height:6px}
.pm-legend i.c{background:var(--bad)}

/* sections */
.pm-sec{margin-top:3.4rem}
.pm-sec h2{font-size:1.6rem!important;font-weight:750!important;letter-spacing:-.025em;margin:0 0 .35rem!important;
  padding:0!important;color:var(--text)!important}
.pm-sec .sub{color:var(--muted);margin:0 0 1.25rem;font-size:.96rem;line-height:1.55}
.pm-how{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem}
.pm-hw{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1.1rem}
.pm-hw.gate{border-color:rgba(124,92,255,.55);background:linear-gradient(180deg,rgba(124,92,255,.12),var(--surface))}
.pm-hw .n{font-family:var(--mono)!important;font-size:.72rem;color:var(--faint)}
.pm-hw.gate .n{color:var(--violet)}
.pm-hw .t{font-weight:700;margin:.35rem 0 .4rem}
.pm-hw .d{font-size:.85rem;color:var(--muted);line-height:1.58}
.pm-proof{margin-top:.9rem;padding:.85rem 1rem;border-radius:12px;border:1px solid var(--line);background:var(--surface);
  font-size:.87rem;color:var(--muted)}
.pm-proof b{color:var(--ok);font-size:1.02rem}
.pm-corpus{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.6rem}
.pm-sch{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:.9rem 1rem}
.pm-sch b{display:block;font-size:.98rem}
.pm-sch span{display:block;font-size:.76rem;color:var(--muted);margin-top:.15rem;line-height:1.4}
.pm-sch em{display:block;font-style:normal;font-family:var(--mono)!important;font-size:.72rem;color:var(--teal);margin-top:.55rem}
.pm-foot{margin-top:3.4rem;padding-top:1.3rem;border-top:1px solid var(--line);display:flex;justify-content:space-between;
  align-items:flex-start;gap:1.2rem;flex-wrap:wrap;font-size:.78rem;color:var(--faint);line-height:1.7}
.pm-foot code{font-family:var(--mono)!important;color:var(--muted);background:transparent;font-size:.72rem}
.pm-foot .r{text-align:right}

[data-testid="stExpander"] details{background:var(--surface);border:1px solid var(--line)!important;border-radius:14px}
[data-testid="stExpander"] summary p{color:var(--muted)!important;font-size:.86rem!important}

@media (max-width:820px){
  .pm-steps,.pm-how,.pm-tiles{grid-template-columns:repeat(2,1fr)}
  .pm-cmp{grid-template-columns:1fr}
  .pm-navr .pm-chip{display:none}
  .pm-foot .r{text-align:left}
  [data-testid="stMainBlockContainer"],.block-container{padding:1rem 1rem 3rem!important}
}
</style>"""


# ---------------------------------------------------------------- engine
@st.cache_resource(show_spinner="Loading the verifier — the first load takes about a minute…")
def load_engine() -> Pramaan:
    return Pramaan()


@st.cache_resource
def answer_cache() -> dict:
    """Answers already verified, shared across visitors.

    A live answer takes 10-70s on CPU; someone clicking an example should not
    wait for one another visitor already ran. Entries are the real verified
    output, and every cached answer is labelled as such on screen.
    """
    return {}


def _remember(cache: dict, key, value, cap: int = 256) -> None:
    cache[key] = value
    while len(cache) > cap:
        cache.pop(next(iter(cache)))


def corpus_stats(engine: Pramaan) -> dict:
    by = defaultdict(lambda: {"clauses": 0, "docs": set()})
    for c in engine.retriever.clauses:
        by[c.scheme]["clauses"] += 1
        by[c.scheme]["docs"].add(c.doc_id)
    schemes = []
    for name, d in sorted(by.items()):
        short, _, full = name.partition(" (")
        schemes.append({"short": short, "full": full.rstrip(")"),
                        "clauses": d["clauses"], "docs": len(d["docs"])})
    return {
        "clauses": len(engine.retriever.clauses),
        "docs": len({c.doc_id for c in engine.retriever.clauses}),
        "schemes": schemes,
    }


# ---------------------------------------------------------------- page chrome
def render_nav(provider: str) -> None:
    live = ("Extractive mode — no LLM key" if provider == "extractive"
            else "Every answer checked against official documents")
    st.html(
        '<nav class="pm-nav">'
        '<div class="pm-brand"><span class="pm-logo">◈</span>PRAMAAN</div>'
        '<div class="pm-navr">'
        f'<span class="pm-chip"><span class="pm-dot"></span>{E(live)}</span>'
        f'<a class="pm-link" href="{REPO_URL}" target="_blank" rel="noopener">GitHub ↗</a>'
        '</div></nav>'
    )


def render_hero(stats: dict) -> None:
    pills = [
        (str(stats["clauses"]), "clauses indexed"),
        (str(stats["docs"]), "official documents"),
        (str(len(stats["schemes"])), "welfare schemes"),
        (MEASURED_ACCURACY, "measured accuracy"),
    ]
    pill_html = "".join(
        f'<div class="pm-stat"><b>{E(v)}</b><span>{E(k)}</span></div>' for v, k in pills
    )
    st.html(
        '<section class="pm-hero">'
        '<div class="pm-eyebrow">Proof-grounded eligibility guidance</div>'
        '<h1>The AI advisor that <span class="pm-grad">refuses to guess.</span></h1>'
        '<p class="pm-lede">Ask about an Indian welfare scheme. Every sentence of the answer is '
        'checked against a clause in an official government document — anything it cannot '
        'prove is deleted before you see it, and if too little survives, it refuses.</p>'
        f'<div class="pm-stats">{pill_html}</div>'
        '</section>'
    )


def render_how() -> None:
    steps = [
        ("01", "Retrieve", "Hybrid search — meaning plus exact terms — finds the governing "
         "clauses among those extracted from official PDFs.", False),
        ("02", "Draft", "A language model writes an answer from those clauses alone, stating "
         "the rules rather than ruling on your case.", False),
        ("03", "Verify · the gate", "A separate classifier checks every sentence against the "
         "clauses and deletes anything none of them entails. A prompt can be talked out of; "
         "a threshold cannot.", True),
        ("04", "Deliver or refuse", "Proved sentences reach you with their citation. If too "
         "little survives, it refuses and points you to a real office.", False),
    ]
    cards = "".join(
        f'<div class="pm-hw{" gate" if gate else ""}"><div class="n">{n}</div>'
        f'<div class="t">{E(t)}</div><div class="d">{E(d)}</div></div>'
        for n, t, d, gate in steps
    )
    st.html(
        '<section class="pm-sec"><h2>How it works</h2>'
        '<p class="sub">Most AI systems trust the prompt to keep the model honest. '
        'PRAMAAN does not ask — it checks.</p>'
        f'<div class="pm-how">{cards}</div>'
        f'<div class="pm-proof"><b>{E(MEASURED_ACCURACY)}</b> on a 13-question test set · '
        f'{E(MEASURED_DETAIL)} · 100% of kept sentences carry a citation</div>'
        '</section>'
    )


def render_corpus(stats: dict) -> None:
    cards = "".join(
        f'<div class="pm-sch"><b>{E(s["short"])}</b><span>{E(s["full"])}</span>'
        f'<em>{s["clauses"]} clauses · {s["docs"]} document{"s" if s["docs"] != 1 else ""}</em></div>'
        for s in stats["schemes"]
    )
    st.html(
        '<section class="pm-sec"><h2>What it knows</h2>'
        f'<p class="sub">{stats["clauses"]} clauses extracted verbatim from {stats["docs"]} official '
        'Government of India documents. Nothing is paraphrased — every answer quotes its source.</p>'
        f'<div class="pm-corpus">{cards}</div></section>'
    )


def render_footer(engine: Pramaan, provider: str) -> None:
    model = model_for(provider) if provider != "extractive" else "none"
    cfg = (f"drafter {provider} · {model} · verifier {engine.verifier.name} · "
           f"threshold {ENTAILMENT_THRESHOLD} · retrieval {engine.retriever.backend}")
    st.html(
        '<div class="pm-foot">'
        '<div>Built for CodeArambh 2.0 · Team Techtonics — Ujjawal Srivastava, Amey Dongre, '
        'Tejas Gupta<br>Not an official government service. Always confirm your eligibility '
        'with the relevant department.</div>'
        f'<div class="r"><code>{E(cfg)}</code><br>'
        f'<a class="pm-flink" href="{REPO_URL}" target="_blank" rel="noopener">Source on GitHub ↗</a></div>'
        '</div>'
    )


# ---------------------------------------------------------------- live progress
def stepper_html(current: str, detail: str = "", *, refused: bool = False,
                 elapsed: float = 0.0, cached: bool = False, verified: bool = True) -> str:
    keys = [k for k, _, _ in STEPS]
    at = keys.index(current)
    cells = []
    for i, (key, title, desc) in enumerate(STEPS):
        if current == "done":
            state = "refused" if key == "done" and refused else "done"
            # A refusal can happen before anything is checked -- the drafter
            # found no evidence at all. Ticking "Verify" then would claim work
            # that never happened, which is the one thing this app must not do.
            if key == "verify" and not verified:
                state = "skipped"
        else:
            state = "done" if i < at else ("active" if i == at else "pending")
        text = detail if (i == at and detail) else desc
        if key == "verify" and state == "skipped":
            text = "Skipped — nothing to check"
        if key == "done" and current == "done":
            if cached:
                text = f"From cache · verified in {elapsed:.1f}s"
            elif refused:
                text = f"Refused after {elapsed:.1f}s"
            else:
                text = f"Answered in {elapsed:.1f}s"
        mark = " ✓" if state == "done" else ""
        cells.append(
            f'<div class="pm-step {state}"><div class="n">STEP 0{i + 1}{mark}</div>'
            f'<div class="t">{title}</div><div class="d">{E(text)}</div></div>'
        )
    return f'<div class="pm-steps">{"".join(cells)}</div>'


def pipeline_progress(slot, t0: float, n_clauses: int):
    def cb(stage: str, **d) -> None:
        if stage == "retrieve":
            slot.html(stepper_html("retrieve", f"Searching {n_clauses} clauses"))
        elif stage == "draft":
            slot.html(stepper_html("draft", f"Using the {d.get('clauses', 0)} best clauses"))
        elif stage == "verify":
            i, n = d.get("i", 0), d.get("n", 0)
            slot.html(stepper_html("verify", f"Sentence {i} of {n}" if i else f"{n} sentences to check"))
        elif stage == "done":
            slot.html(stepper_html("done", refused=d.get("abstained", False),
                                   elapsed=time.time() - t0))
    return cb


def plain_progress(slot):
    def cb(stage: str, **d) -> None:
        if stage == "plain_draft":
            msg = "Asking the same model with no documents and no gate…"
        else:
            msg = (f"Checking the plain chatbot's sentence {d.get('i', 0)} of "
                   f"{d.get('n', 0)} against the documents…")
        slot.html(f'<div class="pm-pending"><span class="pm-spin"></span>{E(msg)}</div>')
    return cb


# ---------------------------------------------------------------- results
def _hl(vd) -> str:
    if vd.supported:
        return "yes"
    return "con" if vd.contradicted_by is not None else "no"


def _hl_title(vd) -> str:
    if vd.contradicted_by is None:
        return ""
    c = vd.contradicted_by
    return (f' title="Contradicted by {E(c.doc_title, quote=True)}, p.{c.page} '
            f'({vd.contradiction:.2f})"')


def _bar(score: float) -> str:
    return (f'<span class="pm-bar"><i style="width:{max(score, 0.02) * 100:.0f}%"></i>'
            f'<b style="left:{ENTAILMENT_THRESHOLD * 100:.0f}%"></b></span>')


def render_answer(out, nums: dict) -> None:
    v = out.verification
    parts = []
    for vd in v.kept:
        n = nums.get(vd.clause.clause_id) if vd.clause else None
        tag = f'<span class="pm-cite">{n}</span>' if n else ""
        parts.append(f"{E(vd.sentence)}{tag}")
    badges = f'<span class="pm-badge ok">{len(v.kept)} of {len(v.verdicts)} sentences proved</span>'
    if out.cached:
        badges = '<span class="pm-badge cache">from cache</span>' + badges
    st.html(
        '<div class="pm-card">'
        f'<h3><span>Verified answer</span><span class="pm-badges">{badges}</span></h3>'
        f'<div class="pm-answer">{" ".join(parts)}</div>'
        '</div>'
    )


def render_refusal(out) -> None:
    reason = out.reason or ""
    note = ""
    if reason.startswith(PROVIDER_FAILURE):
        note = ('<p class="pm-small">This is a problem reaching the language model, not a '
                'judgement about your question. Try again in a moment.</p>')
    st.html(
        '<div class="pm-refuse">'
        '<div class="h"><span class="pm-rico">!</span>PRAMAAN is not answering this one</div>'
        f'<p>{E(out.answer)}</p>'
        f'<div class="why">why: {E(reason)}</div>{note}'
        f'<div class="pm-next">{E(out.next_step())}</div>'
        '</div>'
    )


def render_gate(v, nums: dict, refused: bool) -> None:
    k = len(v.stripped)
    if not k:
        sub = "Every sentence the model wrote was proved against a clause — nothing deleted this time."
        badge = '<span class="pm-badge ok">0 deleted</span>'
    else:
        sub = (f"The model wrote {k} sentence{'s' if k != 1 else ''} that no clause supports. "
               + ("So little survived that PRAMAAN refused." if refused
                  else "Deleted before you saw the answer."))
        badge = f'<span class="pm-badge bad">{k} deleted</span>'
    rows = []
    for vd in v.verdicts:
        kept = vd.supported
        if kept and vd.clause is not None:
            n = nums.get(vd.clause.clause_id)
            where = (f'cites <span class="pm-num">{n}</span> {E(vd.clause.doc_title)}, '
                     f'p.{vd.clause.page}') if n else "proved"
        elif vd.contradicted_by is not None:
            c = vd.contradicted_by
            where = (f"deleted — contradicted by {E(c.doc_title)}, p.{c.page} "
                     f"({vd.contradiction:.2f})")
        else:
            where = "deleted — no clause entails it"
        rows.append(
            f'<div class="pm-row {"kept" if kept else "del"}">'
            f'<span class="pm-ic">{"✓" if kept else "✕"}</span><div>'
            f'<div class="pm-st">{E(vd.sentence)}</div>'
            f'<div class="pm-meta">{_bar(vd.score)}<span class="pm-score">{vd.score:.2f}</span>'
            f'<span>{where}</span></div></div></div>'
        )
    st.html(
        '<div class="pm-card">'
        f'<h3><span>What the gate did</span><span class="pm-badges">{badge}</span></h3>'
        f'<p class="pm-sub">{E(sub)} The white tick on each bar is the '
        f'{ENTAILMENT_THRESHOLD:.2f} threshold a sentence must clear.</p>'
        f'<div class="pm-rows">{"".join(rows)}</div>'
        '</div>'
    )


def render_tiles(out) -> None:
    v = out.verification
    drafted = len(v.verdicts) if v else 0
    kept = len(v.kept) if v else 0
    deleted = len(v.stripped) if v else 0
    tiles = [
        ("", str(drafted), "sentences drafted"),
        ("ok", str(kept), "proved and kept"),
        ("bad" if deleted else "", str(deleted), "deleted by the gate"),
        ("", f"{out.elapsed:.1f}s",
         "verified earlier · served from cache" if out.cached else "end to end"),
    ]
    st.html('<div class="pm-tiles">' + "".join(
        f'<div class="pm-tile {c}"><b>{E(val)}</b><span>{E(lab)}</span></div>'
        for c, val, lab in tiles) + "</div>")


def render_sources(out, nums: dict) -> None:
    cards = []
    for c in out.citations:
        short = c.scheme.partition(" (")[0]
        fetched = f"fetched {E(c.retrieved)}" if c.retrieved else ""
        cards.append(
            '<div class="pm-src">'
            f'<div class="top"><span class="pm-num">{nums[c.clause_id]}</span>'
            f'<span class="pm-scheme">{E(short)}</span></div>'
            f'<div class="pm-doc">{E(c.doc_title)} · page {c.page} · {E(c.authority)}</div>'
            f'<div class="pm-quote">{E(c.text)}</div>'
            f'<div class="pm-srcf"><a href="{E(c.url, quote=True)}" target="_blank" '
            f'rel="noopener">Open the source PDF ↗</a><span>{fetched}</span></div>'
            '</div>'
        )
    n = len(cards)
    st.html(
        f'<div class="pm-h"><h3>Where this comes from</h3>'
        f'<span>{n} clause{"s" if n != 1 else ""}, quoted verbatim</span></div>'
        f'<div class="pm-srcs">{"".join(cards)}</div>'
    )


def render_compare(out, plain) -> None:
    summary = ""
    if plain.error:
        left = f'<p class="pm-small">Could not reach the plain chatbot: {E(plain.error)}</p>'
    elif plain.verification and plain.verification.verdicts:
        pv = plain.verification
        left = '<div class="body">' + " ".join(
            f'<span class="pm-hl {_hl(vd)}"{_hl_title(vd)}>{E(vd.sentence)}</span>'
            for vd in pv.verdicts) + "</div>"
        found, total = len(pv.kept), len(pv.verdicts)
        refuted = sum(1 for vd in pv.verdicts if vd.contradicted_by is not None)
        summary = ('<span class="pm-badges">'
                   f'<span class="pm-badge {"ok" if found == total else "bad"}">'
                   f'{found} of {total} found in the documents</span>'
                   + (f'<span class="pm-badge bad">{refuted} contradicted</span>' if refuted else "")
                   + '</span>')
    else:
        left = f'<div class="body">{E(plain.text or "(no answer)")}</div>'

    if out.abstained:
        right = ('<div class="body"><b>Refused.</b> No answer could be proved from the '
                 'official documents, so none was given.</div>')
        rbadge = '<span class="pm-badge warn">refused</span>'
    else:
        v = out.verification
        right = f'<div class="body">{E(out.answer)}</div>'
        rbadge = f'<span class="pm-badge ok">{len(v.kept)} of {len(v.verdicts)} proved</span>'

    st.html(
        '<div class="pm-cmph"><h3>Same question, two systems</h3>'
        '<span>A red sentence is not necessarily false — it is one the official documents we '
        'hold do not support. That is exactly the kind of sentence PRAMAAN will not say.</span></div>'
        '<div class="pm-cmp">'
        f'<div class="pm-side"><div class="pm-sideh"><span class="k">Plain chatbot</span>{summary}</div>'
        '<div class="s">Same model · no documents · no verification</div>'
        f'{left}'
        '<div class="pm-legend"><span><i class="g"></i>found in the documents</span>'
        '<span><i class="r"></i>not found — unproven</span>'
        '<span><i class="c"></i>contradicted by the documents</span></div></div>'
        f'<div class="pm-side ours"><div class="pm-sideh"><span class="k">PRAMAAN</span>{rbadge}</div>'
        '<div class="s">Same model · official documents · verification gate</div>'
        f'{right}</div>'
        '</div>'
    )


def render_result(out, plain) -> None:
    nums = {c.clause_id: i for i, c in enumerate(out.citations, 1)}
    if plain is not None:
        render_compare(out, plain)
    if out.abstained:
        render_refusal(out)
    else:
        render_answer(out, nums)
    if out.verification and out.verification.verdicts:
        render_gate(out.verification, nums, out.abstained)
    render_tiles(out)
    if out.citations and not out.abstained:
        render_sources(out, nums)
    if out.retrieved:
        with st.expander(f"Clauses retrieved before drafting · {len(out.retrieved)}"):
            st.html("".join(
                f'<div class="pm-ret"><span class="pm-score">{s:.3f}</span>'
                f'<span class="pm-doc">{E(c.scheme.partition(" (")[0])} · {E(c.doc_title)} · '
                f'p.{c.page}</span><div class="pm-quote">{E(c.text[:420])}'
                f'{"…" if len(c.text) > 420 else ""}</div></div>'
                for c, s in out.retrieved))


# ---------------------------------------------------------------- flow
def run(engine: Pramaan, question: str, slot, provider: str, stats: dict) -> None:
    cache = answer_cache()
    model = model_for(provider) if provider != "extractive" else ""
    key = (re.sub(r"\s+", " ", question.strip().lower()), provider, model, ENTAILMENT_THRESHOLD)

    hit = cache.get(key)
    if hit is not None:
        out = replace(hit, cached=True)
        # Show the finished pipeline straight away. Otherwise, while a slow
        # comparison runs below, this slot sits empty above stale results.
        slot.html(stepper_html(
            "done", refused=out.abstained, elapsed=out.elapsed, cached=True,
            verified=bool(out.verification and out.verification.verdicts)))
    else:
        out = engine.ask(question, progress=pipeline_progress(slot, time.time(), stats["clauses"]))
        # Never cache a provider failure: it says nothing about the question,
        # and caching it would keep refusing after the provider recovered.
        if not out.reason.startswith(PROVIDER_FAILURE):
            _remember(cache, key, out)

    plain = None
    if st.session_state.get("compare") and provider != "extractive":
        pkey = key + ("plain",)
        plain = cache.get(pkey)
        if plain is None:
            pslot = st.empty()
            plain = engine.compare_plain(question, [c for c, _ in out.retrieved],
                                         progress=plain_progress(pslot))
            pslot.empty()
            if not plain.error:
                _remember(cache, pkey, plain)

    st.session_state.last = {"out": out, "plain": plain}


def _pick(question: str) -> None:
    st.session_state.q = question
    st.session_state.pending = True


def _go() -> None:
    st.session_state.pending = True


def main() -> None:
    st.html(CSS)
    try:
        engine = load_engine()
    except Exception as exc:
        render_nav("extractive")
        st.html(
            '<div class="pm-refuse"><div class="h"><span class="pm-rico">!</span>'
            'The engine could not start</div>'
            f'<p>{E(type(exc).__name__)}: {E(str(exc))}</p>'
            '<div class="why">run scripts/build_index.py, and check that torch installed</div></div>'
        )
        return

    provider = detect_provider()
    stats = corpus_stats(engine)
    render_nav(provider)
    render_hero(stats)
    if provider == "extractive":
        st.html(
            '<div class="pm-note">No language-model key is configured, so answers are '
            '<b>extractive</b> — copied straight from clauses. The app works, but the gate is '
            'not really being tested. Add a free <code>GROQ_API_KEY</code> to see it work.</div>'
        )

    st.session_state.setdefault("q", "")
    with st.container(key="askcard"):
        st.html('<div class="pm-label">Ask about a welfare scheme</div>')
        st.text_area(
            "Your question", key="q", height=100, label_visibility="collapsed",
            placeholder="e.g. I own 2 hectares and paid income tax last year — do I qualify for PM-KISAN?",
        )
        left, right = st.columns([3, 1.15], vertical_alignment="center")
        with left:
            st.toggle(
                "Also ask a plain chatbot, side by side", key="compare",
                disabled=provider == "extractive",
                help="The same model, given no documents and no verification gate. Its "
                     "sentences are then checked against the official documents, so you can "
                     "see what an ordinary chatbot would have told you. Adds 20-40 seconds.",
            )
        with right:
            st.button("Verify & answer  →", type="primary", key="askbtn",
                      on_click=_go, use_container_width=True)
        st.html('<div class="pm-label pm-gap">Or try an example</div>')
        for start in (0, 3):
            cols = st.columns(3)
            for offset, col in enumerate(cols):
                idx = start + offset
                label, question = EXAMPLES[idx]
                col.button(label, key=f"ex{idx}", on_click=_pick, args=(question,),
                           use_container_width=True)

    step_slot = st.empty()
    if st.session_state.pop("pending", False):
        question = st.session_state.get("q", "").strip()
        if question:
            run(engine, question, step_slot, provider, stats)
        else:
            st.html('<div class="pm-note">Type a question first, or pick one of the examples.</div>')

    last = st.session_state.get("last")
    if last:
        out = last["out"]
        step_slot.html(stepper_html(
            "done", refused=out.abstained, elapsed=out.elapsed, cached=out.cached,
            verified=bool(out.verification and out.verification.verdicts)))
        render_result(out, last.get("plain"))

    render_how()
    render_corpus(stats)
    render_footer(engine, provider)


if __name__ == "__main__":
    main()
