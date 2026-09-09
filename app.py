"""PRAMAAN - Streamlit demo.

The interface deliberately shows its own working: what was retrieved, what the
model drafted, and what the gate deleted. A system that claims it refuses to
guess has to let you watch it refuse.
"""
from __future__ import annotations

import os
import tempfile

import streamlit as st

# Streamlit Community Cloud supplies secrets through st.secrets, not the
# environment, so copy them across BEFORE pramaan is imported -- config.py
# reads os.getenv at import time. Existing env vars win, and a missing secrets
# file is normal when running locally off a .env.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from pramaan import asr  # noqa: E402
from pramaan.config import ENTAILMENT_THRESHOLD, TOP_K  # noqa: E402
from pramaan.drafter import detect_provider, model_for  # noqa: E402
from pramaan.pipeline import Pramaan  # noqa: E402

st.set_page_config(page_title="PRAMAAN", page_icon="🪔", layout="wide")

EXAMPLES = [
    "I am a farmer with 1 hectare of land. Can I get PM-KISAN?",
    "I paid income tax last year. Am I still eligible for PM-KISAN?",
    "I am 42 years old. Can I join Atal Pension Yojana?",
    "How much does PMJJBY cost per year and what does it pay out?",
    "What is the entry age for PM Kisan Maan Dhan Yojana?",
    "Will PRAMAAN tell me tomorrow's weather?",
]


@st.cache_resource(show_spinner="Loading retriever and verifier…")
def load_engine():
    return Pramaan()


def sidebar():
    with st.sidebar:
        st.markdown("### PRAMAAN")
        st.caption("Proof-grounded eligibility guidance for Indian welfare schemes.")
        st.markdown(
            "Every sentence you see below has been checked against a clause in an "
            "official government document. Sentences that failed the check were "
            "deleted before the answer reached you."
        )
        st.divider()
        st.markdown("**Corpus**")
        try:
            eng = load_engine()
            schemes = sorted({c.scheme for c in eng.retriever.clauses})
            st.caption(f"{len(eng.retriever.clauses)} clauses · {len(schemes)} schemes")
            for s in schemes:
                st.caption(f"· {s}")
            st.divider()
            st.markdown("**Configuration**")
            st.caption(f"Retrieval: {eng.retriever.backend} · top-{TOP_K}")
            st.caption(f"Verifier: {eng.verifier.name}")
            st.caption(f"Entailment threshold: {ENTAILMENT_THRESHOLD}")
            provider = detect_provider()
            if provider == "extractive":
                st.warning(
                    "No LLM key found, so answers are **extractive** (sentences "
                    "copied from clauses). The app works, but the gate is not "
                    "being exercised. Add a free GROQ_API_KEY to see it work.",
                    icon="⚠️",
                )
            else:
                st.caption(f"Drafter: {provider} · {model_for(provider)}")
        except Exception:
            st.caption("Index not built yet.")
        st.divider()
        st.caption(
            "Prototype for CodeArambh 2.0. Not an official government service. "
            "Always confirm your own eligibility with the relevant department."
        )


def render_outcome(out):
    if out.abstained:
        st.warning("**PRAMAAN is not answering this one.**")
        st.markdown(out.answer)
        if out.reason:
            st.caption(f"Reason: {out.reason}")
        st.info(out.next_step())
    else:
        st.markdown("#### Answer")
        st.markdown(out.answer)
        st.markdown("#### Where this comes from")
        for c in out.citations:
            with st.container(border=True):
                st.markdown(f"**{c.scheme}**")
                st.caption(f"{c.doc_title} · page {c.page} · {c.authority}")
                st.markdown(f"> {c.text}")
                cap = f"[Source document]({c.url})"
                if c.retrieved:
                    cap += f" · retrieved {c.retrieved}"
                st.caption(cap)

    v = out.verification
    if not v:
        return

    st.markdown("#### Verification trace")
    cols = st.columns(4)
    cols[0].metric("Sentences drafted", len(v.verdicts))
    cols[1].metric("Kept", len(v.kept))
    cols[2].metric("Blocked", len(v.stripped))
    cols[3].metric("Catch-rate", f"{v.catch_rate:.0%}")

    with st.expander("Show every sentence and its score"):
        for i, verdict in enumerate(v.verdicts, 1):
            mark = "✅" if verdict.supported else "🚫"
            st.markdown(f"{mark} **{i}.** {verdict.sentence}")
            if verdict.supported:
                st.caption(f"entailment {verdict.score:.2f} · {verdict.citation}")
            else:
                st.caption(
                    f"entailment {verdict.score:.2f} — below {ENTAILMENT_THRESHOLD}, "
                    "removed from the answer"
                )

    if v.stripped:
        with st.expander(f"What the model wrote that you never saw ({len(v.stripped)})"):
            st.caption(
                "These sentences were fluent and plausible. No clause supported them, "
                "so the gate deleted them."
            )
            for verdict in v.stripped:
                st.markdown(f"- ~~{verdict.sentence}~~")

    with st.expander("Retrieved clauses (before drafting)"):
        for c, score in out.retrieved:
            st.caption(f"similarity {score:.3f} · {c.scheme} · {c.doc_title} p.{c.page}")
            st.markdown(f"> {c.text[:400]}{'…' if len(c.text) > 400 else ''}")

    st.caption(f"Answered in {out.elapsed:.1f}s")


def main():
    sidebar()
    st.title("PRAMAAN")
    st.markdown(
        "##### The AI advisor that refuses to guess."
    )

    if "question" not in st.session_state:
        st.session_state.question = ""

    st.markdown("**Try one of these**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.question = ex

    question = st.text_area(
        "Ask about a scheme, in English or Hindi",
        value=st.session_state.question,
        height=90,
        placeholder="e.g. I own 2 hectares and paid income tax last year — do I qualify for PM-KISAN?",
    )

    if asr.available():
        audio = st.audio_input("…or ask out loud")
        if audio is not None:
            with st.spinner("Transcribing…"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                    fh.write(audio.getvalue())
                    path = fh.name
                try:
                    text, lang = asr.transcribe(path)
                    if text:
                        question = text
                        st.success(f"Heard ({lang}): {text}")
                except asr.ASRUnavailable as exc:
                    st.caption(str(exc))
                finally:
                    os.unlink(path)
    else:
        st.caption("Voice input off — `pip install faster-whisper` to enable it.")

    if st.button("Ask", type="primary") and question.strip():
        try:
            engine = load_engine()
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(
                f"Could not start the engine: {type(exc).__name__}: {exc}\n\n"
                "If this is about the verifier, install torch "
                "(`pip install -r requirements.txt`), or set "
                "`PRAMAAN_VERIFIER_BACKEND=llm` with an LLM key set."
            )
            return
        with st.spinner("Retrieving clauses, drafting, verifying…"):
            out = engine.ask(question)
        render_outcome(out)


if __name__ == "__main__":
    main()
