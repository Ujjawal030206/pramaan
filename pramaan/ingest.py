"""Turn official PDFs into citable clauses.

A "clause" here is a chunk of verbatim document text plus enough metadata to
put a citation on screen: which scheme, which document, which page. We never
rewrite the text -- if we cannot quote it, we cannot cite it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from .config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, RAW_DIR, SOURCES_FILE


@dataclass
class Clause:
    clause_id: str
    text: str
    scheme: str
    doc_id: str
    doc_title: str
    authority: str
    url: str
    page: int
    retrieved: str = ""

    def citation(self) -> str:
        return f"{self.doc_title}, p. {self.page} ({self.authority})"


# Government PDFs are full of running headers, page numbers and dotted leaders
# from tables of contents. None of that is clause text.
_NOISE = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.I),
    re.compile(r"^\s*\d{1,3}\s*$"),
    re.compile(r"^[\s.·_—-]{4,}$"),
]
_DOTTED_TOC = re.compile(r"\.{5,}\s*\d+\s*$")


def _clean_line(line: str) -> str:
    line = line.replace("\xa0", " ").rstrip()
    line = re.sub(r"[ \t]{2,}", " ", line)
    return line


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _DOTTED_TOC.search(s):
        return True
    return any(p.match(s) for p in _NOISE)


def _page_paragraphs(raw: str) -> list[str]:
    """Split a page's text into paragraphs, dropping boilerplate."""
    lines = [_clean_line(x) for x in raw.splitlines()]
    lines = [x for x in lines if not _is_noise(x)]

    paras, cur = [], []
    for line in lines:
        if not line.strip():
            if cur:
                paras.append(" ".join(cur).strip())
                cur = []
            continue
        cur.append(line.strip())
    if cur:
        paras.append(" ".join(cur).strip())
    return [p for p in paras if p]


# These documents are written as numbered clauses -- "10.1", "4.2", "(a)",
# "vi)". A clause number is a topic boundary, and chunking across one is what
# ruins retrieval: gluing "10.1 The financial benefit of Rs.6000/- per year..."
# onto the end of a paragraph about nodal officers submitting lists produces an
# embedding dominated by nodal officers, and the benefit clause becomes
# unfindable by anyone asking how much money they get.
_CLAUSE_START = re.compile(
    r"""^\s*(?:
          \d+(?:\.\d+)+[.)]?\s+          # 10.1  4.2.1
        | \d+[.)]\s+(?=[A-Z(])           # 5.  7)   followed by a capital
        | \(\s*(?:[a-z]|[ivxlc]{1,4})\s*\)\s+   # (a)  (iv)
        | (?:[a-z]|[ivxlc]{1,4})[.)]\s+(?=[A-Z(])  # a)  vi)
    )""",
    re.VERBOSE,
)


def _split_long(text: str) -> list[str]:
    """Break an over-long clause on sentence boundaries."""
    out, cur = [], ""
    for s in re.split(r"(?<=[.;:])\s+", text):
        if cur and len(cur) + len(s) + 1 > CHUNK_MAX_CHARS:
            out.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur.strip())
    return out


# An enumerated list ITEM -- "(a)", "iv)", "v)". Distinct from a section number
# like "5.2", because a list item is meaningless without the stem that
# introduces it, whereas a numbered section is self-contained.
_LIST_ITEM = re.compile(
    r"^\s*\(?\s*(?:[a-z]|[ivxlc]{1,4})\s*[).]\s+", re.IGNORECASE
)

# A numbered section like "5.2" or "10.1.3" -- self-contained, and its arrival
# means any list we were inside has finished.
_SECTION_NUM = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")


def _pack(paras: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Group (paragraph, page) pairs into clause-aligned chunks.

    Two things this has to get right, both learned the hard way:

    1. Chunk across page boundaries. The PM-KISAN exclusion list introduces
       itself on page 2 and runs to page 3; chunking per page left "v) All
       Persons who paid Income Tax in last assessment year" stranded with no
       stem, so it entailed nothing and a true sentence about income-tax payers
       was deleted from the answer.
    2. Carry the stem onto list items. "All Persons who paid Income Tax in last
       assessment year" is a noun phrase. Only with "The following categories
       shall not be eligible for benefit" in front of it does it state a rule.

    Chunks are tagged with the page their first paragraph came from, which is
    what the citation shows.
    """
    packed: list[tuple[str, int]] = []
    buf, buf_page = "", 0
    stems: list[str] = []

    for text, page in paras:
        stripped = text.rstrip()
        is_item = bool(_LIST_ITEM.match(text))

        # A stem is a paragraph that ends in a colon -- that is exactly how
        # these documents introduce a list. We keep the two most recent so a
        # nested item inherits both levels: "4.1 The following categories ...
        # shall not be eligible ...:" plus "(b) Farmer families ...:".
        if stripped.endswith(":"):
            stems.append(stripped)
            del stems[:-2]
        elif _SECTION_NUM.match(text) and not is_item:
            stems.clear()  # a new numbered section means the list has ended

        starts_clause = bool(_CLAUSE_START.match(text))
        too_big = buf and len(buf) + len(text) + 1 > CHUNK_MAX_CHARS
        if buf and (too_big or (starts_clause and len(buf) >= CHUNK_MIN_CHARS)):
            packed.append((buf, buf_page))
            # Guard against duplicating the stem inside the NEW chunk only. An
            # earlier version tested the chunk being closed, which meant the
            # first item of every list -- the one directly after its stem --
            # was the one item that never got the stem attached.
            prefix = " ".join(stems)
            if is_item and prefix and not text.startswith(prefix[:40]):
                buf, buf_page = f"{prefix} {text}", page
            else:
                buf, buf_page = text, page
        elif buf:
            buf = f"{buf} {text}"
        else:
            buf, buf_page = text, page

    if buf:
        packed.append((buf, buf_page))

    out: list[tuple[str, int]] = []
    for chunk, page in packed:
        if len(chunk) > CHUNK_MAX_CHARS:
            out.extend((piece, page) for piece in _split_long(chunk))
        else:
            out.append((chunk, page))
    return [(c, p) for c, p in out if len(c) >= 60]


def load_manifest() -> list[dict]:
    data = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    return data["documents"]


def extract_clauses(entry: dict, pdf_path: Path) -> list[Clause]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))

    # Flatten the whole document into a paragraph stream first. Clauses and
    # lists routinely straddle a page break, and chunking per page cuts them.
    stream: list[tuple[str, int]] = []
    for pageno, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            continue
        stream.extend((para, pageno) for para in _page_paragraphs(raw))

    out: list[Clause] = []
    per_page: dict[int, int] = {}
    for chunk, pageno in _pack(stream):
        i = per_page.get(pageno, 0)
        per_page[pageno] = i + 1
        out.append(
            Clause(
                clause_id=f"{entry['id']}#p{pageno}c{i}",
                text=chunk,
                scheme=entry["scheme"],
                doc_id=entry["id"],
                doc_title=entry["title"],
                authority=entry["authority"],
                url=entry["url"],
                page=pageno,
                retrieved=entry.get("retrieved", ""),
            )
        )
    return out


def build_clause_set() -> list[Clause]:
    clauses: list[Clause] = []
    for entry in load_manifest():
        path = RAW_DIR / f"{entry['id']}.pdf"
        if not path.exists():
            print(f"  ! missing {path.name} -- run scripts/fetch_corpus.py")
            continue
        got = extract_clauses(entry, path)
        print(f"  {entry['id']:<16} {len(got):>5} clauses")
        clauses.extend(got)
    return clauses


def save_clauses(clauses: list[Clause], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in clauses:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_clauses(path: Path) -> list[Clause]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Clause(**json.loads(line)))
    return out
