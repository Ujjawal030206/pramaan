"""Download every official document listed in corpus/sources.yaml.

Run this before build_index.py. Documents are cached, so re-running only
fetches what is missing. The retrieval date is written back into sources.yaml
so that citations can state when we last saw the document -- a scheme circular
that was current in September may not be current in March.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from pramaan.config import RAW_DIR, SOURCES_FILE  # noqa: E402

# Several state and ministry portals still present incomplete certificate
# chains. We are fetching public documents over a read-only GET, so we proceed
# and record the URL we used; nothing secret is transmitted.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

UA = "Mozilla/5.0 (compatible; PRAMAAN-corpus-fetch/0.1; hackathon prototype)"


def fetch_one(entry: dict) -> tuple[dict, str]:
    dest = RAW_DIR / f"{entry['id']}.pdf"
    if dest.exists() and dest.stat().st_size > 1000:
        return entry, f"cached ({dest.stat().st_size // 1024} KB)"
    try:
        req = urllib.request.Request(entry["url"], headers={"User-Agent": UA})
        data = urllib.request.urlopen(req, timeout=60, context=_CTX).read()
    except Exception as exc:
        return entry, f"FAILED {type(exc).__name__}: {exc}"

    if not data.startswith(b"%PDF"):
        return entry, f"FAILED not a PDF (got {data[:16]!r})"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return entry, f"downloaded ({len(data) // 1024} KB)"


def main() -> int:
    manifest = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    entries = manifest["documents"]
    today = dt.date.today().isoformat()

    print(f"Fetching {len(entries)} official documents\n")
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for entry, status in ex.map(fetch_one, entries):
            print(f"  {entry['id']:<16} {status}")
            if not status.startswith("FAILED"):
                entry["retrieved"] = today
                ok += 1

    SOURCES_FILE.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"\n{ok}/{len(entries)} documents available in {RAW_DIR}")
    if ok == 0:
        print("Nothing fetched -- check your network, then re-run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
