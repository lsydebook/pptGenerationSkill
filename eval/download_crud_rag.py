"""Download the CRUD-RAG Read split (questions + gold news), not the 80k corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

SPLIT_URL = (
    "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/"
    "data/crud_split/split_merged.json"
)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "split_merged.json"


def download(url: str, dest: Path, timeout_s: float = 300.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n  -> {dest}")
    with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 256):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print(f"\r  {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="")
                    else:
                        print(f"\r  {done / 1e6:.1f} MB", end="")
            print()
    part.replace(dest)
    print(f"saved {dest.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        print(f"already exists: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
        return 0
    try:
        download(SPLIT_URL, args.out)
    except httpx.HTTPError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
