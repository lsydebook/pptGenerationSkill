"""Command-line entry point for Weekly Report PPT Agent v2.

Examples:
    python cli.py --input examples/input_sample.json --out out/weekly.pptx
    python cli.py -i examples/input_sample.json -o out/weekly.pptx --ai-image
    python cli.py -i examples/input_sample.json --accent "#2B5B84"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weekly Report PPT Agent v2")
    parser.add_argument("--input", "-i", help="path to input JSON file")
    parser.add_argument("--out", "-o", default="out/weekly.pptx", help="output .pptx path")
    parser.add_argument(
        "--accent",
        default="",
        help="accent color hex, e.g. #2B5B84",
    )
    parser.add_argument(
        "--ai-image",
        action="store_true",
        help="enable AI image generation (wan2.6-t2i)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (DEBUG/INFO/WARNING/ERROR)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    if not args.input:
        print("error: --input is required", file=sys.stderr)
        return 2

    from agent.graph import build_graph

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))

    graph = build_graph()

    final = graph.invoke(
        {
            "raw_input": raw,
            "output_path": args.out,
            "base_dir": str(Path(__file__).resolve().parent),
            "mock": False,
            "accent_color": args.accent,
            "enable_ai_image": args.ai_image,
        },
    )

    rendered = final.get("rendered_path")
    if not rendered:
        print("error: rendering failed", file=sys.stderr)
        return 1

    specs = final.get("slide_specs") or []
    print(f"OK: {len(specs)} slides -> {rendered}")

    warnings = final.get("warnings") or []
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
