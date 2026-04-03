"""
Run the API server or CLI for local PDF extraction.

Usage:
  API server:  python main.py [--reload]
  CLI extract: python main.py extract <path-to-resume.pdf> [--output out.json]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from pdf_reader import extract_text_from_pdf
from resume_extract import extract_resume_with_raw

ARTIFACTS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = ARTIFACTS_DIR / "result"


def _slugify_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    safe = safe.strip("._-")
    return safe or "resume"


def _build_result_paths(pdf_path: Path) -> tuple[Path, Path, Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = _slugify_filename(pdf_path.stem)
    input_path = run_dir / f"{stem}_input-pdf.txt"
    raw_path = run_dir / f"{stem}_raw-result.json"
    result_path = run_dir / f"{stem}_result.json"
    return input_path, raw_path, result_path


def cmd_extract(pdf_path: str, output_path: str | None) -> int:
    """Extract resume JSON from a PDF file and print or write to file."""
    path = Path(pdf_path)
    try:
        text = extract_text_from_pdf(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    input_path, raw_path, result_path = _build_result_paths(path)
    input_path.write_text(text or "", encoding="utf-8")

    try:
        result, raw_result = extract_resume_with_raw(text)
    except Exception as e:
        print(f"Extraction error: {e}", file=sys.stderr)
        return 1

    raw_path.write_text(
        json.dumps(raw_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    data = result.model_dump(mode="json")
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    result_path.write_text(json_str, encoding="utf-8")

    if output_path:
        Path(output_path).write_text(json_str, encoding="utf-8")
        print(f"Wrote {output_path}")
    else:
        print(json_str)
    return 0


def cmd_serve(reload: bool = False) -> None:
    """Run the FastAPI app with uvicorn."""
    import os

    os.environ["RELOAD"] = "1" if reload else "0"
    from api import run_app

    run_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="ATS PDF Reader: API server or CLI extract")
    sub = parser.add_subparsers(dest="command", help="Command")

    serve_p = sub.add_parser("serve", help="Run the API server (default)")
    serve_p.add_argument("--reload", action="store_true", help="Enable uvicorn reload")
    serve_p.set_defaults(cmd="serve")

    extract_p = sub.add_parser("extract", help="Extract resume JSON from a PDF file")
    extract_p.add_argument("pdf_path", help="Path to resume PDF")
    extract_p.add_argument("--output", "-o", help="Write JSON to file (default: stdout)")
    extract_p.set_defaults(cmd="extract")

    args = parser.parse_args()

    if args.command == "extract":
        return cmd_extract(args.pdf_path, getattr(args, "output", None))
    if args.command == "serve" or args.command is None:
        cmd_serve(reload=getattr(args, "reload", False))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
