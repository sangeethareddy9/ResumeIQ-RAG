"""
src/test_parser.py

Quick manual test: run the parser against the sample resumes and print
what got extracted. Not a formal test suite yet -- just a sanity check
that PDF and DOCX parsing both actually work before we build anything
on top of them.

Run from the project root with:  python -m src.test_parser
"""

from pathlib import Path

from src.parser import parse_document
from src.preprocess import clean_text

SAMPLE_DIR = Path("data/sample_resumes")


def run():
    files = list(SAMPLE_DIR.glob("*.pdf")) + list(SAMPLE_DIR.glob("*.docx"))

    if not files:
        print(f"No files found in {SAMPLE_DIR.resolve()}")
        return

    for file_path in files:
        file_bytes = file_path.read_bytes()
        result = parse_document(file_bytes, file_path.name)

        print("=" * 60)
        print(f"File:      {result.filename}")
        print(f"Type:      {result.file_type}")
        print(f"Status:    {result.status}")
        print(f"Usable:    {result.is_usable}")
        print(f"Chars:     {result.char_count}")

        if result.error_message:
            print(f"Error:     {result.error_message}")

        if result.raw_text:
            preview = result.raw_text[:200].replace("\n", " | ")
            print(f"Preview:   {preview}...")

        print()


if __name__ == "__main__":
    run()