import re
import sys
from pathlib import Path


def clean_srt(input_path: str) -> str:
    """Remove subtitle numbers and timestamps, returning only the spoken text."""
    text = Path(input_path).read_text(encoding="utf-8-sig")

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", text.strip())

    lines = []
    for block in blocks:
        block_lines = block.strip().splitlines()
        # Filter out the subtitle number (a line with only digits)
        # and the timestamp line (contains -->)
        content = [
            line.strip()
            for line in block_lines
            if not re.fullmatch(r"\d+", line.strip())
            and "-->" not in line
            and line.strip()
        ]
        lines.extend(content)

    return " ".join(lines)


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "2. High-Ticket Client Demonstration.srt"
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: file '{input_file}' not found.")
        sys.exit(1)

    cleaned = clean_srt(str(input_path))

    output_path = input_path.with_suffix(".txt")
    output_path.write_text(cleaned, encoding="utf-8")
    print(f"Saved cleaned text to: {output_path}")


if __name__ == "__main__":
    main()
