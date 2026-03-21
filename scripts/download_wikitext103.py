from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def write_split(split: str, output: Path, limit: int) -> None:
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=f"{split}[:{limit}]")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        for entry in dataset:
            text = entry.get("text", "").strip()
            if text:
                fp.write(text.replace("\n", " ") + "\n")
    print(f"Wrote {split} [{limit}] -> {output} (lines: {limit})")


def main() -> None:
    parser = argparse.ArgumentParser("download-wikitext103")
    parser.add_argument("--limit", type=int, default=2000, help="Number of lines per split")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/wikitext103"),
        help="Directory to save split files",
    )
    args = parser.parse_args()
    write_split("train", args.output_dir / "train.txt", args.limit)
    write_split("validation", args.output_dir / "validation.txt", args.limit)


if __name__ == "__main__":
    main()
