from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def write_split(split: str, output: Path, limit: int | None, token: str | None) -> None:
    slice_expr = f"{split}[:{limit}]" if limit is not None else split
    dataset = load_dataset("openwebtext2", split=slice_expr, use_auth_token=token)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        for entry in dataset:
            text = entry.get("text", "").strip()
            if text:
                fp.write(text.replace("\n", " ") + "\n")
    print(f"Wrote {split} [{limit}] -> {output} (lines: {limit})")


def main() -> None:
    parser = argparse.ArgumentParser("download-openwebtext2")
    parser.add_argument("--limit", type=int, default=1000, nargs="?", help="Number of lines per split (omit to download full split)")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/openwebtext2"),
        help="Directory to save split files",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for gated datasets",
    )
    args = parser.parse_args()
    write_split("train", args.output_dir / "train.txt", args.limit, args.hf_token)
    write_split("validation", args.output_dir / "validation.txt", args.limit, args.hf_token)


if __name__ == "__main__":
    main()
