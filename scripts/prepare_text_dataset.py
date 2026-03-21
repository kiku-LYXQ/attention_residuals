from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoTokenizer


def encode_lines(tokenizer, lines: Iterable[str]) -> torch.Tensor:
    text = "\n".join(lines).strip()
    if not text:
        return torch.empty(0, dtype=torch.long)
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return enc["input_ids"].flatten()


def prepare_tokens(tokenizer_name: str, text_file: Path, output_path: Path, batch_lines: int = 256) -> None:
    if not text_file.exists():
        raise FileNotFoundError(f"text_file {text_file} not found")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_chunks: List[torch.Tensor] = []
    buffer: List[str] = []
    with text_file.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            buffer.append(line.strip())
            if len(buffer) >= batch_lines:
                ids = encode_lines(tokenizer, buffer)
                if ids.numel():
                    token_chunks.append(ids)
                buffer.clear()
        if buffer:
            ids = encode_lines(tokenizer, buffer)
            if ids.numel():
                token_chunks.append(ids)
    if not token_chunks:
        raise ValueError("no tokens were produced from the text file")
    tokens = torch.cat(token_chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, output_path)


def main() -> None:
    parser = argparse.ArgumentParser("prepare-text-dataset")
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--text_file", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_lines", type=int, default=256)
    args = parser.parse_args()
    prepare_tokens(args.tokenizer_name, args.text_file, args.output_path, args.batch_lines)


if __name__ == "__main__":
    main()
