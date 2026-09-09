#!/usr/bin/env python3
"""Validate and normalize the Edu-Eval JSONL release."""

import argparse
import json
from pathlib import Path


REQUIRED_METADATA = {
    "科目",
    "学段1",
    "学段2",
    "年级",
    "一级目录",
    "二级目录",
    "内容",
}


def load_jsonl(path: Path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def normalize(rows, expected_count: int):
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")

    prompts = set()
    normalized = []
    for index, row in enumerate(rows, 1):
        prompt = row.get("generated_prompt")
        metadata = row.get("original_data")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row {index} has no generated_prompt")
        if prompt in prompts:
            raise ValueError(f"Duplicate generated_prompt at row {index}")
        if not isinstance(metadata, dict) or not REQUIRED_METADATA.issubset(metadata):
            missing = REQUIRED_METADATA - set(metadata or {})
            raise ValueError(f"Row {index} has incomplete original_data: {sorted(missing)}")
        prompts.add(prompt)
        normalized.append(
            {
                "id": f"edu-eval-{index:04d}",
                "status": "success",
                "original_data": {key: metadata[key] for key in metadata},
                "generated_prompt": prompt.strip(),
            }
        )
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=226)
    args = parser.parse_args()

    rows = normalize(load_jsonl(args.input), args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
