#!/usr/bin/env python3
"""Offline validation for the public repository and Edu-Eval data."""

import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "edu_eval.jsonl"
FORBIDDEN = [
    re.compile(pattern, re.I)
    for pattern in (
        r"openapi-qb",
        r"ai-notebook-inspire",
        r"/inspire/hdd/",
        r"[A-Z]:\\Users\\",
        r"[A-Z]:\\Work\\",
        r"\bsk-[A-Za-z0-9_-]{16,}\b",
    )
]


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


for path in ROOT.glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in (ROOT / "scripts").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 226:
    fail(f"expected 226 dataset rows, found {len(rows)}")
if len({row["id"] for row in rows}) != len(rows):
    fail("dataset IDs are not unique")
if len({row["generated_prompt"] for row in rows}) != len(rows):
    fail("dataset prompts are not unique")

subjects = Counter(row["original_data"]["科目"] for row in rows)
stages = Counter(row["original_data"]["学段1"] for row in rows)
if subjects != {"数学": 78, "语文": 44, "英语": 52, "物理": 52}:
    fail(f"unexpected subject distribution: {subjects}")
if stages != {"小学": 78, "初中": 96, "高中": 52}:
    fail(f"unexpected stage distribution: {stages}")

scan_suffixes = {".py", ".json", ".jsonl", ".md", ".txt", ".toml", ".yml", ".yaml", ".sh", ".bat"}
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in scan_suffixes:
        continue
    if any(part in {"evaluation_outputs", "__pycache__"} for part in path.parts):
        continue
    if path.resolve() == Path(__file__).resolve():
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in FORBIDDEN:
        if pattern.search(text):
            fail(f"private value pattern {pattern.pattern!r} found in {path.relative_to(ROOT)}")

digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
print("Release validation passed")
print(f"dataset_rows={len(rows)}")
print(f"dataset_sha256={digest}")
print(f"subjects={dict(subjects)}")
print(f"stages={dict(stages)}")
