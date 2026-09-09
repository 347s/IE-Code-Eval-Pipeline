# Edu-Eval

`edu_eval.jsonl` contains 226 prompts for evaluating generation of interactive,
self-contained educational HTML pages across Chinese K–12 curricula.

## Official release and citation

Edu-Eval was first released through
[Science Data Bank](https://www.scidb.cn/s/nA3aIb) in accordance with the data
publication requirements of *Acta Automatica Sinica*. The JSONL file in this
repository is a convenience copy for reproducible evaluation. Cite the associated
paper through the repository's `CITATION.cff` and include the Science Data Bank
landing page when using the benchmark.

The associated model is available as
[Septend/IE-code-30B](https://huggingface.co/Septend/IE-code-30B) on Hugging Face.

## Coverage

| Dimension | Counts |
|---|---|
| Subject | Mathematics 78; Chinese 44; English 52; Physics 52 |
| School stage | Primary 78; middle school 96; high school 52 |
| Total | 226 |

The paper additionally reports 9 observed subject–stage combinations, 103 leaf
subject–stage–knowledge-point categories, and 90 knowledge-point names after
deduplication within subjects.

## Schema

Each UTF-8 JSONL row contains:

- `id`: stable public ID (`edu-eval-0001` through `edu-eval-0226`).
- `status`: retained compatibility field; all released rows are `success`.
- `original_data`: subject, school stage, curriculum stage, grades, two-level
  curriculum category, and curriculum description in Chinese.
- `generated_prompt`: the generation instruction accepted by `app.py`.

No model response, score, screenshot, personal information, or API configuration
is included. Dataset order is fixed because the archived evaluator used row index
as the task identifier.

## Intended use

The dataset is intended for research evaluation of HTML generation, rendering,
interaction, and educational-content alignment. LLM/VLM judging is stochastic;
publish the judge checkpoint, endpoint implementation, generation parameters,
weights, and random seeds with reported results.

The prompts request executable JavaScript. Evaluate generated pages only in a
controlled browser environment without personal accounts or sensitive files.

## Provenance and license

The tasks were organized from K–12 curriculum metadata and transformed into
interactive HTML generation instructions for the Edu-Eval benchmark. The code is
MIT licensed. A software license does not automatically cover the dataset; consult
the official Science Data Bank record for the authoritative release information
and applicable data terms.

Before making the GitHub repository public, the repository owner must confirm the
right to redistribute the curriculum-derived fields and select a dataset license.
Record that decision in `data/LICENSE.md`; until then, the included file is a
release candidate for owner review rather than a representation that third-party
reuse rights have already been granted.
