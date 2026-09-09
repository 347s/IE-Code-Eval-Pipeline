# Contributing

Please open an issue before substantial changes. Pull requests should explain
which evaluation layer changes, why it changes, and whether historical scores
remain comparable. Run these checks before submitting:

```sh
python scripts/validate_release.py
python -m unittest discover -s tests -v
```

Do not commit API keys, internal endpoints, generated HTML, screenshots, model
outputs, or reports. Changes to prompts, scoring weights, runtime gating, browser
actions, screenshot sampling, or judge-model settings must be documented because
they can change benchmark results.
