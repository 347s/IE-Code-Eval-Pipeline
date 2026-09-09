# Pre-release checklist

- [ ] Rotate the credentials found in the private archive before publication.
- [ ] Confirm the repository owner and replace the placeholder copyright holder.
- [ ] Confirm that the benchmark prompts and generated HTML may be redistributed.
- [ ] Confirm licenses for any bundled fonts, images, JavaScript, and CDN assets.
- [x] Include all 226 normalized prompts in `data/edu_eval.jsonl`.
- [ ] Document the exact judge model checkpoint, serving framework, and revision.
- [ ] Document generation parameters and random seeds for every evaluated model.
- [ ] Reconcile the manuscript scoring formula with the archived 0.20/0.25/0.25/0.30 profile.
- [ ] Export sanitized summary CSV files with relative paths and no internal hostnames.
- [ ] Run secret scanning and license scanning on the final Git history.
- [ ] Run a clean-environment smoke test on Windows and Linux.
