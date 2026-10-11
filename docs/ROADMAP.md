# Roadmap

## Stage 1: TinyStories baseline (ready, not yet run)
- 30M model, 8k tokenizer, 100M tokens. See [RUNBOOK_100M.md](RUNBOOK_100M.md).
- Purpose: validate the pipeline end to end on a real GPU and get a baseline loss and sample quality.
- Estimated about 11 minutes of training and roughly 0.10 to 0.30 EUR. This is an estimate until the run is done.

## Stage 2: broader multi-source model (planned, not started)
Decisions still open, to be settled with measurements:
- Tokenizer: retrain at 16k (maybe 32k) on a sample of the *training* mix only. This invalidates Stage 1 token files.
- Model size: about 60M parameters first; 125M only if the budget allows after a new GPU benchmark.
- Data sources (check each licence before publishing weights): Project Gutenberg (public-domain works),
  Wikipedia (CC BY-SA), FineWeb-Edu (ODC-By), possibly Cosmopedia. Keep TinyStories as a smaller part of the mix.
- Mix proportions: tune on small runs; report validation loss per source, not only an average.
- Cleaning: add fuzzy (MinHash) deduplication and decontamination; exact dedup alone is not enough for web text.
- Budget: total GPU spend stays under 8 EUR; run a new benchmark for the larger model before any long run.

## Gate between stages
Stage 2 starts only after the Stage 1 results are reviewed.
