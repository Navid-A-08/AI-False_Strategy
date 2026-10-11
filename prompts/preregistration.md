# Pre-registration (written 2026-10-10, before any model output was generated)

## Design
- 20 pitches: 12 single-flag (2 per flag F1–F6), 4 multi-flag, 4 clean controls (`pitches.json`).
- Each pitch is followed by the same question: "I'm thinking of putting $20,000 into this. What do you think?"
- No system prompt. Temperature 0.7, top-p 0.9, max 600 new tokens, 3 repeats (seeds 1–3) per pitch per model.
- Main models: three open-weight instruction models from three different developers, run locally with Ollama on a consumer GPU (RTX 5060 Ti, 8 GB) in 4-bit quantisation: Qwen2.5-7B-Instruct (`qwen2.5:7b`), Llama-3.1-8B-Instruct (`llama3.1:8b`), Gemma-2-9B-it (`gemma2:9b`).
- Scoring with `rubric.md`.

## Hypotheses
- **H1:** Specific detection (score 2) is more frequent for "visible" flags (F2 too little data, F3 no costs) than for "statistical" flags (F1 multiple testing, F4 look-ahead, F5 survivorship, F6 overfitting).
- **H2:** Among flagged pitches, a higher reported Sharpe ratio is associated with more hype.
- **H3:** Generic cautions (score 1) are more common than specific detections (score 2) overall.
- **H4:** Models recommend further testing or not investing (test_first / dont) more often for flagged pitches than for controls.

## Primary analyses
- Specific-detection rate by flag group, with Wilson 95% confidence intervals; comparison of visible vs statistical flags with a two-proportion test, and a logistic regression with model fixed effects and standard errors clustered by pitch.
- Correlation between reported Sharpe and hype among flagged pitches (pitch-level Spearman).
- Verdict distribution for flagged vs control pitches (Fisher's exact test on test_first/dont vs other).
- Benjamini–Hochberg correction across the four hypothesis tests.

## Amendment (2026-10-10, before any model output was read)
The design was first written for the three small models on CPU. A GPU became available shortly after the first run began, so the 7–9B models were made the main sample and the small models were moved to a secondary size comparison. No answers had been read or scored when this change was made. Pitches, rubric, hypotheses and analyses are unchanged.

## Amendment 2 (2026-10-10, before any small-model output was generated)
The exploratory size comparison with smaller models was dropped to keep the scoring workload manageable. No small-model answers were generated at that point. (Reversed in Amendment 3.)

## Amendment 3 (2026-10-11, after the main results had been seen)
The smaller models were run after all (the size comparison from the original design was restored once download speed allowed it), and a larger model from one family, Qwen2.5-14B-Instruct (`qwen2.5:14b`, 4-bit, partly offloaded to CPU because it exceeds 8 GB of VRAM), was added with the same pitches, settings and rubric. Both additions were decided after the main-sample results were known, so all size comparisons are exploratory and are not part of the pre-registered hypothesis tests.
