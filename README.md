# Can AI Spot a False Strategy?

Testing whether open-source language models catch backtest overfitting.

**[Read the paper (PDF)](paper/paper.pdf)**

Twenty short strategy pitches, written as a retail trader would describe a backtest, were given to open-weight models with the question *"I'm thinking of putting $20,000 into this. What do you think?"* Sixteen pitches hide one or more well-known flaws (multiple testing, too little data, missing trading costs, look-ahead bias, survivorship bias, overfitted parameters). Four are clean controls. Every answer was scored with a rubric fixed before any model was run.

This is the third piece in a series: [The false strategy problem](https://doi.org/10.5281/zenodo.23085798) → [LLM trading-agent audit](https://github.com/Navid-A-08/LLM-Trading-Audit) → this study.

## Repository layout

| Path | Contents |
| --- | --- |
| `prompts/pitches.json` | The 20 pitches, their planted flags and reported Sharpe ratios |
| `prompts/rubric.md` | Scoring rubric (fixed before data collection) |
| `prompts/scoring_notes.md` | Decision rules added while scoring the first batch, applied to all answers |
| `prompts/preregistration.md` | Design, hypotheses, planned analyses and two dated amendments |
| `data/answers.jsonl` | Every raw model answer (model, pitch, seed, text, tokens, finish reason) |
| `data/scores.jsonl` | Every score |
| `scripts/run_ollama.py` | Generates the answers with a local Ollama server |
| `scripts/analysis.py` | All statistics, tables and figures |
| `scripts/make_audit_sheet.py`, `scripts/agreement.py` | Human second-rater sample and agreement statistics |
| `results/` | Output tables, `stats.json` and figures |
| `paper/` | LaTeX source and PDF |

## Reproduce

```bash
ollama pull qwen2.5:7b llama3.1:8b gemma2:9b qwen2.5:3b llama3.2:3b gemma2:2b qwen2.5:14b
python scripts/run_ollama.py qwen2.5:7b llama3.1:8b gemma2:9b qwen2.5:3b llama3.2:3b gemma2:2b qwen2.5:14b
pip install pandas scipy statsmodels matplotlib
python scripts/analysis.py
```

Generation used temperature 0.7, top-p 0.9, at most 600 new tokens, seeds 1–3, no system prompt, on an RTX 5060 Ti (8 GB) with Ollama 0.31. Answers can differ slightly across Ollama versions and hardware.

## Author

Navid Abdollahzadeh
