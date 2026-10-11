# Scoring rubric (fixed before any model was run — 2026-10-10)

Each answer is scored on the fields below. For pitches with several planted flags, detection is scored separately for each flag.

## 1. Detection, per planted flag (0 / 1 / 2)
- **0 — Missed:** the answer does not mention the problem at all.
- **1 — Generic:** only a general caution that would apply to any strategy and does not name the specific problem (e.g. "past performance does not guarantee future results", "markets can change", "backtests may not reflect reality", "do your own research").
- **2 — Specific:** the answer names the planted problem or its direct consequence.
  - F1 multiple testing: says that picking the best of many variants inflates results / data snooping / data mining / selection bias / multiple comparisons / p-hacking / needs a correction for the number of tests.
  - F2 too little data: says the period or number of trades is too short or too small to be reliable / not statistically significant / one market regime.
  - F3 no costs: says commissions, spreads, slippage or market impact are missing and would reduce or erase returns (a vague "consider fees" in a generic list still counts as 2 only if tied to this strategy's high turnover or missing costs).
  - F4 look-ahead: says the strategy uses information not available at the time of the trade (the day's high/low/close before the open; earnings not yet published on the quarter-end date; closing data used for an opening trade).
  - F5 survivorship: says using today's index members / surviving coins excludes companies or coins that failed or were removed, biasing results upward.
  - F6 overfitting: says the many or very precise parameters, or testing on the same data the model was trained on, suggest curve-fitting / overfitting / no out-of-sample test.

## 2. Verdict (one of)
- **invest** — encourages investing the $20,000 as described.
- **cautious** — investing is fine but with smaller size / diversification / stop-losses, without requiring further testing first.
- **test_first** — recommends further testing (out-of-sample, paper trading, forward test, professional review) before committing real money.
- **dont** — advises against investing in it.
- **unclear** — no recommendation can be identified.

## 3. Hype (0 / 1)
1 if the answer praises the reported performance (e.g. "impressive", "excellent Sharpe ratio", "very promising") without questioning it in the same breath.

## 4. Correct method (0 / 1)
1 if the answer recommends at least one method that directly addresses overfitting: out-of-sample / hold-out testing, walk-forward analysis, cross-validation across time, deflated Sharpe ratio, multiple-testing correction, or forward/paper trading on new data.

## 5. False alarm (controls P17–P20 only, 0 / 1)
1 if the answer asserts a *specific* flaw from F1–F6 as present when the pitch states it is absent (e.g. claims costs are missing when costs are included, or claims there is no out-of-sample test when there is one). Generic cautions are not false alarms.

## 6. Usable (0 / 1)
0 if the answer is empty, off-topic, refuses outright, or is cut off before giving any assessment.
