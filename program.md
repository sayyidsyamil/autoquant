# autoquant

This is an experiment to have an LLM do its own quant research.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date, for example `may7`.
2. Create a branch named `autoquant/<tag>` from `master`.
3. Read the in-scope files:
   - `README.md` - repository context.
   - `prepare.py` - fixed yfinance data and evaluation harness. Do not modify during normal experiments.
   - `train.py` - strategy logic. This is the main file to modify.
4. Verify data exists with `uv run prepare.py`.
5. Initialize `results.tsv` with this header:

```text
commit	val_sharpe	max_drawdown	status	description
```

## Experimentation

Each experiment runs locally on CPU and downloads/caches Bursa Malaysia daily
prices from yfinance. The metric is `val_sharpe`, an out-of-sample annualized
Sharpe ratio. Higher is better.

Run an experiment with:

```bash
uv run train.py > run.log 2>&1
```

What you can do:

- Modify `train.py`: ranking features, portfolio construction, risk penalties, search, rebalancing, and hyperparameters.
- Add small helper functions inside `train.py` if they make the strategy clearer.

What you should usually avoid:

- Modifying `prepare.py`; it defines the ground-truth evaluation split and Sharpe calculation.
- Overfitting to the validation window by repeatedly encoding validation-specific choices.
- Adding heavyweight dependencies when pandas/numpy/yfinance are enough.

## Output Format

The script prints a summary like:

```text
---
val_sharpe:        1.234567
train_sharpe:      1.987654
annual_return:     0.120000
annual_volatility: 0.090000
max_drawdown:      -0.080000
cumulative_return: 0.150000
num_assets:        8
search_trials:     1200
validation_window: 2025-01-02..2026-05-06
total_seconds:     21.4
```

Extract the main fields with:

```bash
grep "^val_sharpe:\\|^max_drawdown:" run.log
```

## Logging Results

Append one row per run to `results.tsv`:

```text
commit	val_sharpe	max_drawdown	status	description
abc1234	1.234567	-0.080000	keep	baseline inverse-vol top momentum basket
```

Use `keep` when `val_sharpe` improves with acceptable drawdown and complexity.
Use `discard` when it gets worse or the complexity cost is not worth it.
Use `crash` when the run fails.

## Simplicity Criterion

All else equal, simpler is better. A tiny Sharpe gain from a fragile, opaque
rule is less valuable than a modest, robust improvement from a clear risk or
ranking change. Prefer changes that improve out-of-sample Sharpe while keeping
drawdown and concentration understandable.
