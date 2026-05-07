# autoquant

AutoQuant is a fork of `karpathy/autoresearch` adapted from autonomous LLM
pretraining experiments into autonomous quant research.

The original repo optimized validation bits per byte (`val_bpb`) after a fixed
training budget. This fork swaps the domain and metric:

- Data source: `yfinance`
- Market: Bursa Malaysia stocks (`.KL` tickers)
- Score: out-of-sample annualized Sharpe ratio (`val_sharpe`)
- Direction: higher is better

This is research code, not financial advice.

## How it works

The project keeps the small three-file shape:

- `prepare.py` - fixed data download, cache, returns, and evaluation utilities.
- `train.py` - the editable strategy experiment. Change ranking, weighting, and search logic here.
- `program.md` - operating instructions for an autonomous agent running experiments.

By default, `prepare.py` caches adjusted close data under `~/.cache/autoquant/`.
The validation split starts after `2024-12-31`, so experiments optimize on older
data and report out-of-sample Sharpe on newer data.

## Quick Start

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prepare.py
uv run train.py
```

Expected output ends with a summary like:

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

## Experiment Ideas

- Change the default Bursa Malaysia universe in `prepare.py`.
- Improve `train.py` ranking features: momentum windows, volatility filters, dividend proxies, trend filters.
- Add walk-forward rebalancing instead of one static validation portfolio.
- Penalize concentration, drawdown, turnover, or sector crowding.
- Compare long-only equal weight, inverse volatility, minimum variance, and random-search portfolios.

## Logging Results

Use `results.tsv` for local experiment notes. It is ignored by git.

```text
commit	val_sharpe	max_drawdown	status	description
abc1234	1.234567	-0.080000	keep	baseline inverse-vol top momentum basket
```

## License

MIT, inherited from the upstream project.
