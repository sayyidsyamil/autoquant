"""
AutoQuant strategy experiment.

The editable surface is intentionally compact: change the ranking, weighting,
and search logic below, then run `uv run train.py`. The score is val_sharpe,
an out-of-sample annualized Sharpe ratio on Malaysia stocks from yfinance.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd

from prepare import (
    RISK_FREE_RATE,
    TRADING_DAYS,
    daily_returns,
    download_prices,
    evaluate_portfolio,
    train_val_split,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

SEED = 42
SEARCH_SECONDS = 20
LOOKBACK_DAYS = 252
MIN_ASSETS = 4
MAX_ASSETS = 10
CORRELATION_PENALTY = 0.20
MOMENTUM_WEIGHT = 0.70
SHARPE_WEIGHT = 0.30
MAX_SINGLE_NAME_WEIGHT = 0.30


def trailing_metrics(train_returns: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Rank stocks by a blend of recent momentum and trailing Sharpe."""
    window = train_returns.tail(lookback_days)
    excess = window - RISK_FREE_RATE / TRADING_DAYS
    momentum = (1.0 + window).prod() - 1.0
    volatility = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (excess.mean() * TRADING_DAYS) / volatility.replace(0, np.nan)
    score = MOMENTUM_WEIGHT * momentum.rank(pct=True) + SHARPE_WEIGHT * sharpe.rank(pct=True)
    return pd.DataFrame(
        {
            "momentum": momentum,
            "volatility": volatility,
            "sharpe": sharpe,
            "score": score,
        }
    ).dropna().sort_values("score", ascending=False)


def inverse_vol_weights(train_returns: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Build long-only inverse-volatility weights with a single-name cap."""
    vol = train_returns[tickers].tail(LOOKBACK_DAYS).std(ddof=1)
    raw = (1.0 / vol.replace(0, np.nan)).dropna()
    weights = raw / raw.sum()
    for _ in range(10):
        overweight = weights > MAX_SINGLE_NAME_WEIGHT
        if not overweight.any():
            break
        weights[overweight] = MAX_SINGLE_NAME_WEIGHT
        remainder = 1.0 - weights[overweight].sum()
        under = weights[~overweight]
        if under.empty:
            break
        weights[~overweight] = remainder * under / under.sum()
    return weights / weights.sum()


def diversification_score(train_returns: pd.DataFrame, tickers: list[str]) -> float:
    """Prefer baskets whose average pairwise correlation is lower."""
    if len(tickers) < 2:
        return 0.0
    corr = train_returns[tickers].tail(LOOKBACK_DAYS).corr().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(upper))


def candidate_from_top_ranked(train_returns: pd.DataFrame, n_assets: int) -> pd.Series:
    metrics = trailing_metrics(train_returns)
    shortlist = metrics.head(max(MAX_ASSETS * 2, n_assets)).index.tolist()
    picked: list[str] = []
    for ticker in shortlist:
        trial = picked + [ticker]
        if len(trial) == 1:
            picked.append(ticker)
            continue
        corr_penalty = diversification_score(train_returns, trial)
        adjusted = float(metrics.loc[ticker, "score"]) - CORRELATION_PENALTY * corr_penalty
        if adjusted > 0:
            picked.append(ticker)
        if len(picked) == n_assets:
            break
    if len(picked) < n_assets:
        picked = shortlist[:n_assets]
    return inverse_vol_weights(train_returns, picked)


def random_candidate(train_returns: pd.DataFrame, rng: random.Random) -> pd.Series:
    metrics = trailing_metrics(train_returns)
    pool = metrics.head(min(len(metrics), MAX_ASSETS * 2)).index.tolist()
    n_assets = rng.randint(MIN_ASSETS, min(MAX_ASSETS, len(pool)))
    sampled = rng.sample(pool, n_assets)
    weights = inverse_vol_weights(train_returns, sampled)
    jitter = pd.Series((rng.lognormvariate(0, 0.35) for _ in sampled), index=sampled)
    weights = weights * jitter
    return weights / weights.sum()


def train_score(weights: pd.Series, train_returns: pd.DataFrame) -> float:
    """Use train Sharpe minus drawdown/correlation penalties to avoid obvious overfit."""
    report = evaluate_portfolio(weights, train_returns, train_returns)
    corr = diversification_score(train_returns, weights[weights > 0].index.tolist())
    return report.train_sharpe + report.max_drawdown - CORRELATION_PENALTY * corr


def search_portfolio(train_returns: pd.DataFrame) -> tuple[pd.Series, float, int]:
    rng = random.Random(SEED)
    best_weights: pd.Series | None = None
    best_score = float("-inf")
    trials = 0
    deadline = time.time() + SEARCH_SECONDS

    deterministic_ns = range(MIN_ASSETS, min(MAX_ASSETS, len(train_returns.columns)) + 1)
    for n_assets in deterministic_ns:
        weights = candidate_from_top_ranked(train_returns, n_assets)
        score = train_score(weights, train_returns)
        trials += 1
        if score > best_score:
            best_score, best_weights = score, weights

    while time.time() < deadline:
        weights = random_candidate(train_returns, rng)
        score = train_score(weights, train_returns)
        trials += 1
        if score > best_score:
            best_score, best_weights = score, weights

    assert best_weights is not None
    return best_weights.sort_values(ascending=False), best_score, trials


def main() -> None:
    t_start = time.time()
    random.seed(SEED)
    np.random.seed(SEED)

    prices = download_prices()
    returns = daily_returns(prices)
    train_returns, val_returns = train_val_split(returns)

    weights, objective, trials = search_portfolio(train_returns)
    report = evaluate_portfolio(weights, train_returns, val_returns)

    print("Portfolio weights:")
    for ticker, weight in weights.items():
        if weight > 0:
            print(f"  {ticker:8s} {weight:7.2%}")

    print("---")
    print(f"val_sharpe:        {report.val_sharpe:.6f}")
    print(f"train_sharpe:      {report.train_sharpe:.6f}")
    print(f"annual_return:     {report.annual_return:.6f}")
    print(f"annual_volatility: {report.annual_volatility:.6f}")
    print(f"max_drawdown:      {report.max_drawdown:.6f}")
    print(f"cumulative_return: {report.cumulative_return:.6f}")
    print(f"num_assets:        {report.num_assets}")
    print(f"search_trials:     {trials}")
    print(f"train_objective:   {objective:.6f}")
    print(f"validation_window: {report.start}..{report.end}")
    print(f"total_seconds:     {time.time() - t_start:.1f}")


if __name__ == "__main__":
    main()
