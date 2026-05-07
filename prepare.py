"""
Fixed data and evaluation utilities for AutoQuant experiments.

AutoQuant uses yfinance daily price data for Bursa Malaysia stocks and scores
experiments by out-of-sample annualized Sharpe ratio. Higher val_sharpe is better.

Usage:
    uv run prepare.py                 # download/cache the default universe
    uv run prepare.py --refresh       # force a fresh yfinance download
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants (fixed evaluation harness)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoquant"
PRICE_CACHE = CACHE_DIR / "malaysia_prices.csv"

START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()
TRAIN_END_DATE = "2024-12-31"
RISK_FREE_RATE = 0.03
TRADING_DAYS = 252
MIN_OBSERVATIONS = 252

DEFAULT_TICKERS = [
    "1155.KL",  # Malayan Banking
    "1023.KL",  # CIMB Group
    "1295.KL",  # Public Bank
    "5819.KL",  # Hong Leong Bank
    "1066.KL",  # RHB Bank
    "5347.KL",  # Tenaga Nasional
    "4863.KL",  # Telekom Malaysia
    "6947.KL",  # CelcomDigi
    "5225.KL",  # IHH Healthcare
    "5183.KL",  # Petronas Chemicals
    "6033.KL",  # Petronas Gas
    "5681.KL",  # Petronas Dagangan
    "3816.KL",  # MISC
    "7277.KL",  # Dialog Group
    "1961.KL",  # IOI Corp
    "2445.KL",  # Kuala Lumpur Kepong
    "5285.KL",  # Sime Darby Plantation
    "4197.KL",  # Sime Darby
    "4707.KL",  # Nestle Malaysia
    "5296.KL",  # MR DIY
    "7113.KL",  # Top Glove
    "5168.KL",  # Hartalega
]


@dataclass(frozen=True)
class PortfolioReport:
    val_sharpe: float
    train_sharpe: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    cumulative_return: float
    num_assets: int
    start: str
    end: str


def download_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted daily close prices and cache them as CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if PRICE_CACHE.exists() and not refresh:
        return load_prices()

    symbols = sorted(set(tickers))
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no price data for the configured universe")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(prices, pd.DataFrame):
        prices = prices.to_frame()
    if len(symbols) == 1:
        prices.columns = symbols

    prices = clean_prices(prices)
    if prices.empty:
        raise RuntimeError("no usable price series remained after cleaning")

    prices.to_csv(PRICE_CACHE, index_label="Date")
    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep liquid-enough columns and forward-fill occasional market holidays."""
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.dropna(axis=1, thresh=MIN_OBSERVATIONS)
    prices = prices.ffill(limit=5).dropna(axis=0, how="all")
    prices = prices.dropna(axis=1)
    prices = prices.loc[:, prices.nunique() > 1]
    return prices


def load_prices() -> pd.DataFrame:
    if not PRICE_CACHE.exists():
        return download_prices(refresh=True)
    prices = pd.read_csv(PRICE_CACHE, index_col="Date", parse_dates=True)
    return clean_prices(prices)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns.dropna(axis=0, how="all").dropna(axis=1)


def train_val_split(returns: pd.DataFrame, train_end: str = TRAIN_END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = returns.loc[:train_end]
    val = returns.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1):]
    if len(train) < MIN_OBSERVATIONS:
        raise RuntimeError(f"train split too small: {len(train)} rows")
    if len(val) < 30:
        raise RuntimeError(f"validation split too small: {len(val)} rows")
    common = train.columns.intersection(val.columns)
    return train[common].dropna(axis=1), val[common].dropna(axis=1)


def normalize_weights(weights: pd.Series | dict[str, float], columns: Iterable[str]) -> pd.Series:
    weights = pd.Series(weights, dtype=float).reindex(list(columns)).fillna(0.0)
    weights = weights.clip(lower=0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("portfolio weights must contain at least one positive finite value")
    return weights / total


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | dict[str, float]) -> pd.Series:
    w = normalize_weights(weights, returns.columns)
    aligned = returns[w.index].dropna(axis=0, how="any")
    return aligned @ w


def annualized_sharpe(series: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    series = series.dropna()
    if len(series) < 2:
        return float("nan")
    daily_excess = series - risk_free_rate / TRADING_DAYS
    vol = float(daily_excess.std(ddof=1))
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * daily_excess.mean() / vol)


def max_drawdown(series: pd.Series) -> float:
    equity = (1.0 + series.dropna()).cumprod()
    if equity.empty:
        return float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def evaluate_portfolio(
    weights: pd.Series | dict[str, float],
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
) -> PortfolioReport:
    train_p = portfolio_returns(train_returns, weights)
    val_p = portfolio_returns(val_returns, weights)
    return PortfolioReport(
        val_sharpe=annualized_sharpe(val_p),
        train_sharpe=annualized_sharpe(train_p),
        annual_return=float(val_p.mean() * TRADING_DAYS),
        annual_volatility=float(val_p.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        max_drawdown=max_drawdown(val_p),
        cumulative_return=float((1.0 + val_p).prod() - 1.0),
        num_assets=int((normalize_weights(weights, val_returns.columns) > 0).sum()),
        start=val_p.index.min().date().isoformat(),
        end=val_p.index.max().date().isoformat(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="force a fresh yfinance download")
    args = parser.parse_args()

    prices = download_prices(refresh=args.refresh)
    returns = daily_returns(prices)
    train, val = train_val_split(returns)
    print(f"AutoQuant cache: {PRICE_CACHE}")
    print(f"Tickers ready: {len(prices.columns)}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(val)}")


if __name__ == "__main__":
    main()
