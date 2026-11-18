"""Reproduce the Z strategies in Qureshi and Zaman (arXiv:2403.07998v1)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import linprog
from scipy.sparse import coo_array

from .selection import DEFAULT_PRICES, EXPECTED_SHA256, _design, adf_pvalues, sha256
from .trading import WINDOW, rolling_endpoint, summarize_original_style
from .costs import COST_MODEL, PairPositions, charge_costs, pair_weights

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/paper'
START, END = '2017-01-01', '2023-05-31'
BASELINE_PAIRS = 250
Z_THRESHOLD = 2.0
REFERENCE = {
    'MZ': (0.7487, 0.6634, 1.23, 1.12),
    'BZ': (0.3830, -0.1340, 0.48, -0.13),
}


def adf_one_lag(residuals):
    """OLS t-statistic for lagged level with intercept and one lagged difference."""
    x, y = _design(residuals, 1)
    x0, x1 = x[:, :, 0], x[:, :, 1]
    xx = np.sum(x0 * x0, axis=1)
    xy = np.sum(x0 * x1, axis=1)
    yy = np.sum(x1 * x1, axis=1)
    xd = np.sum(x0 * y, axis=1)
    yd = np.sum(x1 * y, axis=1)
    det = xx * yy - xy**2
    with np.errstate(divide='ignore', invalid='ignore'):
        b0 = (yy * xd - xy * yd) / det
        b1 = (xx * yd - xy * xd) / det
        error = y - b0[:, None] * x0 - b1[:, None] * x1
        variance = np.sum(error**2, axis=1) / (y.shape[1] - 3)
        t = b0 / np.sqrt(variance * yy / det)
    return np.where(det > 0, t, np.nan)


def score_edges(log_history):
    """Use one direction per unordered pair, with columns in ticker order."""
    centered = log_history - log_history.mean(axis=0)
    gram = centered.T @ centered
    x, y = np.triu_indices(log_history.shape[1], k=1)
    stats = np.full(len(x), np.nan)
    for start in range(0, len(x), 512):
        batch = slice(start, start + 512)
        i, j = x[batch], y[batch]
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = gram[i, j] / gram[i, i]
            residuals = (centered[:, j] - centered[:, i] * beta).T
        stats[batch] = adf_one_lag(residuals)
    valid = np.isfinite(stats)
    return x[valid], y[valid], stats[valid]


def maximum_weight_pairs(x, y, score, n_assets):
    """Maximize total edge score with at most one edge incident to each asset."""
    edges = np.arange(len(score))
    incidence = coo_array((np.ones(2 * len(edges)), (np.r_[x, y], np.tile(edges, 2))),
                          shape=(n_assets, len(edges))).tocsc()
    relaxation = linprog(-score, A_ub=incidence, b_ub=np.ones(n_assets), method='highs-ds')
    if not relaxation.success:
        raise RuntimeError(relaxation.message)

    def blossom(ids):
        # Exact integer representation of the computed floating-point scores.
        ratios = [float(score[i]).as_integer_ratio() for i in ids]
        scale = max((denominator for _, denominator in ratios), default=1)
        graph = nx.Graph()
        for i, (numerator, denominator) in zip(ids, ratios):
            graph.add_edge(int(x[i]), int(y[i]), edge=i, weight=numerator * (scale // denominator))
        return np.array(sorted(graph[a][b]['edge'] for a, b in nx.max_weight_matching(graph)), dtype=int)

    incumbent = blossom(edges[relaxation.x > 0])
    potential = np.maximum(-relaxation.ineqlin.marginals, 0.)
    violation = max(0., float(np.max(score - potential[x] - potential[y])))
    potential += (violation + 1e-10) / 2
    slack = potential[x] + potential[y] - score
    # Every matching leaves nonnegative unused node potential and edge slack.
    # An edge whose slack exceeds the total bound gap cannot improve the incumbent.
    gap = potential.sum() - score[incumbent].sum()
    keep = slack <= gap + 1e-8
    keep[incumbent] = True
    return blossom(edges[keep])


def daily_signals(logs, x, y):
    """Return close-t Z signals for every complete rolling regression window."""
    px, py = logs[:, x], logs[:, y]
    beta, endpoint = rolling_endpoint(px, py)
    beta, endpoint = beta[WINDOW - 1:], endpoint[WINDOW - 1:]
    x_windows = sliding_window_view(px, WINDOW, axis=0)
    y_windows = sliding_window_view(py, WINDOW, axis=0)
    # Remove the rolling intercept by centering the residuals within each window.
    residuals = y_windows - x_windows * beta[:, :, None]
    residuals -= residuals.mean(axis=2, keepdims=True)
    sigma = np.sqrt(np.sum(residuals**2, axis=2) / (WINDOW - 2))
    z = np.divide(endpoint, sigma, out=np.zeros_like(endpoint), where=sigma > 0)
    z = np.clip(z, -3, 3)
    return beta, {
        'Z': np.where(np.abs(z) >= Z_THRESHOLD, np.sign(z), 0.),
    }


def pair_contributions(beta, signal, x_return, y_return):
    """Dollar-hedged return numerator and portfolio weight for each pair."""
    active = np.isfinite(beta) & np.isfinite(signal) & (signal != 0)
    beta = np.where(active, beta, 0.)
    signal = np.where(active, signal, 0.)
    return signal * (beta * x_return - y_return) / (1 + np.abs(beta)), np.abs(signal)


def ratio_returns(numerator, weight):
    return np.divide(numerator, weight, out=np.zeros_like(weight), where=weight > 0)


def portfolio_returns(beta, signal, x_return, y_return):
    """Dollar-hedge each pair and normalize active pair weights by |signal|."""
    numerator, weights = pair_contributions(beta, signal, x_return, y_return)
    weight = weights.sum(axis=1)
    gross = ratio_returns(numerator.sum(axis=1), weight)
    return gross, weights > 0


def formation_graph(raw):
    eligible = np.flatnonzero(np.isfinite(raw).all(axis=0) & (raw > 0).all(axis=0))
    i, j, t = score_edges(np.log(raw[:, eligible]))
    return eligible[i], eligible[j], t


def run_month(prices):
    cutoff = prices.index[WINDOW - 1]
    raw = prices.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(raw)
    x, y, t = formation_graph(raw[:WINDOW])
    p = adf_pvalues(t)
    baseline = np.lexsort((y, x, t, p))[:BASELINE_PAIRS]
    if len(baseline) != BASELINE_PAIRS:
        raise ValueError(f'{cutoff}: too few eligible pairs')
    matching = maximum_weight_pairs(x, y, -t, raw.shape[1])
    marked = prices.ffill().to_numpy()
    asset_returns = marked[WINDOW:] / marked[WINDOW - 1:-1] - 1
    missing_mark = ~np.isfinite(raw[WINDOW:])
    dates = prices.index[WINDOW:]
    daily, selections, positions = [], [], {}
    for method, ids in [('M', matching), ('B', baseline)]:
        a, b = x[ids], y[ids]
        selections.append(pd.DataFrame({'cutoff': cutoff, 'method': method,
                                        'asset_x': prices.columns[a], 'asset_y': prices.columns[b],
                                        'adf_t': t[ids], 'adf_p': p[ids]}))
        beta, signals = daily_signals(logs[:-1], a, b)
        rx = np.nan_to_num(asset_returns[:, a], nan=0.)
        ry = np.nan_to_num(asset_returns[:, b], nan=0.)
        stale = missing_mark[:, a] | missing_mark[:, b]
        for kind, signal in signals.items():
            gross, active = portfolio_returns(beta, signal, rx, ry)
            wx, wy = pair_weights(beta, signal)
            positions[method + kind] = PairPositions(
                dates, tuple(zip(prices.columns[a], prices.columns[b])), wx, wy, rx, ry)
            daily.append(pd.DataFrame({'date': dates, 'cutoff': cutoff, 'strategy': method + kind,
                                       'gross_return': gross,
                                       'active_pairs': active.sum(axis=1),
                                       'pairs_with_missing_end_quote': (stale & active).sum(axis=1)}))
    return pd.concat(daily, ignore_index=True), pd.concat(selections, ignore_index=True), positions


def performance(daily, compare_paper=True):
    daily = daily[daily.strategy.isin(REFERENCE)]
    records = []
    for strategy, group in daily.groupby('strategy', sort=False):
        ref_gross, ref_net, ref_gross_sharpe, ref_net_sharpe = (
            REFERENCE[strategy] if compare_paper else (np.nan,) * 4)
        for basis, target_return, target_sharpe in [('gross', ref_gross, ref_gross_sharpe),
                                                  ('net', ref_net, ref_net_sharpe)]:
            r = group[f'{basis}_return']
            stats = summarize_original_style(r)
            wealth = np.r_[1., (1 + r).cumprod().to_numpy()]
            records.append({'strategy': strategy, 'basis': basis, 'days': len(r),
                            'cumulative_return': stats['cumulative_return'], 'sharpe': stats['sharpe'],
                            'sortino': stats['sortino'],
                            'annualized_return': stats['annualized_return'],
                            'min_daily_return': r.min(), 'max_daily_return': r.max(), 'skew': r.skew(),
                            'max_drawdown': np.min(wealth / np.maximum.accumulate(wealth) - 1),
                            'paper_cumulative_return': target_return, 'paper_sharpe': target_sharpe,
                            'cumulative_return_gap': stats['cumulative_return'] - target_return,
                            'sharpe_gap': stats['sharpe'] - target_sharpe})
    return pd.DataFrame(records)


def add_input_arguments(parser):
    parser.add_argument('--prices', type=Path, help='Alternative long-form adjusted-price CSV or wide Parquet')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results')


def load_prices(path):
    path = DEFAULT_PRICES if path is None else Path(path).resolve()
    price_hash = sha256(path)
    if path == DEFAULT_PRICES and price_hash != EXPECTED_SHA256:
        raise ValueError('Unexpected original price input')
    if path.suffix.lower() == '.csv':
        prices = pd.read_csv(path, usecols=['date', 'ticker', 'adjusted_price'],
                             dtype={'ticker': str, 'adjusted_price': float}, parse_dates=['date'])
        if prices[['date', 'ticker']].isna().any().any():
            raise ValueError('Each price row needs a date and ticker')
        prices = prices.pivot(index='date', columns='ticker', values='adjusted_price')
    else:
        prices = pd.read_parquet(path)
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or not prices.index.is_unique):
        raise ValueError('Prices require a unique daily datetime index')
    if prices.columns.hasnans:
        raise ValueError('Prices require a ticker for every column')
    prices.columns = prices.columns.map(str)
    if not prices.columns.is_unique:
        raise ValueError('Prices require a unique column for each ticker')
    return prices.sort_index().sort_index(axis=1), price_hash


def trading_sample(prices, lookback):
    sample = prices.loc[START:END]
    if sample.empty:
        raise ValueError(f'No trading dates between {START} and {END}')
    history = prices.index.get_loc(sample.index[0])
    if history < lookback:
        raise ValueError(f'{sample.index[0]:%Y-%m-%d}: {history} prior trading days, '
                         f'{lookback} required')
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    add_input_arguments(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    prices, price_hash = load_prices(args.prices)
    sample = trading_sample(prices, WINDOW)
    output = args.output_root / 'paper'
    months = sample.groupby(sample.index.to_period('M'))
    daily, selections, cost_states = [], [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for _, month in months:
            first = prices.index.get_loc(month.index[0])
            last = prices.index.get_loc(month.index[-1])
            jobs.append(pool.submit(run_month, prices.iloc[first - WINDOW:last + 1]))
        for n, job in enumerate(jobs, 1):
            rows, pairs, positions = job.result()
            rows = charge_costs(rows, positions, cost_states, liquidate=n == len(jobs))
            daily.append(rows)
            selections.append(pairs)
            print(f'{n}/{len(jobs)} {rows.date.iloc[0]:%Y-%m} M={pairs.method.eq("M").sum()} '
                  f'B={pairs.method.eq("B").sum()}', flush=True)
    daily = pd.concat(daily, ignore_index=True)
    for strategy in REFERENCE:
        if not pd.DatetimeIndex(daily.loc[daily.strategy.eq(strategy), 'date']).equals(sample.index):
            raise ValueError(f'{strategy}: return dates differ from the trading calendar')
    metrics = performance(daily, compare_paper=price_hash == EXPECTED_SHA256)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / 'daily.csv', index=False)
    pd.concat(selections, ignore_index=True).to_csv(output / 'selections.csv', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    (output / 'run.json').write_text(json.dumps({
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'paper': 'https://arxiv.org/html/2403.07998v1',
        'price_sha256': price_hash, 'months': len(jobs), 'trading_days': len(sample),
        'first_date': str(sample.index[0].date()), 'last_date': str(sample.index[-1].date()),
        'regression_window': WINDOW, 'adf_lag': 1, 'baseline_pairs': BASELINE_PAIRS,
        'z_threshold': Z_THRESHOLD, 'transaction_cost': COST_MODEL,
    }, indent=2) + '\n')
    print(metrics[['strategy', 'basis', 'cumulative_return', 'annualized_return', 'sharpe',
                   'paper_cumulative_return', 'paper_sharpe']].to_string(index=False))


if __name__ == '__main__':
    main()
