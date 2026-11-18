"""Trade the highest-scoring half of RISK3's pairs within its monthly variance budget."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd

from . import paper
from .costs import COST_MODEL, CostState, PairPositions, pair_weights, transaction_costs
from .paper_overlap import TRAIN_DAYS
from .trading import WINDOW, summarize_original_style

PARENT = 'RISK3'
MODEL = 'RISK3_half_adf_budget'
SUITE = 'risk3_adf_half'
HISTORY = WINDOW + TRAIN_DAYS


def rank_parent(pairs):
    if (pairs[['asset_x', 'asset_y']].isna().any().any()
            or pairs.duplicated(['asset_x', 'asset_y']).any()
            or not np.isfinite(pairs.adf_score).all()):
        raise ValueError('RISK3 selections require unique pairs and finite ADF scores')
    return pairs.sort_values(['adf_score', 'asset_x', 'asset_y'],
                             ascending=[False, True, True], ignore_index=True)


def capital_fraction(variance, budget):
    if not np.isfinite([variance, budget]).all() or min(variance, budget) < 0:
        raise ValueError('Training variance and inherited budget must be finite and nonnegative')
    return 1.0 if variance <= budget else float(np.sqrt(budget / variance))


def z_history(logs, x, y):
    # Bound the memory occupied by 504-day rolling residuals, then normalize weights across all pairs.
    shape = (len(logs) - WINDOW + 1, len(x))
    beta, signal = np.empty(shape), np.empty(shape)
    for start in range(0, len(x), 24):
        batch = slice(start, start + 24)
        slopes, signals = paper.daily_signals(logs, x[batch], y[batch])
        beta[:, batch], signal[:, batch] = slopes, signals['Z']
    return beta, signal


def run_month(prices, parent_pairs, budget, trading_days):
    cutoff = prices.index[HISTORY - 1]
    ranked = rank_parent(parent_pairs)
    keep = len(ranked) // 2
    kept = ranked.iloc[:keep].copy()
    kept['rank'] = np.arange(1, keep + 1)
    kept['model'] = MODEL
    x = prices.columns.get_indexer(ranked.asset_x)
    y = prices.columns.get_indexer(ranked.asset_y)
    if (x < 0).any() or (y < 0).any():
        raise ValueError(f'{cutoff}: a RISK3 ticker is missing from the price input')
    with np.errstate(invalid='ignore', divide='ignore'):
        logs = np.log(prices.to_numpy())
    returns = prices.ffill().pct_change(fill_method=None).fillna(0.).to_numpy()

    train = slice(WINDOW, HISTORY)       # Last training return ends at the formation close.
    top_x, top_y = x[:keep], y[:keep]
    beta, signal = z_history(logs[:HISTORY - 1], top_x, top_y)
    gross, _ = paper.portfolio_returns(beta, signal, returns[train, top_x], returns[train, top_y])
    variance = float(gross.var(ddof=1))
    allocation = capital_fraction(variance, budget)
    diagnostics = {
        'cutoff': cutoff, 'parent_pairs': len(ranked), 'selected_pairs': keep,
        'unscaled_training_variance': variance, 'risk_budget': budget,
        'capital_fraction': allocation, 'scaled_training_variance': variance * allocation**2,
    }

    end = HISTORY + trading_days
    beta, signal = z_history(logs[TRAIN_DAYS:end - 1], x, y)
    keys = tuple(zip(ranked.asset_x, ranked.asset_y))
    positions = {}
    for model, count, fraction in [(PARENT, len(ranked), 1.), (MODEL, keep, allocation)]:
        wx, wy = pair_weights(beta[:, :count], signal[:, :count])
        wx *= fraction
        wy *= fraction
        for execution, lag in [('signal_close', 0), ('next_close', 1)]:
            # A delayed month's last target earns the first return of the following month.
            period = slice(HISTORY + lag, end + lag)
            dates = prices.index[period]
            positions[model, execution] = PairPositions(
                dates, keys[:count], wx[:len(dates)], wy[:len(dates)],
                returns[period, x[:count]], returns[period, y[:count]])
    return kept, diagnostics, positions


def daily_rows(block, state, final_date):
    gross = (block.x_weight * block.x_return + block.y_weight * block.y_return).sum(axis=1)
    fees, rebalances, exits = transaction_costs(block, gross, state, block.dates[-1] == final_date)
    size = np.abs(block.x_weight) + np.abs(block.y_weight)
    return pd.DataFrame({
        'date': block.dates, 'gross_return': gross, 'active_pairs': (size > 0).sum(axis=1),
        'gross_exposure': size.sum(axis=1), 'rebalance_turnover': rebalances,
        'liquidation_turnover': exits, 'turnover': rebalances + exits, 'fee': fees,
        'net_return': gross - fees,
    })


def performance(daily):
    rows = []
    for (model, execution), group in daily.groupby(['model', 'execution'], sort=False):
        row = {'model': model, 'execution': execution, 'days': len(group)}
        for basis in ('gross', 'net'):
            stats = summarize_original_style(group[basis + '_return'])
            for metric in ('cumulative_return', 'annualized_return', 'sharpe'):
                row[f'{basis}_{metric}'] = stats[metric]
        wealth = np.r_[1., (1 + group.net_return).cumprod()]
        row['net_annual_vol'] = group.net_return.std() * np.sqrt(252)
        row['net_max_drawdown'] = (wealth / np.maximum.accumulate(wealth) - 1).min()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    paper.add_input_arguments(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    parent = args.output_root / 'paper_overlap'
    output = args.output_root / SUITE
    prices, price_hash = paper.load_prices(args.prices)
    # The pinned Parquet uses microseconds; the parent CSV dates are read in nanoseconds.
    prices.index = prices.index.as_unit('ns')
    sample = paper.trading_sample(prices, HISTORY)
    meta = json.loads((parent / 'run.json').read_text())
    expected = {'price_sha256': price_hash, 'transaction_cost': COST_MODEL, 'training_days': TRAIN_DAYS,
                'first_date': str(sample.index[0].date()), 'last_date': str(sample.index[-1].date()),
                'trading_days': len(sample), 'months': sample.index.to_period('M').nunique()}
    for field, value in expected.items():
        if meta[field] != value:
            raise ValueError(f'RISK3 parent and extension differ in {field}')

    pairs = pd.read_csv(parent / 'selections.csv', parse_dates=['cutoff'],
                        dtype={'asset_x': str, 'asset_y': str})
    pairs = pairs[pairs.model.eq(PARENT) & pairs.signal.eq('Z')]
    months = pd.read_csv(parent / 'monthly.csv', parse_dates=['cutoff'])
    months = months[months.model.eq(PARENT) & months.signal.eq('Z')].set_index('cutoff', verify_integrity=True)
    daily, selections, diagnostics, states = [], [], [], {}
    # There is no earlier signal to execute for the first delayed return.
    empty = np.empty((1, 0))
    cash = PairPositions(sample.index[:1], (), empty, empty, empty, empty)
    for model in (PARENT, MODEL):
        state = states.setdefault((model, 'next_close'), CostState())
        first_day = daily_rows(cash, state, sample.index[-1])
        daily.append(first_day.assign(model=model, execution='next_close', cutoff=pd.NaT))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for _, month in sample.groupby(sample.index.to_period('M')):
            first, last = prices.index.get_indexer([month.index[0], month.index[-1]])
            cutoff = prices.index[first - 1]
            selected = pairs[pairs.cutoff.eq(cutoff)]
            params = months.loc[cutoff]
            if selected.empty or len(selected) != params.pairs:
                raise ValueError(f'{cutoff}: incomplete RISK3 parent selections')
            tickers = pd.unique(pd.concat([selected.asset_x, selected.asset_y], ignore_index=True))
            extra_day = int(month.index[-1] < sample.index[-1])
            panel = prices.iloc[first - HISTORY:last + 1 + extra_day].loc[:, tickers]
            jobs.append(pool.submit(run_month, panel, selected, params.risk_budget, len(month)))
        for job in jobs:
            kept, diagnostic, positions = job.result()
            selections.append(kept)
            diagnostics.append(diagnostic)
            for key, block in positions.items():
                if not len(block.dates):
                    continue
                model, execution = key
                rows = daily_rows(block, states.setdefault(key, CostState()), sample.index[-1])
                daily.append(rows.assign(model=model, execution=execution, cutoff=diagnostic['cutoff']))

    daily = pd.concat(daily, ignore_index=True)
    for _, group in daily.groupby(['model', 'execution']):
        if not pd.DatetimeIndex(group.date).equals(sample.index):
            raise ValueError('RISK3 extension returns differ from the market trading calendar')
    reference = pd.read_csv(parent / 'daily.csv', parse_dates=['date'])
    reference = reference[reference.model.eq(PARENT) & reference.signal.eq('Z')].set_index('date')
    replay = daily[daily.model.eq(PARENT) & daily.execution.eq('signal_close')].set_index('date')
    columns = ['gross_return', 'net_return', 'active_pairs', 'turnover', 'fee']
    pd.testing.assert_frame_equal(replay[columns], reference[columns], check_dtype=False,
                                  check_exact=False, rtol=1e-10, atol=1e-12)
    metrics = performance(daily)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [('daily', daily), ('monthly', pd.DataFrame(diagnostics)),
                        ('selections', pd.concat(selections, ignore_index=True)), ('metrics', metrics)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    manifest = dict(expected, model=MODEL, parent=PARENT,
                    parent_selections_sha256=paper.sha256(parent / 'selections.csv'),
                    parent_monthly_sha256=paper.sha256(parent / 'monthly.csv'),
                    ranking='descending parent adf_score; ties by asset_x then asset_y',
                    pair_count='floor(parent_pairs / 2)', risk_budget='parent monthly risk_budget',
                    capital_fraction='min(1, sqrt(risk_budget / unscaled_training_variance)); 1 at zero variance',
                    cash_return=0, executions=['signal_close', 'next_close'])
    (output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
