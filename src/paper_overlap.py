"""Compare overlap and training-risk constraints under the paper replay rules."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_array, vstack

from . import paper
from .costs import PairPositions, charge_costs, pair_weights, terminal_holdings, transaction_model, volume_rule
from .training import TrainingBook
from .trading import summarize_original_style

TRAIN_DAYS = 252
CAP = 3
SEARCH_STOP = 'full_pass_with_zero_accepted_swaps'
REMOVALS = 24
OUTPUT = paper.ROOT / 'results/paper_overlap'
MODELS = ('M', 'CAP3', 'RISK3')


def degree(x, y, ids, n_assets):
    return np.bincount(np.r_[x[ids], y[ids]], minlength=n_assets)


def maximum_score(x, y, score, baseline, n_assets):
    """Fixed-K maximum score over the full graph, with the degree cap alone."""
    k = len(baseline)
    edges = np.arange(len(score))
    incident = coo_array((np.ones(2 * len(edges)), (np.r_[x, y], np.tile(edges, 2))),
                         shape=(n_assets, len(edges))).tocsc()
    count = coo_array((np.ones(len(edges)), (np.zeros(len(edges), int), edges)),
                      shape=(1, len(edges))).tocsc()
    upper = np.full(n_assets, CAP)
    # HiGHS shares its thread scheduler between LP and MILP in each worker.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected')
        lp = linprog(-score, A_ub=incident, b_ub=upper, A_eq=count, b_eq=[k],
                     bounds=(0, 1), method='highs-ds', options={'threads': 1})
    if not lp.success:
        raise RuntimeError(lp.message)
    if np.max(np.abs(lp.x - np.rint(lp.x))) < 1e-8:
        return np.flatnonzero(lp.x > .5)

    def solve(ids):
        matrix = vstack([incident[:, ids], count[:, ids]], format='csc')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected')
            result = milp(-score[ids], integrality=np.ones(len(ids), np.uint8),
                          bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, np.r_[np.zeros(n_assets), k],
                                                       np.r_[upper, k]),
                          options={'mip_rel_gap': 1e-10, 'threads': 1})
        if not result.success:
            raise RuntimeError(result.message)
        return ids[result.x > .5]

    support = lp.x > 0
    reduced_cost = lp.lower.marginals
    seed = np.unique(np.r_[edges[support | (reduced_cost <= .1)], baseline])
    incumbent = solve(seed)
    # A zero-valued edge's lower-bound reduced cost is its minimum objective
    # penalty when forced in. Keep every edge that could still beat the seed.
    gap = -lp.fun - score[incumbent].sum()
    remaining = np.unique(np.r_[edges[support | (reduced_cost <= gap + 1e-7)], incumbent])
    return solve(remaining)


def training_variance(num, weight, ids, returns_for=paper.ratio_returns):
    returns = returns_for(num[:, ids].sum(axis=1), weight[:, ids].sum(axis=1))
    return float(np.var(returns, ddof=1))


def risk_selection(x, y, score, num, weight, baseline, cap_only, n_assets, returns_for=paper.ratio_returns,
                   *, swaps_per_pass=None):
    """Increase ADF score through feasible one-pair replacements."""
    budget = training_variance(num, weight, baseline, returns_for)
    tolerance = budget * 1e-9 + 1e-16
    cap_risk = training_variance(num, weight, cap_only, returns_for)
    if cap_risk <= budget + tolerance:
        return cap_only, budget, True, 0

    ids = baseline.copy()
    n = num[:, ids].sum(axis=1)
    w = weight[:, ids].sum(axis=1)
    counts = degree(x, y, ids, n_assets)
    selected = np.zeros(len(score), dtype=bool)
    selected[ids] = True
    ranking = np.argsort(-score, kind='stable')
    total_swaps = 0
    while True:
        accepted = 0
        for add in ranking:
            if selected[add]:
                continue
            a, b = x[add], y[add]
            allowed = score[ids] < score[add] - 1e-12
            if counts[a] == CAP:
                allowed &= (x[ids] == a) | (y[ids] == a)
            if counts[b] == CAP:
                allowed &= (x[ids] == b) | (y[ids] == b)
            replace = np.flatnonzero(allowed)
            if not len(replace):
                continue
            replace = replace[np.argsort(score[ids[replace]], kind='stable')[:REMOVALS]]
            remove = ids[replace]
            trial_num = n[:, None] + num[:, add, None] - num[:, remove]
            trial_weight = w[:, None] + weight[:, add, None] - weight[:, remove]
            risk = returns_for(trial_num, trial_weight).var(axis=0, ddof=1)
            valid = np.flatnonzero(risk <= budget + tolerance)
            if not len(valid):
                continue
            best = valid[0]
            old = remove[best]
            counts[x[old]] -= 1
            counts[y[old]] -= 1
            counts[a] += 1
            counts[b] += 1
            selected[old] = False
            selected[add] = True
            ids[replace[best]] = add
            n, w = trial_num[:, best], trial_weight[:, best]
            accepted += 1
        if swaps_per_pass is not None:
            swaps_per_pass.append(accepted)
        total_swaps += accepted
        if accepted == 0:
            break
    if training_variance(num, weight, ids, returns_for) > budget + tolerance:
        raise ValueError('Risk search violated its training budget')
    return ids, budget, False, total_swaps


def run_month(prices, source, available=None):
    train_end = TRAIN_DAYS + paper.WINDOW
    cutoff = prices.index[train_end - 1]
    raw = prices.to_numpy()
    x, y, t = paper.formation_graph(raw[TRAIN_DAYS:train_end])
    lookup = {(a, b): i for i, (a, b) in enumerate(zip(prices.columns[x], prices.columns[y]))}
    baseline = np.array([lookup[a, b] for a, b in zip(source.asset_x, source.asset_y)])
    np.testing.assert_allclose(t[baseline], source.adf_t, rtol=1e-10, atol=1e-10)
    k = len(baseline)
    cap_only = maximum_score(x, y, -t, baseline, raw.shape[1])
    assert len(cap_only) == k and degree(x, y, cap_only, raw.shape[1]).max() <= CAP
    assert -t[cap_only].sum() >= -t[baseline].sum() - 1e-8

    pool = np.union1d(np.r_[baseline, cap_only], np.argsort(t, kind='stable')[:2 * k])
    a, b, scores = x[pool], y[pool], -t[pool]
    base_ids = np.searchsorted(pool, baseline)
    cap_ids = np.searchsorted(pool, cap_only)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(raw)
    marked = prices.ffill().to_numpy()
    returns = marked[paper.WINDOW:] / marked[paper.WINDOW - 1:-1] - 1
    returns = np.nan_to_num(returns, nan=0.)
    stale = ~np.isfinite(raw[train_end:])
    dates = prices.index[train_end:]
    formation_beta = np.empty((TRAIN_DAYS, len(pool)))
    formation_signal = np.empty_like(formation_beta)
    trade_beta = np.empty((len(dates), len(pool)))
    trade_signals = {'Z': np.empty_like(trade_beta)}
    for start in range(0, len(pool), 24):
        block = slice(start, start + 24)
        ai, bi = a[block], b[block]
        # Use the same OLS summation origin as the paper replay for trading.
        # The preceding 252 training returns retain their original rolling windows.
        train_beta, train_signals = paper.daily_signals(logs[:train_end - 1], ai, bi)
        out_beta, out_signals = paper.daily_signals(logs[TRAIN_DAYS:-1], ai, bi)
        formation_beta[:, block] = train_beta
        formation_signal[:, block] = train_signals['Z']
        trade_beta[:, block] = out_beta
        trade_signals['Z'][:, block] = out_signals['Z']

    daily, selected, diagnostics, positions = [], [], [], {}
    for kind in ('Z',):
        can_trade = None if available is None else available[paper.WINDOW-1:train_end-1]
        book = TrainingBook(formation_beta, formation_signal, returns[:TRAIN_DAYS], a, b,
                            can_trade, marked[paper.WINDOW-1:train_end])
        train_n, train_w = book.num, book.weight
        swaps_per_pass = []
        risk_ids, budget, exact, moves = risk_selection(
            a, b, scores, train_n, train_w, base_ids, cap_ids, raw.shape[1], book.returns,
            swaps_per_pass=swaps_per_pass)
        for model, ids in zip(MODELS, (base_ids, cap_ids, risk_ids)):
            is_risk = model == 'RISK3'
            max_degree = int(degree(a, b, ids, raw.shape[1]).max())
            assert len(np.unique(ids)) == k and max_degree <= (1 if model == 'M' else CAP)
            train_risk = training_variance(train_n, train_w, ids, book.returns)
            assert not is_risk or train_risk <= budget * (1 + 1e-9) + 1e-16
            gross, active = paper.portfolio_returns(trade_beta[:, ids], trade_signals[kind][:, ids],
                                                   returns[TRAIN_DAYS:, a[ids]], returns[TRAIN_DAYS:, b[ids]])
            wx, wy = pair_weights(trade_beta[:, ids], trade_signals[kind][:, ids])
            positions[model] = PairPositions(
                dates, tuple(zip(prices.columns[a[ids]], prices.columns[b[ids]])),
                wx, wy, returns[TRAIN_DAYS:, a[ids]], returns[TRAIN_DAYS:, b[ids]])
            daily.append(pd.DataFrame({
                'date': dates, 'cutoff': cutoff, 'model': model, 'signal': kind,
                'gross_return': gross,
                'active_pairs': active.sum(axis=1),
                'pairs_with_missing_end_quote': ((stale[:, a[ids]] | stale[:, b[ids]]) & active).sum(axis=1),
            }))
            selected.append(pd.DataFrame({
                'cutoff': cutoff, 'model': model, 'signal': kind,
                'asset_x': prices.columns[a[ids]], 'asset_y': prices.columns[b[ids]],
                'adf_score': scores[ids],
            }))
            diagnostics.append({
                'cutoff': cutoff, 'model': model, 'signal': kind, 'pairs': k,
                'max_degree': max_degree, 'score': float(scores[ids].sum()),
                'risk_budget': budget, 'training_variance': train_risk,
                'realized_variance': float(np.var(gross, ddof=1)),
                'global_optimal': exact if is_risk else True,
                'accepted_swaps': moves if is_risk else 0,
                'search_passes': len(swaps_per_pass) if is_risk else 0,
                'swaps_per_pass': json.dumps(swaps_per_pass if is_risk else []),
                'candidate_edges': len(t), 'risk_search_pool': len(pool),
            })
    return pd.concat(daily), pd.concat(selected), diagnostics, positions


def bootstrap(new, reference):
    """Paired circular moving blocks, 21 trading days, 10,000 draws."""
    rng = np.random.default_rng(20261109)
    size = len(new)
    differences = []
    for _ in range(20):
        starts = rng.integers(size, size=(500, (size + 20) // 21))
        ids = ((starts[:, :, None] + np.arange(21)) % size).reshape(500, -1)[:, :size]
        differences.extend(((new - reference)[ids].mean(axis=1) * 252).tolist())
    low, high = np.quantile(differences, [.025, .975])
    return float((new - reference).mean() * 252), low, high


def summarize(daily, months):
    daily = daily[daily.signal.eq('Z')]
    months = months[months.signal.eq('Z')]
    rows, contrasts = [], []
    for (kind, model), d in daily.groupby(['signal', 'model'], sort=False):
        md = months[months.signal.eq(kind) & months.model.eq(model)]
        gross, net = d.gross_return, d.net_return
        gross_stats = summarize_original_style(gross)
        net_stats = summarize_original_style(net)
        wealth = np.r_[1., np.cumprod(1 + net.to_numpy())]
        rows.append({
            'signal': kind, 'model': model, 'gross_cumulative': gross_stats['cumulative_return'],
            'net_cumulative': net_stats['cumulative_return'],
            'gross_annualized_return': gross_stats['annualized_return'],
            'net_annualized_return': net_stats['annualized_return'],
            'net_annual_vol': net.std() * np.sqrt(252),
            'gross_sharpe': gross_stats['sharpe'], 'net_sharpe': net_stats['sharpe'],
            'net_max_drawdown': np.min(wealth / np.maximum.accumulate(wealth) - 1),
            'training_variance_ratio_mean': (md.training_variance / md.risk_budget).mean(),
            'training_over_budget_months': (md.training_variance > md.risk_budget * (1 + 1e-9) + 1e-16).sum(),
            'realized_over_budget_months': (md.realized_variance > md.risk_budget * (1 + 1e-9) + 1e-16).sum(),
            'global_optimum_months': md.global_optimal.sum(),
            'missing_end_quote_pair_days': d.pairs_with_missing_end_quote.sum(),
        })
    for kind in ('Z',):
        md = months[months.signal.eq(kind)].pivot(index='cutoff', columns='model', values='score')
        returns = daily[daily.signal.eq(kind)].pivot(index='date', columns='model', values='net_return')
        for model, base in [('CAP3', 'M'), ('RISK3', 'CAP3'), ('RISK3', 'M')]:
            n, r = returns[model].to_numpy(), returns[base].to_numpy()
            diff, low, high = bootstrap(n, r)
            contrasts.append({
                'signal': kind, 'model': model, 'reference': base,
                'mean_monthly_score_change': (md[model] / md[base] - 1).mean(),
                'net_cumulative_change': np.prod(1 + n) - np.prod(1 + r),
                'annual_mean_return_change': diff, 'ci_95_low': low, 'ci_95_high': high,
            })
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    paper.add_input_arguments(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    prices, price_hash, market = paper.load_market(args.prices)
    sample = paper.trading_sample(prices, paper.WINDOW + TRAIN_DAYS)
    paper_output = args.output_root / 'paper'
    output = args.output_root / 'paper_overlap'
    metadata = json.loads((paper_output / 'run.json').read_text())
    if metadata['price_sha256'] != price_hash:
        raise ValueError('Paper replay and experiment must use the same price input')
    if metadata.get('transaction_cost') != transaction_model(market):
        raise ValueError('Re-run the paper baseline with the current transaction-cost model first')
    if metadata.get('volume_execution', volume_rule(None)) != volume_rule(market):
        raise ValueError('Re-run the paper baseline with the same volume execution rule first')
    if (metadata['first_date'] != str(sample.index[0].date())
            or metadata['last_date'] != str(sample.index[-1].date())
            or metadata['trading_days'] != len(sample)):
        raise ValueError('Paper baseline and overlap experiment must use the same trading sample')
    reference = pd.read_csv(paper_output / 'selections.csv', parse_dates=['cutoff'],
                            dtype={'asset_x': str, 'asset_y': str})
    m = reference[reference.method.eq('M')]
    original = pd.read_csv(paper_output / 'daily.csv', parse_dates=['date', 'cutoff'])
    columns = ['date', 'cutoff', 'gross_return', 'net_return', 'fee', 'active_pairs',
               'pairs_with_missing_end_quote', 'rebalance_turnover', 'liquidation_turnover', 'turnover']
    daily, selections, months, cost_states = [], [], [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for _, month in sample.groupby(sample.index.to_period('M')):
            first = prices.index.get_loc(month.index[0])
            last = prices.index.get_loc(month.index[-1])
            cutoff = prices.index[first - 1]
            panel = slice(first - paper.WINDOW - TRAIN_DAYS, last + 1)
            available = None if market is None else market.available.iloc[panel].to_numpy()
            jobs.append(pool.submit(run_month, prices.iloc[panel], m[m.cutoff.eq(cutoff)], available))
        for index, job in enumerate(jobs, 1):
            rows, pairs, checks, positions = job.result()
            rows = charge_costs(rows, positions, cost_states, group_column='model',
                                liquidate=index == len(jobs), market=market)
            if market is not None:
                for check in checks:
                    check['realized_variance'] = rows.loc[rows.model.eq(check['model']), 'gross_return'].var()
            cutoff = rows.cutoff.iloc[0]
            for kind in ('Z',):
                a = rows[rows.signal.eq(kind) & rows.model.eq('M')][columns].reset_index(drop=True)
                b = original[original.strategy.eq('M' + kind) & original.cutoff.eq(cutoff)]
                pd.testing.assert_frame_equal(a, b[columns].reset_index(drop=True),
                                              check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-10)
            daily.append(rows)
            selections.append(pairs)
            months.extend(checks)
            print(f'{index}/{len(jobs)} {rows.date.iloc[0]:%Y-%m}', flush=True)
    daily, months = pd.concat(daily, ignore_index=True), pd.DataFrame(months)
    metrics, comparisons = summarize(daily, months)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [('daily', daily), ('selections', pd.concat(selections, ignore_index=True)),
                        ('monthly', months), ('metrics', metrics), ('comparisons', comparisons)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    (output / 'run.json').write_text(json.dumps({
        'price_sha256': price_hash, 'paper_baseline_sha256': paper.sha256(paper_output / 'selections.csv'),
        'months': len(jobs), 'trading_days': len(sample), 'cap': CAP, 'training_days': TRAIN_DAYS,
        'first_date': str(sample.index[0].date()), 'last_date': str(sample.index[-1].date()),
        'risk_search_pool': 'M and exact CAP3 plus top 2K ADF edges',
        'risk_search_stop': SEARCH_STOP, 'risk_search_removals_per_insert': REMOVALS,
        'transaction_cost': transaction_model(market),
        'volume_execution': volume_rule(market), 'terminal_holdings': terminal_holdings(cost_states),
        'bootstrap': 'paired circular, 21 days, 10000 draws, seed 20261109',
    }, indent=2) + '\n')
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == '__main__':
    main()
