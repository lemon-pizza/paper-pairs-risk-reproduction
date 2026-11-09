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

TRAIN_DAYS = 252
CAP = 3
PASSES = 3
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


def training_variance(num, weight, ids):
    returns = paper.ratio_returns(num[:, ids].sum(axis=1), weight[:, ids].sum(axis=1))
    return float(np.var(returns, ddof=1))


def risk_selection(x, y, score, num, weight, baseline, cap_only, n_assets):
    """Increase ADF score through feasible one-pair replacements."""
    budget = training_variance(num, weight, baseline)
    tolerance = budget * 1e-9 + 1e-16
    cap_risk = training_variance(num, weight, cap_only)
    if cap_risk <= budget + tolerance:
        return cap_only, budget, True, 0

    ids = baseline.copy()
    n = num[:, ids].sum(axis=1)
    w = weight[:, ids].sum(axis=1)
    counts = degree(x, y, ids, n_assets)
    selected = np.zeros(len(score), dtype=bool)
    selected[ids] = True
    ranking = np.argsort(-score, kind='stable')
    accepted = 0
    for _ in range(PASSES):
        before = accepted
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
            risk = paper.ratio_returns(trial_num, trial_weight).var(axis=0, ddof=1)
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
        if accepted == before:
            break
    if training_variance(num, weight, ids) > budget + tolerance:
        raise ValueError('Risk search violated its training budget')
    return ids, budget, False, accepted


def run_month(prices, source):
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
    num = {kind: np.empty((len(returns), len(pool))) for kind in ('Q', 'Z')}
    weight = {kind: np.empty_like(array) for kind, array in num.items()}
    for start in range(0, len(pool), 24):
        block = slice(start, start + 24)
        ai, bi = a[block], b[block]
        beta, signals = paper.daily_signals(logs[:-1], ai, bi)
        for kind, signal in signals.items():
            num[kind][:, block], weight[kind][:, block] = paper.pair_contributions(
                beta, signal, returns[:, ai], returns[:, bi])

    daily, selected, diagnostics = [], [], []
    for kind in ('Q', 'Z'):
        n, w = num[kind], weight[kind]
        train_n, train_w = n[:TRAIN_DAYS], w[:TRAIN_DAYS]
        risk_ids, budget, exact, moves = risk_selection(
            a, b, scores, train_n, train_w, base_ids, cap_ids, raw.shape[1])
        for model, ids in zip(MODELS, (base_ids, cap_ids, risk_ids)):
            is_risk = model == 'RISK3'
            max_degree = int(degree(a, b, ids, raw.shape[1]).max())
            assert len(np.unique(ids)) == k and max_degree <= (1 if model == 'M' else CAP)
            train_risk = training_variance(train_n, train_w, ids)
            assert not is_risk or train_risk <= budget * (1 + 1e-9) + 1e-16
            total_weight = w[TRAIN_DAYS:, ids].sum(axis=1)
            gross = paper.ratio_returns(n[TRAIN_DAYS:, ids].sum(axis=1), total_weight)
            fee = np.where(total_weight > 0, paper.DAILY_FEE, 0.)
            active = w[TRAIN_DAYS:, ids] > 0
            daily.append(pd.DataFrame({
                'date': dates, 'cutoff': cutoff, 'model': model, 'signal': kind,
                'gross_return': gross, 'net_return': gross - fee, 'fee': fee,
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
                'candidate_edges': len(t), 'risk_search_pool': len(pool),
            })
    return pd.concat(daily), pd.concat(selected), diagnostics


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
    rows, contrasts = [], []
    for (kind, model), d in daily.groupby(['signal', 'model'], sort=False):
        md = months[months.signal.eq(kind) & months.model.eq(model)]
        gross, net = d.gross_return, d.net_return
        wealth = np.r_[1., np.cumprod(1 + net.to_numpy())]
        rows.append({
            'signal': kind, 'model': model, 'gross_cumulative': (1 + gross).prod() - 1,
            'net_cumulative': wealth[-1] - 1, 'net_annual_vol': net.std() * np.sqrt(252),
            'net_sharpe': net.mean() / net.std() * np.sqrt(252),
            'net_max_drawdown': np.min(wealth / np.maximum.accumulate(wealth) - 1),
            'training_variance_ratio_mean': (md.training_variance / md.risk_budget).mean(),
            'training_over_budget_months': (md.training_variance > md.risk_budget * (1 + 1e-9) + 1e-16).sum(),
            'realized_over_budget_months': (md.realized_variance > md.risk_budget * (1 + 1e-9) + 1e-16).sum(),
            'global_optimum_months': md.global_optimal.sum(),
            'missing_end_quote_pair_days': d.pairs_with_missing_end_quote.sum(),
        })
    for kind in ('Q', 'Z'):
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
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    price_hash = paper.sha256(paper.DEFAULT_PRICES)
    metadata = json.loads((paper.OUTPUT / 'run.json').read_text())
    if price_hash != paper.EXPECTED_SHA256 or metadata['price_sha256'] != price_hash:
        raise ValueError('Paper replay and experiment must use the same price input')
    prices = pd.read_parquet(paper.DEFAULT_PRICES).sort_index(axis=1)
    reference = pd.read_csv(paper.OUTPUT / 'selections.csv', parse_dates=['cutoff'])
    m = reference[reference.method.eq('M')]
    original = pd.read_csv(paper.OUTPUT / 'daily.csv', parse_dates=['date', 'cutoff'])
    columns = ['date', 'cutoff', 'gross_return', 'net_return', 'fee', 'active_pairs',
               'pairs_with_missing_end_quote']
    sample = prices.loc[paper.START:paper.END]
    daily, selections, months = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for _, month in sample.groupby(sample.index.to_period('M')):
            first = prices.index.get_loc(month.index[0])
            last = prices.index.get_loc(month.index[-1])
            cutoff = prices.index[first - 1]
            jobs.append(pool.submit(run_month, prices.iloc[first - paper.WINDOW - TRAIN_DAYS:last + 1],
                                    m[m.cutoff.eq(cutoff)]))
        for index, job in enumerate(jobs, 1):
            rows, pairs, checks = job.result()
            cutoff = rows.cutoff.iloc[0]
            for kind in ('Q', 'Z'):
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('daily', daily), ('selections', pd.concat(selections, ignore_index=True)),
                        ('monthly', months), ('metrics', metrics), ('comparisons', comparisons)]:
        frame.to_csv(OUTPUT / f'{name}.csv', index=False)
    (OUTPUT / 'run.json').write_text(json.dumps({
        'price_sha256': price_hash, 'paper_baseline_sha256': paper.sha256(paper.OUTPUT / 'selections.csv'),
        'months': len(jobs), 'trading_days': len(sample), 'cap': CAP, 'training_days': TRAIN_DAYS,
        'risk_search_pool': 'M and exact CAP3 plus top 2K ADF edges',
        'risk_search_passes': PASSES, 'risk_search_removals_per_insert': REMOVALS,
        'fee': paper.DAILY_FEE, 'bootstrap': 'paired circular, 21 days, 10000 draws, seed 20261109',
    }, indent=2) + '\n')
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == '__main__':
    main()
