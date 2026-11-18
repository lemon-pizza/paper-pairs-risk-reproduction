"""One-way execution costs on traded notional, with positions carried across months."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


FEE_BPS = 3.0
FEE_RATE = FEE_BPS / 10_000
COST_MODEL = {
    'one_way_bps': FEE_BPS,
    'basis': 'absolute traded notional / pre-trade portfolio equity',
    'netting': 'within each pair leg; no netting across different pairs',
    'rebalance': 'daily targets versus price-drifted holdings after previous costs',
    'monthly_boundary': 'carry retained pairs; close removed pairs and open new pairs',
    'terminal_liquidation': True,
    'execution_prices': 'existing close-to-close marked prices; missing quotes forward-filled',
}


@dataclass
class PairPositions:
    dates: pd.DatetimeIndex
    pairs: tuple
    x_weight: np.ndarray
    y_weight: np.ndarray
    x_return: np.ndarray
    y_return: np.ndarray


@dataclass
class CostState:
    # Signed marked notionals as fractions of current (after-cost) equity.
    holdings: dict = field(default_factory=dict)
    last_date: pd.Timestamp | None = None


class ExecutionMarket:
    """Observed close availability and marked asset returns, including former selections."""

    def __init__(self, prices, available):
        if not (prices.index.equals(available.index) and prices.columns.equals(available.columns)):
            raise ValueError('Execution availability must use the price calendar and tickers')
        self.available = available
        self.quoted = prices.notna().to_numpy()
        self.returns = prices.ffill().pct_change(fill_method=None).fillna(0.).to_numpy()


def volume_rule(market):
    return 'positive_volume_and_valid_close' if market is not None else 'price_only_no_volume_data'


def transaction_model(market):
    if market is None:
        return COST_MODEL
    return dict(COST_MODEL,
                execution_prices='observed execution close with positive volume and a valid price; forward fill marks only',
                monthly_boundary='removed positions target zero; blocked shares carry until an executable close',
                terminal_liquidation='attempt at final close; mark and report any remaining holdings')


def terminal_holdings(states):
    rows = []
    for key, state in states.items():
        label = {'strategy': key} if isinstance(key, str) else {'strategy': key[0], 'execution': key[1]}
        for (x, y), (wx, wy) in sorted(state.holdings.items()):
            rows.append(dict(label, asset_x=x, asset_y=y, x_weight=wx, y_weight=wy))
    return rows


def pair_weights(beta, signal):
    """Leg weights on pre-trade equity; the sum of absolute pair weights is one."""
    beta, signal = np.asarray(beta, float), np.asarray(signal, float)
    if beta.shape != signal.shape or beta.ndim != 2:
        raise ValueError('beta and signal must be matching time-by-pair arrays')
    active = np.isfinite(beta) & np.isfinite(signal) & (signal != 0)
    beta, signal = np.where(active, beta, 0.), np.where(active, signal, 0.)
    total = np.abs(signal).sum(axis=1, keepdims=True)
    scale = np.divide(signal, total, out=np.zeros_like(signal), where=total > 0)
    return scale * beta / (1 + np.abs(beta)), -scale / (1 + np.abs(beta))


def transaction_costs(positions, gross, state, liquidate=False):
    """Charge actual changes in each sleeve's holdings, including terminal exits.

    Targets use equity immediately before the rebalance. Fees are paid from cash;
    end-of-day equity is therefore pre-trade equity * (1 + gross - fee). Existing
    leg notionals drift with their asset returns and are divided by this equity
    before the next day's trades. A reversal counts both closing and reopening.
    """
    dates = pd.DatetimeIndex(positions.dates)
    pairs = tuple(positions.pairs)
    gross = np.asarray(gross, float)
    arrays = tuple(np.asarray(a, float) for a in (
        positions.x_weight, positions.y_weight, positions.x_return, positions.y_return))
    expected = (len(dates), len(pairs))
    if not len(dates) or any(a.shape != expected for a in arrays) or gross.shape != (len(dates),):
        raise ValueError('Position dimensions must match the dates and pairs')
    if len(set(pairs)) != len(pairs) or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Pairs and chronological trading dates must be unique')
    if state.last_date is not None and dates[0] <= state.last_date:
        raise ValueError('Cost blocks must follow the previous block in time')
    if not all(np.isfinite(a).all() for a in (*arrays, gross)):
        raise ValueError('Position weights and returns must be finite')
    wx, wy, rx, ry = arrays
    if np.any(rx < -1) or np.any(ry < -1):
        raise ValueError('Asset simple returns cannot be below -100%')
    if not np.allclose((wx * rx + wy * ry).sum(axis=1), gross, atol=1e-12, rtol=1e-10):
        raise ValueError('Leg P&L does not match the reported gross return')

    previous_x = np.array([state.holdings.get(p, (0., 0.))[0] for p in pairs])
    previous_y = np.array([state.holdings.get(p, (0., 0.))[1] for p in pairs])
    retained = set(pairs)
    removed = sum(abs(x) + abs(y) for p, (x, y) in state.holdings.items() if p not in retained)
    rebalance = np.zeros(len(dates))
    closing = np.zeros(len(dates))
    fees = np.zeros(len(dates))
    for t in range(len(dates)):
        rebalance[t] = np.abs(wx[t] - previous_x).sum() + np.abs(wy[t] - previous_y).sum()
        if t == 0:
            rebalance[t] += removed
        end_x, end_y = wx[t] * (1 + rx[t]), wy[t] * (1 + ry[t])
        if liquidate and t == len(dates) - 1:
            closing[t] = np.abs(end_x).sum() + np.abs(end_y).sum()
        fees[t] = FEE_RATE * (rebalance[t] + closing[t])
        equity = 1 + gross[t] - fees[t]
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError(f'{dates[t]}: non-positive equity after transaction costs')
        previous_x, previous_y = end_x / equity, end_y / equity
    state.holdings = {} if liquidate else {
        p: (float(x), float(y)) for p, x, y in zip(pairs, previous_x, previous_y)
        if x != 0 or y != 0
    }
    state.last_date = dates[-1]
    return fees, rebalance, closing


def executed_days(block, state, liquidate, market=None):
    """Rebalance at the preceding close, mark to this day's close, then attempt terminal exits."""
    dates = pd.DatetimeIndex(block.dates)
    if market is None:
        desired_gross = (block.x_weight * block.x_return + block.y_weight * block.y_return).sum(axis=1)
        fees, rebalance, closing = transaction_costs(block, desired_gross, state, liquidate)
        size = np.abs(block.x_weight) + np.abs(block.y_weight)
        return pd.DataFrame({'gross_return': desired_gross, 'active_pairs': (size > 0).sum(axis=1),
                             'gross_exposure': size.sum(axis=1), 'rebalance_turnover': rebalance,
                             'liquidation_turnover': closing, 'turnover': rebalance + closing,
                             'fee': fees, 'net_return': desired_gross - fees})

    row = market.available.index.get_indexer(dates)
    if (row <= 0).any() or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Each P&L date needs a preceding exchange close')
    if state.last_date is not None and market.available.index[row[0] - 1] != state.last_date:
        raise ValueError('Volume-aware blocks must follow the preceding trading day')
    # A discontinued pair keeps zero targets until all its previously filled shares can be sold.
    keys = tuple(block.pairs) + tuple(sorted(state.holdings.keys() - set(block.pairs)))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate pair in execution block')
    columns = market.available.columns
    x = columns.get_indexer([p[0] for p in keys])
    y = columns.get_indexer([p[1] for p in keys])
    if (x < 0).any() or (y < 0).any():
        raise ValueError('No price history for a held ticker')
    assets = np.r_[x, y]
    count = len(keys)
    n = len(block.pairs)
    target = np.zeros((len(row), 2 * count))
    target[:, :n], target[:, count:count + n] = block.x_weight, block.y_weight
    if not np.isfinite(target).all():
        raise ValueError('Nonfinite target weights')
    previous = np.array([state.holdings.get(p, (0., 0.)) for p in keys]).reshape(-1, 2).T.ravel()
    can_trade = market.available.to_numpy()
    rows = []
    for t, day in enumerate(row):
        # Weights earn the return ending on day; timing is the same for delayed targets.
        allowed = can_trade[day - 1, assets]
        held = np.where(allowed, target[t], previous)
        rebalance = np.abs(held - previous).sum()
        blocked = np.abs(target[t] - held)
        blocked_count = (blocked > 1e-12).sum()
        blocked_notional = blocked.sum()
        returns = market.returns[day, assets]
        gross = float(held @ returns)
        marked = held * (1 + returns)
        closing = 0.
        if liquidate and t == len(row) - 1:
            can_exit = can_trade[day, assets]
            closing = np.abs(marked[can_exit]).sum()
            marked = np.where(can_exit, 0., marked)
            blocked_count += (np.abs(marked) > 1e-12).sum()
            blocked_notional += np.abs(marked).sum()
        turnover = rebalance + closing
        fee = FEE_RATE * turnover
        net = gross - fee
        if not np.isfinite(net) or 1 + net <= 0:
            raise ValueError(f'{dates[t]}: non-positive equity after transaction costs')
        previous = marked / (1 + net)
        active = held != 0
        missing = active & ~market.quoted[day, assets]
        rows.append((gross, net, fee, rebalance, closing, turnover, np.abs(held).sum(),
                     (active[:count] | active[count:]).sum(), (missing[:count] | missing[count:]).sum(),
                     blocked_count, blocked_notional))
    state.holdings = {p: (float(previous[i]), float(previous[count + i])) for i, p in enumerate(keys)
                      if previous[i] != 0 or previous[count + i] != 0}
    state.last_date = dates[-1]
    return pd.DataFrame(rows, columns=['gross_return', 'net_return', 'fee', 'rebalance_turnover',
                        'liquidation_turnover', 'turnover', 'gross_exposure', 'active_pairs',
                        'pairs_with_missing_end_quote', 'blocked_leg_orders', 'blocked_notional'])


def charge_costs(daily, positions, states, group_column='strategy', liquidate=False, market=None):
    """Apply sequential portfolio accounting to independently computed monthly signals."""
    daily = daily.reset_index(drop=True).copy()
    if set(daily[group_column]) != set(positions):
        raise ValueError('Every strategy needs a matching position block')
    for group, block in positions.items():
        mask = daily[group_column].eq(group)
        rows = daily.loc[mask]
        if not pd.DatetimeIndex(rows.date).equals(pd.DatetimeIndex(block.dates)):
            raise ValueError('Position dates do not match return dates')
        state = states.setdefault(group, CostState())
        if market is None:
            fees, rebalance, closing = transaction_costs(block, rows.gross_return, state, liquidate)
            daily.loc[mask, 'rebalance_turnover'] = rebalance
            daily.loc[mask, 'liquidation_turnover'] = closing
            daily.loc[mask, 'turnover'] = rebalance + closing
            daily.loc[mask, 'fee'] = fees
            daily.loc[mask, 'net_return'] = rows.gross_return.to_numpy() - fees
        else:
            filled = executed_days(block, state, liquidate, market)
            for column in filled:
                daily.loc[mask, column] = filled[column].to_numpy()
    return daily
