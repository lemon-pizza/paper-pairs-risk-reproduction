"""Fee-free training returns for a fixed pair set with orders blocked at unavailable closes."""

import numpy as np
from scipy.optimize import brentq

from . import paper


def capital_fraction(variance, budget):
    if not np.isfinite([variance, budget]).all() or min(variance, budget) < 0:
        raise ValueError('Training variance and budget must be finite and nonnegative')
    return 1.0 if variance <= budget else float(np.sqrt(budget / variance))


class TrainingBook:
    """Additive leg P&L at the last executable close, before normalization by active pairs."""

    def __init__(self, beta, signal, returns, x, y, available=None, marks=None):
        rx, ry = returns[:, x], returns[:, y]
        direct, self.weight = paper.pair_contributions(beta, signal, rx, ry)
        self.num = direct
        self.origins = np.empty(0, dtype=int)
        self.steps = []
        if available is None:
            return

        active = self.weight > 0
        slopes = np.where(active, beta, 0.0)
        s = np.where(active, signal, 0.0) / (1 + np.abs(slopes))
        days = len(beta)
        clock = np.arange(days)[:, None]
        immediate = np.zeros_like(direct)
        events = []
        for assets, units, changes in [(x, s * slopes, rx), (y, -s, ry)]:
            allowed = available[:, assets]
            immediate += np.where(allowed, units * changes, 0.0)
            # A blocked leg holds shares bought at the most recent permitted close.
            # The training account begins in cash. Holding through a reopening earns its price jump.
            last_fill = np.maximum.accumulate(np.where(allowed, clock, -1), axis=0)
            t, edge = np.nonzero(~allowed & (last_fill >= 0) & (changes != 0))
            origin = last_fill[t, edge]
            ticker = assets[edge]
            coefficient = (
                units[origin, edge] * changes[t, edge] * marks[t, ticker] / marks[origin, ticker]
            )
            nonzero = coefficient != 0
            events.append((t[nonzero] * days + origin[nonzero], edge[nonzero], coefficient[nonzero]))
        self.num = immediate
        codes = np.concatenate([v[0] for v in events])
        if not len(codes):
            return
        unique, index = np.unique(codes, return_inverse=True)
        deferred = np.zeros((len(unique), direct.shape[1]))
        np.add.at(
            deferred,
            (index, np.concatenate([v[1] for v in events])),
            np.concatenate([v[2] for v in events]),
        )
        self.num = np.vstack([immediate, deferred])
        self.origins = unique % days
        times, starts = np.unique(unique // days, return_index=True)
        self.steps = [
            (day, slice(start, stop))
            for day, start, stop in zip(times, starts, np.r_[starts[1:], len(unique)])
        ]

    def returns(self, num, weight, fraction=1.0):
        """Price one portfolio or many one-swap trials using each original fill day's equity."""
        days = len(weight)
        gross = paper.ratio_returns(num[:days], weight) * fraction
        if not self.steps or not np.any(num[days:]):
            return gross
        delayed = paper.ratio_returns(num[days:], weight[self.origins]) * fraction
        wealth = np.ones((days + 1,) + gross.shape[1:])
        cursor = 0
        for day, rows in self.steps:
            if not np.any(delayed[rows]):
                continue
            # Intervening dates have only contemporaneous fills and compound directly.
            wealth[cursor + 1 : day + 1] = wealth[cursor] * np.cumprod(1 + gross[cursor:day], axis=0)
            pnl = (delayed[rows] * wealth[self.origins[rows]]).sum(axis=0)
            gross[day] += pnl / wealth[day]
            wealth[day + 1] = wealth[day] * (1 + gross[day])
            cursor = day + 1
        return gross

    def allocation(self, budget):
        """Scale the extension using training variance of the same executable holdings."""
        num, weight = self.num.sum(axis=1), self.weight.sum(axis=1)

        def variance(fraction):
            return float(self.returns(num, weight, fraction).var(ddof=1))

        full = variance(1.0)
        fraction = capital_fraction(full, budget)
        if fraction == 1.0:
            return full, 1.0, full
        if fraction == 0.0:
            return full, 0.0, 0.0
        if self.steps and np.any(num[len(weight) :]):
            # Frozen share counts can make future portfolio weights depend on the equity path.
            fraction = brentq(lambda f: variance(f) - budget, 0.0, 1.0, xtol=1e-13)
        return full, fraction, variance(fraction)
