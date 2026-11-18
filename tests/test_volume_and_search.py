"""Execution-date volume, locked shares, fee-free formation risk and search termination."""

import unittest

import numpy as np
import pandas as pd

from src import paper_overlap
from src.costs import CostState, ExecutionMarket, FEE_RATE, PairPositions, executed_days, pair_weights
from src.training import TrainingBook


class TradabilityTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2020-01-02', periods=6)
        self.prices = pd.DataFrame(
            {'A': [100, 100, 120, 150, 150, 150], 'B': [100, 100, 100, 100, 100, 100]}, index=self.dates
        )
        self.available = pd.DataFrame(
            {'A': [True, False, False, True, True, True], 'B': [False, True, True, True, True, True]},
            index=self.dates,
        )
        self.market = ExecutionMarket(self.prices, self.available)

    def block(self, days, wx, wy, keys=(('A', 'B'),)):
        dates = self.dates[days]
        shape = (len(dates), len(keys))
        rx = self.market.returns[days, 0:1][:, : len(keys)]
        ry = self.market.returns[days, 1:2][:, : len(keys)]
        return PairPositions(
            dates, keys, np.array(wx).reshape(shape), np.array(wy).reshape(shape), rx, ry
        )

    def test_open_close_and_reopening_use_the_execution_dates_volume(self):
        state = CostState()
        day1 = executed_days(self.block(slice(1, 2), [0.5], [-0.5]), state, False, self.market).iloc[0]
        self.assertAlmostEqual(day1.rebalance_turnover, 0.5)
        self.assertAlmostEqual(day1.fee, 0.5 * FEE_RATE)
        self.assertEqual(day1.blocked_leg_orders, 1)
        self.assertEqual(state.holdings[('A', 'B')][1], 0.0)
        # The pair leaves the selection. A stays invested through two blocked closes.
        carry = self.block(slice(2, 5), [], [], keys=())
        result = executed_days(carry, state, True, self.market)
        self.assertEqual(result.rebalance_turnover.iloc[0], 0.0)
        self.assertEqual(result.rebalance_turnover.iloc[1], 0.0)
        self.assertAlmostEqual(result.gross_return.iloc[0], 0.5 * 0.2 / (1 - day1.fee))
        self.assertGreater(result.gross_return.iloc[1], 0.0)
        self.assertGreater(result.rebalance_turnover.iloc[2], 0.5)
        self.assertEqual(state.holdings, {})

    def test_final_close_without_volume_retains_the_position_and_has_no_exit_fee(self):
        state = CostState()
        result = executed_days(self.block(slice(1, 2), [0.5], [-0.5]), state, True, self.market).iloc[0]
        self.assertEqual(result.liquidation_turnover, 0.0)
        self.assertAlmostEqual(result.fee, 0.5 * FEE_RATE)
        self.assertEqual(result.blocked_leg_orders, 2)
        self.assertAlmostEqual(state.holdings[('A', 'B')][0], 0.5 / (1 - result.fee))

    def test_training_trials_match_an_independent_share_ledger(self):
        rng = np.random.default_rng(10)
        t, n = 21, 5
        prices = np.exp(rng.normal(0, 0.06, (t + 1, n)).cumsum(axis=0))
        available = rng.random((t, n)) > 0.27
        x, y = np.triu_indices(n, k=1)
        signal = rng.choice([-1.0, 0.0, 1.0], (t, len(x)))
        beta = rng.normal(size=signal.shape)
        ret = prices[1:] / prices[:-1] - 1
        book = TrainingBook(beta, signal, ret, x, y, available, prices)
        choices = [np.array([0, 2, 3]), np.array([0, 1, 4, 7])]
        totals = np.stack([book.num[:, ids].sum(axis=1) for ids in choices], axis=1)
        weights = np.stack([book.weight[:, ids].sum(axis=1) for ids in choices], axis=1)
        for f in (1.0, 0.48):
            for j, ids in enumerate(choices):
                wx, wy = pair_weights(beta[:, ids], signal[:, ids])
                targets = f * np.concatenate([wx, wy], axis=1)
                assets = np.r_[x[ids], y[ids]]
                shares = np.zeros(len(assets))
                nav = 1.0
                expected = []
                for day in range(t):
                    tradable = available[day, assets]
                    shares[tradable] = nav * targets[day, tradable] / prices[day, assets[tradable]]
                    pnl = shares @ (prices[day + 1, assets] - prices[day, assets])
                    expected.append(pnl / nav)
                    nav += pnl
                np.testing.assert_allclose(
                    book.returns(totals, weights, f)[:, j], expected, rtol=1e-12, atol=1e-15
                )
        full, f, scaled = book.allocation(0.0001)
        self.assertGreater(full, 0.0001)
        self.assertLess(f, 1.0)
        self.assertAlmostEqual(scaled, 0.0001, places=13)


class SearchStopTests(unittest.TestCase):
    def test_a_fourth_pass_changes_the_solution(self):
        # Each feasible improvement unlocks a higher-ranked edge already visited in that scan.
        loadings = np.array([0.0, 1.0, -2.0, 3.0, -4.0, 5.0, 100.0])
        score = np.arange(len(loadings), dtype=float)
        x = np.arange(0, 14, 2)
        y = x + 1
        num = np.stack([-loadings, loadings])
        weight = np.ones_like(num)
        history = []
        result, budget, exact, moves = paper_overlap.risk_selection(
            x, y, score, num, weight, np.array([0, 1]), np.array([5, 6]), 14, swaps_per_pass=history
        )
        self.assertFalse(exact)
        self.assertEqual(history, [1, 1, 1, 1, 0])
        self.assertEqual(moves, 4)
        self.assertEqual(set(result), {4, 5})
        self.assertLessEqual(paper_overlap.training_variance(num, weight, result), budget)

    def test_direct_cap3_selection_needs_no_passes(self):
        edges = np.array([0, 1])
        history = []
        _, _, exact, moves = paper_overlap.risk_selection(
            edges,
            edges + 2,
            np.array([1.0, 2.0]),
            np.zeros((3, 2)),
            np.ones((3, 2)),
            edges,
            edges,
            4,
            swaps_per_pass=history,
        )
        self.assertTrue(exact)
        self.assertEqual(history, [])
        self.assertEqual(moves, 0)


if __name__ == '__main__':
    unittest.main()
