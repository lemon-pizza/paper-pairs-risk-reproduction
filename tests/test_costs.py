"""Small cash-flow examples for the one-way execution-cost convention."""
import unittest

import numpy as np
import pandas as pd

from src.costs import CostState, PairPositions, charge_costs, pair_weights, transaction_costs
from src.paper import portfolio_returns


PAIR = ('AAA', 'BBB')


def block(wx, wy, rx=None, ry=None, pairs=(PAIR,), start='2020-01-01'):
    wx, wy = np.asarray(wx, float), np.asarray(wy, float)
    if wx.ndim == 1:
        wx, wy = wx[:, None], wy[:, None]
    rx = np.zeros_like(wx) if rx is None else np.asarray(rx, float).reshape(wx.shape)
    ry = np.zeros_like(wy) if ry is None else np.asarray(ry, float).reshape(wy.shape)
    return PairPositions(pd.date_range(start, periods=len(wx)), pairs, wx, wy, rx, ry)


class ExecutionCostTests(unittest.TestCase):
    def test_entry_and_terminal_exit_are_both_charged(self):
        state = CostState()
        fees, rebalance, closing = transaction_costs(block([.5], [-.5]), [0.], state, True)
        np.testing.assert_allclose([fees[0], rebalance[0], closing[0]], [.0006, 1., 1.])
        self.assertEqual(state.holdings, {})

    def test_unchanged_holdings_have_no_new_cost(self):
        state = CostState()
        transaction_costs(block([.5], [-.5]), [0.], state)
        # Keep the same shares: fees have reduced equity, so weights are larger.
        carried = .5 / .9997
        fees, rebalance, _ = transaction_costs(
            block([carried], [-carried], start='2020-01-02'), [0.], state)
        np.testing.assert_allclose([fees[0], rebalance[0]], [0., 0.], atol=1e-15)

    def test_reversal_pays_close_and_open(self):
        state = CostState({PAIR: (.5, -.5)})
        fees, rebalance, _ = transaction_costs(block([-.5], [.5]), [0.], state)
        np.testing.assert_allclose([fees[0], rebalance[0]], [.0006, 2.])

    def test_exit_is_charged_on_a_day_without_active_pairs(self):
        state = CostState({PAIR: (.5, -.5)})
        fees, rebalance, _ = transaction_costs(block([0.], [0.]), [0.], state)
        np.testing.assert_allclose([fees[0], rebalance[0]], [.0003, 1.])
        self.assertEqual(state.holdings, {})

    def test_terminal_trade_uses_end_prices(self):
        fees, rebalance, closing = transaction_costs(
            block([.5], [-.5], [.2], [0.]), [.1], CostState(), True)
        np.testing.assert_allclose([fees[0], rebalance[0], closing[0]], [.00063, 1., 1.1])
        self.assertAlmostEqual(.1 - fees[0], .09937)

    def test_price_drift_and_costs_change_next_rebalance(self):
        state = CostState()
        transaction_costs(block([.5], [-.5], [.2], [0.]), [.1], state)
        fees, rebalance, _ = transaction_costs(
            block([.5], [-.5], start='2020-01-02'), [0.], state)
        traded = abs(.5 - .6 / 1.0997) + abs(-.5 + .5 / 1.0997)
        self.assertAlmostEqual(rebalance[0], traded)
        self.assertAlmostEqual(fees[0], .0003 * traded)

    def test_month_split_preserves_holdings_without_forced_exit(self):
        whole = block([.5, .5, 0.], [-.5, -.5, 0.], [.1, -.1, 0.], [0., .05, 0.])
        gross = np.array([.05, -.075, 0.])
        expected = transaction_costs(whole, gross, CostState(), True)
        state = CostState()
        first = block([.5], [-.5], [.1], [0.])
        second = block([.5, 0.], [-.5, 0.], [-.1, 0.], [.05, 0.], start='2020-01-02')
        one = transaction_costs(first, gross[:1], state)
        two = transaction_costs(second, gross[1:], state, True)
        for full, a, b in zip(expected, one, two):
            np.testing.assert_allclose(full, np.r_[a, b], atol=1e-15)

    def test_removed_pairs_close_without_cross_pair_netting(self):
        state = CostState({PAIR: (.5, -.5)})
        # A different pair sharing AAA is a separate sleeve, so both legs trade.
        fees, rebalance, _ = transaction_costs(
            block([.5], [-.5], pairs=(('AAA', 'CCC'),)), [0.], state)
        np.testing.assert_allclose([fees[0], rebalance[0]], [.0006, 2.])

    def test_pair_order_does_not_change_costs(self):
        state = CostState({PAIR: (.2, -.3), ('CCC', 'DDD'): (-.1, .4)})
        positions = block([[-.1, .2]], [[.4, -.3]], pairs=(('CCC', 'DDD'), PAIR))
        fees, _, _ = transaction_costs(positions, [0.], state)
        np.testing.assert_allclose(fees, [0.], atol=1e-15)

    def test_invalid_regressions_do_not_create_positions(self):
        beta = np.array([[2., -3., np.nan], [1., np.inf, 2.]])
        signal = np.array([[1., -1., 1.], [0., 1., 0.]])
        wx, wy = pair_weights(beta, signal)
        np.testing.assert_allclose(wx, [[1 / 3, 3 / 8, 0.], [0., 0., 0.]])
        np.testing.assert_allclose(wy, [[-1 / 6, 1 / 8, 0.], [0., 0., 0.]])
        rx, ry = np.full_like(beta, .02), np.full_like(beta, -.01)
        gross, _ = portfolio_returns(beta, signal, rx, ry)
        np.testing.assert_allclose(gross, (wx * rx + wy * ry).sum(axis=1))

    def test_fee_columns_reconcile_to_net_return(self):
        positions = block([.5], [-.5], [.2], [0.])
        daily = pd.DataFrame({'date': positions.dates, 'strategy': ['MZ'], 'gross_return': [.1]})
        result = charge_costs(daily, {'MZ': positions}, {}, liquidate=True)
        self.assertAlmostEqual(result.fee.iloc[0], .00063)
        self.assertAlmostEqual(result.net_return.iloc[0], .09937)
        self.assertAlmostEqual(result.turnover.iloc[0], 2.1)

    def test_invalid_inputs_fail_before_charging(self):
        with self.assertRaisesRegex(ValueError, 'Leg P&L'):
            transaction_costs(block([.5], [-.5]), [.1], CostState())
        with self.assertRaisesRegex(ValueError, 'finite'):
            transaction_costs(block([np.nan], [-.5]), [0.], CostState())
        with self.assertRaisesRegex(ValueError, 'previous block'):
            transaction_costs(block([.5], [-.5]), [0.], CostState(last_date=pd.Timestamp('2020-01-01')))


if __name__ == '__main__':
    unittest.main()
