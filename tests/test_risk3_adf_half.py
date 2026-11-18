import unittest

import numpy as np
import pandas as pd

from src import paper, risk3_adf_half as half
from src.costs import pair_weights


class Risk3AdfHalfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        cls.days = 5
        n = half.HISTORY + cls.days
        logs = 5 + rng.normal(0., .003, (n, 4)).cumsum(axis=0)
        logs[half.WINDOW::11, 1] += .09
        logs[half.HISTORY - 1::2, 3] -= .15
        names = ['000001', '000002', '000003', '000004']
        cls.prices = pd.DataFrame(np.exp(logs), index=pd.bdate_range('2013-01-02', periods=n), columns=names)
        cls.parent = pd.DataFrame([
            (names[0], names[2], 7.), (names[1], names[3], 8.),
            (names[1], names[2], 6.), (names[0], names[1], 7.), (names[0], names[3], 5.),
        ], columns=['asset_x', 'asset_y', 'adf_score'])
        cls.budget = 1e-6

    def test_variance_scaling_only_reduces_exposure(self):
        self.assertAlmostEqual(half.capital_fraction(.04, .01), .5)
        self.assertEqual(half.capital_fraction(.01, .04), 1.)
        self.assertEqual(half.capital_fraction(0., 0.), 1.)
        self.assertEqual(half.capital_fraction(.01, 0.), 0.)
        for variance, budget in [(np.nan, 1.), (1., -1.)]:
            with self.assertRaises(ValueError):
                half.capital_fraction(variance, budget)

    def test_floor_ties_and_training_window_independent_of_future_prices(self):
        kept, info, book = half.run_month(self.prices, self.parent, self.budget, self.days)
        expected = [('000002', '000004'), ('000001', '000002')]
        self.assertEqual(list(zip(kept.asset_x, kept.asset_y)), expected)
        self.assertEqual(info['selected_pairs'], info['parent_pairs'] // 2)
        self.assertEqual(kept['rank'].tolist(), [1, 2])
        self.assertLess(info['capital_fraction'], 1.)
        self.assertAlmostEqual(info['scaled_training_variance'], self.budget)
        # Replaying the top half renormalizes its active pairs before applying the monthly allocation.
        x = self.prices.columns.get_indexer(kept.asset_x)
        y = self.prices.columns.get_indexer(kept.asset_y)
        logs = np.log(self.prices.to_numpy()[half.TRAIN_DAYS:half.HISTORY + self.days - 1])
        beta, signals = paper.daily_signals(logs, x, y)
        wx, wy = pair_weights(beta, signals['Z'])
        self.assertTrue((np.abs(wx) + np.abs(wy)).any())
        np.testing.assert_allclose(book[half.MODEL, 'signal_close'].x_weight, wx * info['capital_fraction'])
        np.testing.assert_allclose(book[half.MODEL, 'signal_close'].y_weight, wy * info['capital_fraction'])

        changed = self.prices.copy()
        changed.iloc[half.HISTORY:, 1] *= 1.8
        new_kept, new_info, new_book = half.run_month(changed, self.parent, self.budget, self.days)
        pd.testing.assert_frame_equal(kept, new_kept)
        self.assertEqual(info, new_info)
        np.testing.assert_allclose(new_book[half.MODEL, 'signal_close'].x_weight[0],
                                   book[half.MODEL, 'signal_close'].x_weight[0])

    def test_signal_close_uses_daily_returns_through_month_end(self):
        _, _, book = half.run_month(self.prices, self.parent, self.budget, self.days)
        self.assertEqual(set(book), {(half.PARENT, 'signal_close'), (half.MODEL, 'signal_close')})
        dates = self.prices.index[half.HISTORY:]
        returns = self.prices.pct_change(fill_method=None).loc[dates]
        for block in book.values():
            pd.testing.assert_index_equal(block.dates, dates)
            x, y = zip(*block.pairs)
            np.testing.assert_allclose(block.x_return, returns.loc[:, list(x)])
            np.testing.assert_allclose(block.y_return, returns.loc[:, list(y)])

    def test_single_parent_pair_leaves_top_half_in_cash(self):
        kept, info, book = half.run_month(self.prices, self.parent.iloc[:1], 0., self.days)
        self.assertTrue(kept.empty)
        self.assertEqual(info['scaled_training_variance'], 0.)
        self.assertEqual(book[half.MODEL, 'signal_close'].x_weight.shape, (self.days, 0))


if __name__ == '__main__':
    unittest.main()
