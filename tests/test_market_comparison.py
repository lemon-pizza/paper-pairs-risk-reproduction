"""Input and performance regressions for comparisons across markets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src import paper, paper_overlap
from src.trading import summarize_original_style


class MarketComparisonTests(unittest.TestCase):
    def test_long_prices_preserve_tickers_and_missing_quotes(self):
        with TemporaryDirectory() as tmp:
            csv = Path(tmp) / 'prices.csv'
            csv.write_text('date,ticker,adjusted_price,volume\n'
                           '2017-01-03,000020,120,0\n'
                           '2017-01-02,000030,95,1\n'
                           '2017-01-02,000020,100,2\n')
            prices, _ = paper.load_prices(csv)
            self.assertEqual(list(prices.columns), ['000020', '000030'])
            np.testing.assert_allclose(prices.to_numpy(), [[100, 95], [120, np.nan]])

    def test_parquet_identifiers_round_trip_through_selection_csv(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'prices.parquet'
            panel = pd.DataFrame([[100, 95]], index=pd.to_datetime(['2017-01-02']), columns=[20, 30])
            panel.to_parquet(path)
            prices, _ = paper.load_prices(path)
            self.assertEqual(list(prices.columns), ['20', '30'])

    def test_full_risk_history_must_precede_the_trading_sample(self):
        history_days = paper.WINDOW + paper_overlap.TRAIN_DAYS
        dates = pd.bdate_range(end='2016-12-30', periods=history_days)
        prices = pd.DataFrame({'000020': 100.}, index=dates.append(pd.DatetimeIndex(['2017-01-02'])))
        self.assertEqual(len(paper.trading_sample(prices, history_days)), 1)
        with self.assertRaisesRegex(ValueError, '756 required'):
            paper.trading_sample(prices.iloc[1:], history_days)

    def test_overlap_uses_the_same_trade_day_hedges_as_the_paper_replay(self):
        rng = np.random.default_rng(8)
        first = paper.WINDOW + paper_overlap.TRAIN_DAYS
        n = first + 12
        walk = rng.normal(0, 1e-5, n).cumsum()
        logs = np.column_stack([np.log(1e5) + walk,
                                np.log(8e4) + 1.1 * walk + rng.normal(0, 1e-4, n)])
        logs[first - 1::2, 1] += .003
        prices = pd.DataFrame(np.exp(logs), index=pd.bdate_range('2013-01-02', periods=n),
                              columns=['000020', '000030'])
        panel = prices.iloc[paper_overlap.TRAIN_DAYS:]
        x, y, t = paper.formation_graph(panel.iloc[:paper.WINDOW].to_numpy())
        source = pd.DataFrame({'asset_x': panel.columns[x], 'asset_y': panel.columns[y], 'adf_t': t})
        _, _, _, positions = paper_overlap.run_month(prices, source)
        beta, signals = paper.daily_signals(np.log(panel.iloc[:-1].to_numpy()), x, y)
        wx, wy = paper.pair_weights(beta, signals['Z'])
        self.assertTrue(np.any(wx))
        np.testing.assert_array_equal(positions['M'].x_weight, wx)
        np.testing.assert_array_equal(positions['M'].y_weight, wy)

    def test_annualized_return_is_geometric_and_sharpe_uses_daily_returns(self):
        r = np.tile([.02, -.01, -.005, .001], 126)
        stats = summarize_original_style(r)
        factor = np.prod(1 + r)
        self.assertAlmostEqual(stats['cumulative_return'], factor - 1)
        self.assertAlmostEqual(stats['annualized_return'], np.sqrt(factor) - 1)
        self.assertAlmostEqual(stats['sharpe'], np.sqrt(252) * r.mean() / r.std(ddof=1))


if __name__ == '__main__':
    unittest.main()
