"""Build the two-market performance table from the completed daily backtests."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.trading import summarize_original_style


ROOT = Path(__file__).resolve().parent / 'results'


def market_rows(market, folder):
    baseline = json.loads((folder / 'paper/run.json').read_text())
    overlap = json.loads((folder / 'paper_overlap/run.json').read_text())
    for key in ('price_sha256', 'first_date', 'last_date', 'trading_days', 'months', 'transaction_cost'):
        if baseline[key] != overlap[key]:
            raise ValueError(f'{market}: paper and overlap runs differ in {key}')
    rows = []
    calendar = None
    for suite, key, names in [('paper', 'strategy', ('MZ', 'BZ')),
                              ('paper_overlap', 'model', ('CAP3', 'RISK3'))]:
        data = pd.read_csv(folder / suite / 'daily.csv', parse_dates=['date'])
        for name in names:
            returns = data[data[key].eq(name)]
            dates = pd.DatetimeIndex(returns.date)
            if (len(dates) != baseline['trading_days'] or not dates.is_unique
                    or not dates.is_monotonic_increasing
                    or (calendar is not None and not calendar.equals(dates))):
                raise ValueError(f'{market}: incomplete daily returns for {name}')
            calendar = dates
            row = {'market': market, 'strategy': name, 'first_date': baseline['first_date'],
                   'last_date': baseline['last_date'], 'months': baseline['months'],
                   'trading_days': len(returns), 'one_way_bps': baseline['transaction_cost']['one_way_bps']}
            stats = {basis: summarize_original_style(returns[basis + '_return']) for basis in ('gross', 'net')}
            for metric in ('cumulative_return', 'annualized_return', 'sharpe'):
                for basis in ('gross', 'net'):
                    row[f'{basis}_{metric}'] = stats[basis][metric]
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sp500', type=Path, default=ROOT)
    parser.add_argument('--kospi', type=Path, default=ROOT / 'kospi')
    parser.add_argument('--output', type=Path, default=ROOT / 'market_comparison.csv')
    args = parser.parse_args()
    frame = pd.DataFrame(market_rows('S&P 500', args.sp500) + market_rows('KOSPI', args.kospi))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
