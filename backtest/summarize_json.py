import json
import pandas as pd

with open('backtest/monte_carlo_results_2020_2026.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for sym, res in data.items():
    df = pd.DataFrame(res)
    win_cnt = int((df['net_profit'] > 0).sum())
    loss_cnt = int((df['net_profit'] < 0).sum())
    print(f"=== {sym} DETAILED STATS (40 Windows, 2020-2026) ===")
    print("Total Windows:", len(df))
    print(f"Profitable Windows (>0): {win_cnt} ({win_cnt/len(df)*100:.1f}%)")
    print(f"Loss Windows (<0): {loss_cnt} ({loss_cnt/len(df)*100:.1f}%)")
    print(f"Max Profit Window: ${df['net_profit'].max():.2f} (+{df['return_pct'].max():.2f}%)")
    print(f"Worst Loss Window: ${df['net_profit'].min():.2f} ({df['return_pct'].min():.2f}%)")
    print(f"Average Net Profit / window: ${df['net_profit'].mean():.2f}")
    print(f"Median Net Profit / window: ${df['net_profit'].median():.2f}")
    print(f"Avg Trades / window: {df['trades'].mean():.1f}")
    print(f"Avg Win Rate: {df['win_rate'].mean():.1f}%")
    print(f"Avg Max Drawdown: {df['max_drawdown_pct'].mean():.2f}%")
    print(f"Worst Drawdown across all 40 windows: {df['max_drawdown_pct'].max():.2f}%")
    print(f"Windows with >= 1% weekly profit: {int((df['weekly_return_pct'] >= 1.0).sum())}")
    print()
