"""分析回测中不同退出方式的使用情况和盈利效果"""
import pandas as pd
import sys
sys.path.append('D:/期货数据/铜期货监控/CCI策略系统')

from 完整回测_量能优化_akshare import CompleteBacktest

# 测试品种配置
configs = {
    'AG0': {  # 白银
        'cci_length': 15, 'ma_length': 15, 'cci_oversold': -60,
        'cci_overbought': 190, 'cci_cross_max': 180,
        'stc_oversold': -50, 'stc_cross': -100, 'multiplier': 15,
        'vol_threshold': 0.65
    },
    'FG0': {  # 玻璃
        'cci_length': 14, 'ma_length': 8, 'cci_oversold': -60,
        'cci_overbought': 100, 'cci_cross_max': 150,
        'stc_oversold': -40, 'stc_cross': -70, 'multiplier': 20,
        'vol_threshold': 0.90
    },
    'SA0': {  # 纯碱
        'cci_length': 12, 'ma_length': 5, 'cci_oversold': -40,
        'cci_overbought': 190, 'cci_cross_max': 180,
        'stc_oversold': -50, 'stc_cross': -130, 'multiplier': 20,
        'vol_threshold': 0.50
    }
}

print("=" * 80)
print("退出方式分析报告")
print("=" * 80)

for symbol, params in configs.items():
    print(f"\n【{symbol}】")
    bt = CompleteBacktest(symbol, params)
    bt.run()

    # 统计退出方式
    trades_df = pd.DataFrame(bt.trades)
    if len(trades_df) > 0:
        exit_stats = trades_df['type'].value_counts()
        print(f"\n退出方式统计（共{len(trades_df)}笔交易）:")
        print(f"  {'方式':<15} {'次数':>8} {'占比':>10} {'平均盈亏':>12}")
        print("-" * 50)

        for exit_type, count in exit_stats.items():
            pct = count / len(trades_df) * 100
            avg_pnl = trades_df[trades_df['type'] == exit_type]['pnl'].mean()
            name_map = {
                'stop_loss': '止损',
                'take_profit': '固定止盈(20%)',
                'overbought': 'CCI超买',
                'death_cross': 'CCI死叉'
            }
            print(f"  {name_map.get(exit_type, exit_type):<15} {count:>8} {pct:>9.1f}% {avg_pnl:>11.0f}")

        # 总体统计
        total_pnl = trades_df['pnl'].sum()
        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] < 0]
        print(f"\n总体表现:")
        print(f"  总盈亏: {total_pnl:.0f}")
        print(f"  盈利笔数: {len(win_trades)} ({len(win_trades)/len(trades_df)*100:.1f}%)")
        print(f"  亏损笔数: {len(lose_trades)} ({len(lose_trades)/len(trades_df)*100:.1f}%)")

print("\n" + "=" * 80)
