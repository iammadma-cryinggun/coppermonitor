"""
CCI策略回测系统 - 主程序
=======================
可以运行所有品种的回测和优化
"""

import os
import sys

def print_menu():
    print("\n" + "="*60)
    print("CCI策略回测系统 - 主菜单".center(60))
    print("="*60)
    print("\n请选择功能:")
    print("  1. 基础CCI回测（铜、锡）")
    print("  2. CCI多品种测试（全部6个品种）")
    print("  3. 铜参数优化")
    print("  4. 锡参数优化（网格搜索）")
    print("  5. 铅锌铝参数优化")
    print("  6. 黄金白银棉花优化")
    print("  7. 验证回测逻辑")
    print("  8. 显示所有结果摘要")
    print("  0. 退出")
    print("-"*60)

def run_backtest(script_name):
    """运行指定的回测脚本"""
    script_path = f"D:\\期货数据\\铜期货监控\\{script_name}"
    if os.path.exists(script_path):
        print(f"\n正在运行: {script_name}")
        print("-"*60)
        os.system(f'python "{script_path}"')
    else:
        print(f"\n[错误] 文件不存在: {script_name}")

def show_summary():
    """显示所有回测结果摘要"""
    print("\n" + "="*60)
    print("CCI策略优化结果摘要".center(60))
    print("="*60)

    results = """
┌─────────────────────────────────────────────────────────────┐
│ 品种   │ 推荐参数          │ 测试收益  │ 最大回撤  │ 评价   │
├─────────────────────────────────────────────────────────────┤
│ 白银 🌟│ CCI(22,MA13)     │ +609.45% │ -17.85%  │ 神级   │
│ 锡 🌟  │ CCI(24,MA5)      │ +287.93% │ -11.16%  │ 震撼   │
│ 黄金 ✅ │ CCI(18,MA11)     │ +217.19% │ -16.90%  │ 优秀   │
│ 铅 ✅  │ CCI(18,MA12)     │ +148.20% │ -12.02%  │ 良好   │
│ 铜 ✅  │ CCI(20,MA10)     │ +101.47% │ -14.89%  │ 良好   │
│ 镍     │ CCI(20,MA10)     │ +63.78%  │ -20.73%  │ 一般   │
│ 铝     │ CCI(20,MA10)     │ +63.78%  │ -20.73%  │ 一般   │
│ 锌     │ CCI(20,14)       │ +43.10%  │ -17.46%  │ 一般   │
│ 棉花 ❌ │ CCI(32,MA8)      │ -54.91% │ -57.91%  │ 失败   │
└─────────────────────────────────────────────────────────────┘

【推荐组合】
精英五品种（平均收益+272.85%）:
  白银 + 锡 + 黄金 + 铅 + 铜

优选七品种（平均收益+213.11%）:
  白银 + 锡 + 黄金 + 铅 + 铜 + 镍 + 铝

【关键发现】
✓ 回测逻辑正确：没有未来函数，成交价格真实
✓ 高收益原因：2024-2026年商品大牛市
  白银涨幅+392%，镍涨幅+317%
✓ CCI择时有效：比简单买入持有收益更高
✓ 参数优化有效：所有品种都有改善
⚠ 棉花不适合：下跌趋势，做多策略亏损

【注意事项】
⚠ 合约乘数：代码使用×5，需根据实际调整
⚠ 交易成本：回测未考虑手续费和滑点
⚠ 风险控制：实盘需设置止损和仓位管理
"""
    print(results)

def main():
    while True:
        print_menu()
        choice = input("\n请输入选项 (0-8): ").strip()

        if choice == '0':
            print("\n退出系统")
            break
        elif choice == '1':
            run_backtest('CCI指标回测.py')
        elif choice == '2':
            run_backtest('CCI多品种测试.py')
        elif choice == '3':
            run_backtest('铜_CCI参数优化.py')
        elif choice == '4':
            run_backtest('锡_网格参数优化.py')
        elif choice == '5':
            run_backtest('铅锌铝_CCI参数优化.py')
        elif choice == '6':
            run_backtest('优化黄金白银棉花.py')
        elif choice == '7':
            run_backtest('验证回测逻辑.py')
        elif choice == '8':
            show_summary()
        else:
            print("\n[错误] 无效选项，请重新输入")

        input("\n按Enter键继续...")

if __name__ == "__main__":
    main()
