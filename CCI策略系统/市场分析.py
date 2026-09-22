"""
快速市场分析 - 一键运行
======================
直接分析当前市场信号
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

from 实时监控系统 import monitor_all_symbols
from 最优参数配置 import RECOMMENDED_SYMBOLS

def analyze_market():
    """一键分析当前市场"""
    print("\n" + "="*120)
    print("正在获取实时行情数据...".center(120))
    print("="*120)

    # 分析推荐品种：铜、纯碱、镍
    symbols = RECOMMENDED_SYMBOLS['稳健'] + RECOMMENDED_SYMBOLS['激进']

    monitor_all_symbols(symbols)


if __name__ == "__main__":
    analyze_market()
