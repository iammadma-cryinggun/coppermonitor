"""
快速测试监控系统 - 只测试前3个品种
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

# 使用快速测试
exec(open('实时监控系统_论文因子版.py', 'r', encoding='utf-8').read())

# 只测试前3个品种
quick_monitor(top_n=3)