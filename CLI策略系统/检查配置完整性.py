"""
检查配置文件完整性
"""
import re

# 读取原始配置文件
with open('D:/期货数据/铜期货监控/CCI策略系统/最优参数配置.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有品种名称
symbols = re.findall(r"'([^']+)':\s+\{", content)
original_symbols = symbols

print("="*80)
print("配置完整性检查")
print("="*80)

print(f"\n[原始配置文件] 找到 {len(original_symbols)} 个品种:")
for s in original_symbols:
    print(f"  - {s}")

# 检查实时监控独立版中的品种
with open('D:/期货数据/铜期货监控/CCI策略系统/实时监控_独立版.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取监控文件中的品种
monitor_symbols = re.findall(r'"\[?\'([A-Za-z]+)\]\'?:", content)
monitor_symbols_list = list(set(monitor_symbols))

print(f"\n[监控系统] 包含 {len(monitor_symbols_list)} 个品种:")
for s in monitor_symbols_list:
    print(f"  - {s}")

# 对比
missing = set(original_symbols) - set(monitor_symbols_list)
extra = set(monitor_symbols_list) - set(original_symbols)

if missing:
    print(f"\n[遗漏] {len(missing)} 个品种:")
    for s in missing:
        print(f"  - {s}")

if extra:
    print(f"\n[额外] {len(extra)} 个品种:")
    for s in extra:
        print(f"  - {s}")

print(f"\n总计: 原始{len(original_symbols)}个, 监控{len(monitor_symbols_list)}个")
print("="*80)
