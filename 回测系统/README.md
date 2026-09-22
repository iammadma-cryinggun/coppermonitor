# LME铜回测系统

## 文件结构

```
回测系统/
├── 原始数据/
│   └── data_pool.pkl           # 所有品种的日线数据
├── 程序/
│   ├── data_cleaning_factory.py # 数据清洗程序
│   ├── 回测.py                    # 回测程序
│   └── 贝叶斯优化.py              # 参数优化程序
└── 优化结果/
    ├── cu_optimized_params.json  # 最优参数
    ├── 交易明细.csv               # 详细交易记录
    └── 资金曲线.csv               # 资金变化曲线
```

## 使用说明

### 1. 数据准备
```bash
cd 程序
python data_cleaning_factory.py
```
这会从 `global_futures_daily/` 读取原始数据并生成 `data_pool.pkl`

### 2. 参数优化
```bash
cd 程序
python 贝叶斯优化.py
```
使用贝叶斯优化对全部12个参数进行优化（约7-8分钟）

### 3. 回测验证
```bash
cd 程序
python 回测.py
```
使用优化后的参数进行详细回测，显示所有交易明细

## 策略参数（已优化）

```
EMA_FAST:        10
EMA_SLOW:        12
MACD_FAST:       11
MACD_SLOW:       23
MACD_SIGNAL:     11
RSI_PERIOD:      17
RSI_FILTER:      47
RATIO_TRIGGER:   1.229
STC_LENGTH:      9
STC_FAST:        24
STC_SLOW:        44
STC_SELL_ZONE:   72
```

## 回测结果（修复bug后）

> 以下为修复 look-ahead 入场 + 加入手续费/滑点 后的真实结果（2026-09-04 重跑）：
> 修复前数字（+88.36%/-11.42%/92笔）含同K线收盘价入场 + 零成本，已废弃。

- 总收益率：+67.72%
- 最大回撤：-5.18%
- 交易次数：125笔
- 胜率：60.0%
- 平均杠杆：约0.57倍
- 成本模型：手续费 0.03%/边 + 滑点 0.02%/边（round-trip ≈ 0.10%）

## 交易规则

### 入场条件
- **Sniper信号**：趋势向上 + ratio收缩 + MACD向上 + RSI强势
- **Chase信号**：EMA金叉 + RSI强势

### 出场条件
- 止损：2%固定止损
- 止盈：STC掉头或趋势反转
- 风控：回撤超50%或资金低于30%停止交易

### 仓位管理
- 根据ratio值动态调整（1.0x ~ 2.0x）
- 资金低于50%时减半仓位
- 限制最大杠杆3倍

## 重要说明

本系统已修复以下bug：
1. ✅ 去除内层循环（避免时间穿越和重复交易）
2. ✅ 修正资金状况判断顺序
3. ✅ 修正actual_leverage计算
4. ✅ 优化保证金计算逻辑
5. ✅ 去除入场 look-ahead（信号同bar收盘 → 次bar开盘成交）
6. ✅ 加入手续费 0.03%/边 + 滑点 0.02%/边
7. ✅ 止损成交价改为 max(次日开盘, 止损价)（跳空更差成交）

## 数据来源

- LME铜日线数据（2016-2026）
- COMEX/ICE国际期货日线数据
- 原始数据位于：`global_futures_daily/`

## 版本历史

- v1.0: 初始版本（有bug）
- v2.0: 修复所有bug，回测结果可信（含 look-ahead + 成本修复，2026-09-04）
