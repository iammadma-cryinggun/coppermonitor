# -*- coding: utf-8 -*-
"""
===================================
沪铜策略 - 实盘监控与信号记录（实时数据版 + Telegram通知）
===================================

功能:
1. 获取最新数据（实时API + CSV备用）
2. 运行策略生成交易信号
3. 记录买卖建议到日志
4. 发送Telegram通知（每4小时）
5. 跟踪策略表现 vs 实际表现

数据源:
- 主: ChinaFuturesFetcher (AkShare API) - 实时数据
- 备: 本地CSV文件 - 离线备用
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

# 导入本地模块
from china_futures_fetcher import ChinaFuturesFetcher
from notifier import get_notifier

# ==========================================
# 配置
# ==========================================

# 策略参数（使用策略B实盘版的最优参数）
EMA_FAST = 5
EMA_SLOW = 15
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_FILTER = 45
RATIO_TRIGGER = 1.15
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STC_SELL_ZONE = 85
STOP_LOSS_PCT = 0.02

# 期货配置
FUTURES_CODE = 'CU'  # 沪铜期货代码
HISTORICAL_DAYS = 300  # 获取历史数据天数

# 基础路径（自动检测脚本所在目录）
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / 'logs'
CONFIG_DIR = BASE_DIR / 'config'

# 数据路径（备用）- 相对路径
BACKUP_DATA_PATH = BASE_DIR / 'data' / 'backup.csv'
SIGNAL_LOG_PATH = LOGS_DIR / 'signal_log.json'
TRACKING_PATH = LOGS_DIR / 'performance_tracking.csv'
LOG_FILE = LOGS_DIR / 'strategy_monitor.log'

# 确保日志目录存在
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据获取器
fetcher = ChinaFuturesFetcher()

# Telegram通知器
telegram_notifier = get_notifier()

# 仓位管理参数
def calculate_position_size(ratio, rsi):
    """动态仓位计算（实盘逻辑）"""
    if ratio > 2.0:
        return 2.0
    elif ratio > 1.5:
        return 1.5
    elif ratio > 1.0:
        return 1.2
    else:
        return 1.0


def calculate_indicators(df):
    """计算技术指标"""
    # EMA
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    # MACD & Ratio
    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC
    stc_macd = df['close'].ewm(span=STC_FAST, adjust=False).mean() - \
               df['close'].ewm(span=STC_SLOW, adjust=False).mean()
    stoch_period = STC_LENGTH
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    stoch_k = 100 * (stc_macd - min_macd) / (max_macd - min_macd).replace(0, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    stc_raw = 100 * (stoch_d - min_stoch_d) / (max_stoch_d - min_stoch_d).replace(0, np.nan)
    stc_raw = stc_raw.fillna(50)
    df['stc'] = stc_raw.rolling(window=3).mean()

    # 波动率（仅供参考）
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=20).mean()
    df['volatility'] = df['atr'] / df['close']

    return df


def check_signals(df):
    """
    检查交易信号（实盘逻辑）

    Returns:
        dict: {
            'datetime': 最新K线时间,
            'price': 最新价格,
            'indicators': {各项指标},
            'buy_signal': 买入信号（bool）,
            'sell_signal': 卖出信号（bool）,
            'signal_type': 信号类型,
            'position_size': 建议仓位,
            'stop_loss': 止损价,
            'reason': 信号原因
        }
    """
    if len(df) < 200:
        return {'error': '数据不足，需要至少200根K线'}

    # 获取最新数据
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 预计算前值
    df['ratio_prev'] = df['ratio'].shift(1)
    df['stc_prev'] = df['stc'].shift(1)

    # 信号条件
    trend_up = latest['ema_fast'] > latest['ema_slow']
    ratio_safe = (0 < latest['ratio'] < RATIO_TRIGGER)
    ratio_shrinking = latest['ratio'] < prev['ratio']
    turning_up = latest['macd_dif'] > prev['macd_dif']
    is_strong = latest['rsi'] > RSI_FILTER

    ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (latest['ema_fast'] > latest['ema_slow'])

    sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
    chase_signal = ema_cross and is_strong

    # 买入信号
    buy_signal = sniper_signal or chase_signal
    buy_reason = 'sniper' if sniper_signal else ('chase' if chase_signal else None)

    # 卖出信号（需要持仓状态）
    stc_exit = (df['stc_prev'].iloc[-1] > STC_SELL_ZONE) and (latest['stc'] < df['stc_prev'].iloc[-1])
    trend_exit = latest['ema_fast'] < latest['ema_slow']
    sell_signal = stc_exit or trend_exit
    sell_reason = 'stc' if stc_exit else ('trend' if trend_exit else None)

    # 计算建议仓位（用上一根Ratio，实盘逻辑）
    position_size = calculate_position_size(df['ratio_prev'].iloc[-1], latest['rsi'])

    # 止损价（如果有买入信号）
    stop_loss = latest['close'] * (1 - STOP_LOSS_PCT) if buy_signal else None

    return {
        'datetime': str(latest['datetime']),
        'price': float(latest['close']),
        'indicators': {
            'ema_fast': float(latest['ema_fast']),
            'ema_slow': float(latest['ema_slow']),
            'macd_dif': float(latest['macd_dif']),
            'macd_dea': float(latest['macd_dea']),
            'ratio': float(latest['ratio']),
            'ratio_prev': float(df['ratio_prev'].iloc[-1]),
            'rsi': float(latest['rsi']),
            'stc': float(latest['stc']),
            'stc_prev': float(df['stc_prev'].iloc[-1]),
            'volatility': float(latest['volatility'])
        },
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'signal_type': buy_reason if buy_signal else (sell_reason if sell_signal else None),
        'position_size': position_size,
        'stop_loss': stop_loss,
        'reason': {
            'buy': buy_reason,
            'sell': sell_reason
        },
        'trend': 'up' if trend_up else 'down',
        'strength': 'strong' if latest['ratio'] > 1.5 else ('normal' if latest['ratio'] > 1.0 else 'weak')
    }


def save_signal_log(signal, status='pending', actual_price=None, actual_action=None):
    """
    保存信号到日志

    Args:
        signal: 信号字典
        status: 状态 (pending/executed/ignored)
        actual_price: 实际成交价
        actual_action: 实际操作 (buy/sell/hold)
    """
    log_path = Path(SIGNAL_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取现有日志
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []

    # 添加新信号
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'signal_datetime': signal['datetime'],
        'signal': signal,
        'status': status,
        'actual_price': actual_price,
        'actual_action': actual_action,
        'notes': ''
    }
    logs.append(log_entry)

    # 保存日志
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    logger.info(f"信号已记录: {signal.get('signal_type', 'none')} | 状态: {status}")


def load_position_status():
    """加载当前持仓状态"""
    status_path = LOGS_DIR / 'position_status.json'

    if status_path.exists():
        with open(status_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            'holding': False,
            'entry_price': None,
            'entry_datetime': None,
            'position_size': 1.0,
            'stop_loss': None,
            'signal_id': None
        }


def save_position_status(status):
    """保存持仓状态"""
    status_path = LOGS_DIR / 'position_status.json'
    status_path.parent.mkdir(parents=True, exist_ok=True)

    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def load_market_data():
    """
    加载市场数据（实时API + CSV备用）

    优先级:
    1. ChinaFuturesFetcher (AkShare API) - 实时数据
    2. 本地CSV文件 - 离线备用
    """
    logger.info("=" * 80)
    logger.info("获取市场数据")
    logger.info("=" * 80)

    # 尝试使用实时API
    logger.info(f"[数据源] 尝试从 API 获取 {FUTURES_CODE} (沪铜) 实时数据...")
    df = fetcher.get_historical_data(FUTURES_CODE, days=HISTORICAL_DAYS)

    if df is not None and not df.empty:
        logger.info(f"[API] 成功获取 {len(df)} 条记录")
        logger.info(f"[API] 数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

        # 标准化列名以匹配策略格式
        df = df.rename(columns={
            'date': 'datetime',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '持仓量': 'hold'
        })

        # 确保datetime格式
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 按日期排序
        df = df.sort_values('datetime').reset_index(drop=True)

        return df, 'API'

    # API失败，使用备用CSV文件
    logger.warning(f"[API] 获取失败，尝试使用备用CSV文件...")
    csv_path = Path(BACKUP_DATA_PATH)

    if csv_path.exists():
        logger.info(f"[CSV] 使用备用数据源: {BACKUP_DATA_PATH}")
        try:
            df = pd.read_csv(BACKUP_DATA_PATH)
            df.columns = [c.strip() for c in df.columns]
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.info(f"[CSV] 成功加载 {len(df)} 条记录")
            logger.info(f"[CSV] 数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
            logger.warning(f"[CSV] 注意: CSV数据可能不是最新的，建议更新数据")
            return df, 'CSV'
        except Exception as e:
            logger.error(f"[CSV] 备用数据源也加载失败: {e}")
            return None, None
    else:
        logger.error(f"[CSV] 备用文件不存在: {BACKUP_DATA_PATH}")
        return None, None


def run_monitoring():
    """运行监控"""
    logger.info("=" * 80)
    logger.info("沪铜策略 - 实盘监控（实时数据版）")
    logger.info("=" * 80)

    # 加载数据
    df, data_source = load_market_data()

    if df is None:
        logger.error("数据加载失败，退出监控")
        return

    logger.info(f"\n[数据源] 使用: {data_source}")
    logger.info(f"[数据] 范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]} ({len(df)} 根)")

    # 计算指标
    df = calculate_indicators(df)

    # 检查信号
    signal = check_signals(df)

    # 获取持仓状态
    position = load_position_status()

    # 输出当前状态
    logger.info("\n" + "=" * 80)
    logger.info("当前市场状态")
    logger.info("=" * 80)
    logger.info(f"时间: {signal['datetime']}")
    logger.info(f"价格: {signal['price']:.0f}")
    logger.info(f"趋势: {signal['trend']} ({signal['strength']})")
    logger.info(f"Ratio: {signal['indicators']['ratio']:.2f} (上一根: {signal['indicators']['ratio_prev']:.2f})")
    logger.info(f"RSI: {signal['indicators']['rsi']:.1f}")
    logger.info(f"STC: {signal['indicators']['stc']:.1f} (上一根: {signal['indicators']['stc_prev']:.1f})")
    logger.info(f"波动率: {signal['indicators']['volatility']*100:.2f}%")

    # 信号提示
    logger.info("\n" + "=" * 80)
    logger.info("交易信号")
    logger.info("=" * 80)

    if position['holding']:
        logger.info(f"当前持仓: 是")
        logger.info(f"  入场价: {position['entry_price']:.0f}")
        logger.info(f"  入场时间: {position['entry_datetime']}")
        logger.info(f"  仓位: {position['position_size']:.1f}x")
        logger.info(f"  止损价: {position['stop_loss']:.0f}")

        # 当前盈亏
        current_pnl = (signal['price'] - position['entry_price']) * position['position_size']
        current_pnl_pct = (signal['price'] - position['entry_price']) / position['entry_price'] * 100
        logger.info(f"  当前盈亏: {current_pnl:.0f} 点 ({current_pnl_pct:+.2f}%)")

        # 检查止损
        if signal['price'] <= position['stop_loss']:
            logger.warning(f"[止损触发] 当前价格 {signal['price']:.0f} <= 止损价 {position['stop_loss']:.0f}")
            signal['sell_signal'] = True
            signal['signal_type'] = 'stop_loss'

        # 检查卖出信号
        if signal['sell_signal']:
            logger.info(f"[卖出信号] 类型: {signal['reason']['sell']}")
            logger.info(f"  建议操作: 卖出全部仓位")
            logger.info(f"  预期盈亏: {current_pnl:.0f} 点")

            # 记录信号
            save_signal_log(signal, status='pending')

            # 清空持仓（需要确认）
            logger.info("\n[操作建议] 请确认是否卖出:")
            logger.info(f"  1. 卖出全部仓位")
            logger.info(f"  2. 更新持仓状态为空仓")

    else:
        logger.info("当前持仓: 否")

        if signal['buy_signal']:
            logger.info(f"[买入信号] 类型: {signal['reason']['buy']}")
            logger.info(f"  建议仓位: {signal['position_size']:.1f}x")
            logger.info(f"  止损价格: {signal['stop_loss']:.0f} ({STOP_LOSS_PCT*100}%)")

            # 记录信号
            save_signal_log(signal, status='pending')

            logger.info("\n[操作建议] 请确认是否买入:")
            logger.info(f"  1. 开仓 {signal['position_size']:.1f}x")
            logger.info(f"  2. 入场价: {signal['price']:.0f}")
            logger.info(f"  3. 止损价: {signal['stop_loss']:.0f}")
            logger.info(f"  4. 更新持仓状态")

            # 自动更新持仓状态（需要用户确认）
            new_position = {
                'holding': True,
                'entry_price': signal['price'],
                'entry_datetime': signal['datetime'],
                'position_size': signal['position_size'],
                'stop_loss': signal['stop_loss'],
                'signal_id': f"{signal['datetime']}_{signal.get('signal_type', 'manual')}"
            }
            logger.info(f"\n[自动保存] 新持仓状态已准备: {new_position['signal_id']}")
            logger.info(f"  (请手动执行交易后，更新持仓状态)")

        else:
            logger.info("无买入信号，继续观望")

    # 保存监控记录
    tracking_path = Path(TRACKING_PATH)
    tracking_path.parent.mkdir(parents=True, exist_ok=True)

    # 追踪记录
    tracking_record = {
        'datetime': signal['datetime'],
        'price': signal['price'],
        'ratio': signal['indicators']['ratio'],
        'rsi': signal['indicators']['rsi'],
        'stc': signal['indicators']['stc'],
        'trend': signal['trend'],
        'buy_signal': signal['buy_signal'],
        'sell_signal': signal['sell_signal'],
        'holding': position['holding'],
        'position_size': position.get('position_size', 0) if position['holding'] else 0,
        'data_source': data_source,  # 记录数据源 (API/CSV)
        'timestamp': datetime.now().isoformat()
    }

    # 追加到CSV
    df_tracking = pd.DataFrame([tracking_record])
    if tracking_path.exists():
        df_existing = pd.read_csv(tracking_path)
        df_tracking = pd.concat([df_existing, df_tracking], ignore_index=True)

    df_tracking.to_csv(tracking_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n[记录] 监控记录已保存: {TRACKING_PATH}")

    # Telegram通知
    if telegram_notifier:
        logger.info("\n[Telegram] 发送监控报告...")
        success = telegram_notifier.send_monitoring_report(signal, position, data_source)
        if success:
            logger.info("[Telegram] 报告发送成功")
        else:
            logger.warning("[Telegram] 报告发送失败")
    else:
        logger.info("\n[Telegram] 跳过通知（未配置或配置加载失败）")

    logger.info("\n" + "=" * 80)
    logger.info("监控完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_monitoring()
