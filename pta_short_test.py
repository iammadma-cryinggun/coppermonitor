# -*- coding: utf-8 -*-
"""
期货监控系统（PTA测试版 - 支持做空）

测试内容：
- PTA品种支持做多和做空
- 双向交易信号
- 多空持仓管理

测试目的：
- 验证做空逻辑可行性
- 评估双向交易效果
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import time
import signal
import sys
from typing import Dict, Optional

# 导入本地模块
from china_futures_fetcher import ChinaFuturesFetcher
from notifier import get_notifier

# 全局变量
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"\n收到退出信号 {signum}，准备优雅退出...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# PTA 配置（支持做多和做空）
# ==========================================

PTA_CONFIG = {
    'name': 'PTA',
    'code': 'TA',
    'exchange': 'CZCE',
    'quality_score': 83.8,

    # 做多参数（已有）
    'long_params': {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.25,
        'STC_SELL_ZONE': 75,
        'STOP_LOSS_PCT': 0.02
    },

    # 做空参数（待测试，暂用做多参数的镜像）
    'short_params': {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,  # 做空时用 100-RSI = 60
        'RATIO_TRIGGER': -1.25,  # 负值区
        'STC_BUY_ZONE': 25,  # STC低位买入（平空仓）
        'STOP_LOSS_PCT': 0.02
    },

    'contract_size': 5,
    'margin_rate': 0.07
}

# 固定技术参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50

# 配置
HISTORICAL_DAYS = 300
RUN_INTERVAL_HOURS = 4

# 基础路径
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / 'logs'
POSITIONS_FILE = LOGS_DIR / 'pta_positions_with_short.json'
SIGNAL_LOG_FILE = LOGS_DIR / 'pta_signals_with_short.json'
TRACKING_FILE = LOGS_DIR / 'pta_tracking_with_short.csv'
LOG_FILE = LOGS_DIR / 'pta_monitor_with_short.log'

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

fetcher = ChinaFuturesFetcher()
telegram_notifier = get_notifier()

# ==========================================
# 技术指标计算
# ==========================================

def calculate_indicators(df, params):
    """计算技术指标"""
    df = df.copy()

    # EMA
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

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

    df['ratio_prev'] = df['ratio'].shift(1)
    df['stc_prev'] = df['stc'].shift(1)

    return df

# ==========================================
# 双向信号检测
# ==========================================

def check_signals_bidirectional(df, long_params, short_params):
    """
    检测做多和做空信号

    Returns:
        dict: 包含所有信号的字典
    """
    if len(df) < 200:
        return {'error': '数据不足，需要至少200根K线'}

    latest = df.iloc[-1]
    prev = df.iloc[-1]

    # ==================== 做多信号（原有逻辑） ====================
    trend_up_long = latest['ema_fast'] > latest['ema_slow']
    ratio_safe_long = (0 < latest['ratio'] < long_params['RATIO_TRIGGER'])
    ratio_shrinking_long = latest['ratio'] < prev['ratio']
    turning_up_long = latest['macd_dif'] > prev['macd_dif']
    is_strong_long = latest['rsi'] > long_params['RSI_FILTER']

    ema_cross_up = (prev['ema_fast'] <= prev['ema_slow']) and (latest['ema_fast'] > latest['ema_slow'])

    sniper_long = trend_up_long and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long
    chase_long = ema_cross_up and is_strong_long

    buy_signal = sniper_long or chase_long
    buy_reason = 'sniper_long' if sniper_long else ('chase_long' if chase_long else None)

    # 做多平仓信号
    stc_exit_long = (df['stc_prev'].iloc[-1] > long_params['STC_SELL_ZONE']) and (latest['stc'] < df['stc_prev'].iloc[-1])
    trend_exit_long = latest['ema_fast'] < latest['ema_slow']
    sell_long_signal = stc_exit_long or trend_exit_long
    sell_long_reason = 'stc_exit_long' if stc_exit_long else ('trend_exit_long' if trend_exit_long else None)

    # 做多止损价
    stop_loss_long = latest['close'] * (1 - long_params['STOP_LOSS_PCT']) if buy_signal else None

    # ==================== 做空信号（新增） ====================
    trend_down_short = latest['ema_fast'] < latest['ema_slow']
    ratio_safe_short = (short_params['RATIO_TRIGGER'] < latest['ratio'] < 0)  # 负值区
    ratio_falling_short = latest['ratio'] < prev['ratio']  # Ratio变得更负
    turning_down_short = latest['macd_dif'] < prev['macd_dif']  # MACD下降
    is_weak_short = latest['rsi'] < (100 - short_params['RSI_FILTER'])  # RSI < 60

    ema_cross_down = (prev['ema_fast'] >= prev['ema_slow']) and (latest['ema_fast'] < latest['ema_slow'])

    sniper_short = trend_down_short and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short
    chase_short = ema_cross_down and is_weak_short

    short_signal = sniper_short or chase_short
    short_reason = 'sniper_short' if sniper_short else ('chase_short' if chase_short else None)

    # 做空平仓信号
    stc_exit_short = (df['stc_prev'].iloc[-1] < short_params['STC_BUY_ZONE']) and (latest['stc'] > df['stc_prev'].iloc[-1])
    trend_exit_short = latest['ema_fast'] > latest['ema_slow']
    cover_short_signal = stc_exit_short or trend_exit_short
    cover_short_reason = 'stc_exit_short' if stc_exit_short else ('trend_exit_short' if trend_exit_short else None)

    # 做空止损价
    stop_loss_short = latest['close'] * (1 + short_params['STOP_LOSS_PCT']) if short_signal else None

    # ==================== 返回完整信号 ====================
    return {
        'datetime': str(latest['datetime']),
        'price': float(latest['close']),
        'low': float(latest['low']),
        'high': float(latest['high']),

        # 做多信号
        'buy_signal': buy_signal,
        'buy_reason': buy_reason,
        'stop_loss_long': stop_loss_long,

        # 做多平仓信号
        'sell_long_signal': sell_long_signal,
        'sell_long_reason': sell_long_reason,

        # 做空信号
        'short_signal': short_signal,
        'short_reason': short_reason,
        'stop_loss_short': stop_loss_short,

        # 做空平仓信号
        'cover_short_signal': cover_short_signal,
        'cover_short_reason': cover_short_reason,

        # 技术指标
        'indicators': {
            'ema_fast': float(latest['ema_fast']),
            'ema_slow': float(latest['ema_slow']),
            'macd_dif': float(latest['macd_dif']),
            'macd_dea': float(latest['macd_dea']),
            'ratio': float(latest['ratio']),
            'ratio_prev': float(df['ratio_prev'].iloc[-1]),
            'rsi': float(latest['rsi']),
            'stc': float(latest['stc']),
            'stc_prev': float(df['stc_prev'].iloc[-1])
        },

        'trend': 'up' if trend_up_long else 'down',
        'strength': 'strong' if latest['ratio'] > 1.5 else ('normal' if latest['ratio'] > 1.0 else 'weak')
    }

# ==========================================
# 持仓管理（支持多空）
# ==========================================

def load_position() -> Dict:
    """加载PTA持仓状态"""
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"持仓文件损坏，将创建新文件: {e}")
            backup_path = POSITIONS_FILE.with_suffix('.json.bak')
            if POSITIONS_FILE.exists():
                POSITIONS_FILE.rename(backup_path)

    # 初始化空仓
    return {
        'holding': False,
        'direction': None,  # 'long' or 'short'
        'entry_price': None,
        'entry_datetime': None,
        'stop_loss': None,
        'signal_id': None
    }

def save_position(position: Dict):
    """保存持仓状态"""
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(position, f, ensure_ascii=False, indent=2)

def update_position_bidirectional(signal: dict, position: dict):
    """
    更新PTA持仓状态（支持做多和做空）

    Returns:
        更新后的position字典
    """
    # ==================== 场景1: 当前持有多仓 ====================
    if position['holding'] and position['direction'] == 'long':
        # 检查平多仓信号
        if signal['sell_long_signal']:
            logger.info(f"[PTA] 平多仓信号: {signal['sell_long_reason']}")

            entry_price = position['entry_price']
            exit_price = signal['price']
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            logger.info(f"[PTA] 平多仓: 入场{entry_price:.2f} → 出场{exit_price:.2f} | 盈亏{pnl_pct:+.2f}%")

            # 平多仓
            position = {
                'holding': False,
                'direction': None,
                'entry_price': None,
                'entry_datetime': None,
                'stop_loss': None,
                'signal_id': None
            }

            log_trade('PTA', 'sell_long', signal, pnl_pct)

        # 检查做多止损
        elif signal['low'] <= position['stop_loss']:
            logger.warning(f"[PTA] 多仓止损触发! 最低价 {signal['low']:.2f} <= 止损 {position['stop_loss']:.2f}")

            entry_price = position['entry_price']
            exit_price = signal['price']  # 市场价
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            logger.info(f"[PTA] 平多仓: 入场{entry_price:.2f} → 止损价{position['stop_loss']:.2f} → 市场价{exit_price:.2f} | 盈亏{pnl_pct:+.2f}%")

            position = {
                'holding': False,
                'direction': None,
                'entry_price': None,
                'entry_datetime': None,
                'stop_loss': None,
                'signal_id': None
            }

            signal_with_stoploss = signal.copy()
            signal_with_stoploss['signal_type'] = 'stop_loss_long'
            signal_with_stoploss['stop_loss_price'] = position['stop_loss']
            log_trade('PTA', 'sell_long', signal_with_stoploss, pnl_pct, exit_price)

    # ==================== 场景2: 当前持有空仓 ====================
    elif position['holding'] and position['direction'] == 'short':
        # 检查平空仓信号
        if signal['cover_short_signal']:
            logger.info(f"[PTA] 平空仓信号: {signal['cover_short_reason']}")

            entry_price = position['entry_price']
            exit_price = signal['price']
            # 做空盈亏 = (entry - exit) / entry
            pnl_pct = (entry_price - exit_price) / entry_price * 100

            logger.info(f"[PTA] 平空仓: 入场{entry_price:.2f} → 出场{exit_price:.2f} | 盈亏{pnl_pct:+.2f}%")

            # 平空仓
            position = {
                'holding': False,
                'direction': None,
                'entry_price': None,
                'entry_datetime': None,
                'stop_loss': None,
                'signal_id': None
            }

            log_trade('PTA', 'cover_short', signal, pnl_pct)

        # 检查做空止损
        elif signal['high'] >= position['stop_loss']:
            logger.warning(f"[PTA] 空仓止损触发! 最高价 {signal['high']:.2f} >= 止损 {position['stop_loss']:.2f}")

            entry_price = position['entry_price']
            exit_price = signal['price']  # 市场价
            pnl_pct = (entry_price - exit_price) / entry_price * 100

            logger.info(f"[PTA] 平空仓: 入场{entry_price:.2f} → 止损价{position['stop_loss']:.2f} → 市场价{exit_price:.2f} | 盈亏{pnl_pct:+.2f}%")

            position = {
                'holding': False,
                'direction': None,
                'entry_price': None,
                'entry_datetime': None,
                'stop_loss': None,
                'signal_id': None
            }

            signal_with_stoploss = signal.copy()
            signal_with_stoploss['signal_type'] = 'stop_loss_short'
            signal_with_stoploss['stop_loss_price'] = position['stop_loss']
            log_trade('PTA', 'cover_short', signal_with_stoploss, pnl_pct, exit_price)

    # ==================== 场景3: 当前空仓 ====================
    elif not position['holding']:
        # 优先检查做多信号
        if signal['buy_signal']:
            logger.info(f"[PTA] 做多信号: {signal['buy_reason']}")

            position = {
                'holding': True,
                'direction': 'long',
                'entry_price': signal['price'],
                'entry_datetime': signal['datetime'],
                'stop_loss': signal['stop_loss_long'],
                'signal_id': f"{signal['datetime']}_{signal['buy_reason']}"
            }

            logger.info(f"[PTA] 开多仓: 价格{signal['price']:.2f} | 止损{signal['stop_loss_long']:.2f}")
            log_trade('PTA', 'buy_long', signal, 0)

        # 检查做空信号
        elif signal['short_signal']:
            logger.info(f"[PTA] 做空信号: {signal['short_reason']}")

            position = {
                'holding': True,
                'direction': 'short',
                'entry_price': signal['price'],
                'entry_datetime': signal['datetime'],
                'stop_loss': signal['stop_loss_short'],
                'signal_id': f"{signal['datetime']}_{signal['short_reason']}"
            }

            logger.info(f"[PTA] 开空仓: 价格{signal['price']:.2f} | 止损{signal['stop_loss_short']:.2f}")
            log_trade('PTA', 'sell_short', signal, 0)

    return position

def log_trade(future_name: str, action: str, signal: dict, pnl_pct: float, actual_price: float = None):
    """记录交易"""
    log_path = Path(SIGNAL_LOG_FILE)

    if log_path.exists():
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logs = []
    else:
        logs = []

    trade_price = actual_price if actual_price is not None else signal['price']

    trade_entry = {
        'timestamp': datetime.now().isoformat(),
        'future': future_name,
        'action': action,
        'signal_datetime': signal['datetime'],
        'price': trade_price,
        'signal_type': signal.get('signal_type', 'unknown'),
        'pnl_pct': pnl_pct if action in ['sell_long', 'cover_short'] else None,
        'stop_loss': signal.get('stop_loss_long') or signal.get('stop_loss_short'),
        'indicators': signal['indicators']
    }

    logs.append(trade_entry)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# ==========================================
# 主监控逻辑
# ==========================================

def run_monitoring_pta():
    """运行PTA监控（支持做空）"""
    logger.info("=" * 80)
    logger.info("PTA监控系统（支持做多和做空）")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # 加载持仓
    position = load_position()

    # 获取数据
    logger.info("\n获取PTA数据...")
    df = fetcher.get_historical_data(PTA_CONFIG['code'], days=HISTORICAL_DAYS)

    if df is None or df.empty:
        logger.error("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    logger.info(f"数据获取成功: {len(df)}条记录")

    # 计算指标
    logger.info("\n计算技术指标...")
    df = calculate_indicators(df, PTA_CONFIG['long_params'])

    # 检测信号
    logger.info("\n检测交易信号...")
    signal = check_signals_bidirectional(df, PTA_CONFIG['long_params'], PTA_CONFIG['short_params'])

    if 'error' in signal:
        logger.error(f"信号检测错误: {signal['error']}")
        return

    # 打印当前状态
    logger.info(f"\n{'='*60}")
    logger.info(f"当前价格: {signal['price']:.2f}")
    logger.info(f"趋势: {signal['trend']} ({signal['strength']})")
    logger.info(f"Ratio: {signal['indicators']['ratio']:.2f}")
    logger.info(f"RSI: {signal['indicators']['rsi']:.1f}")
    logger.info(f"STC: {signal['indicators']['stc']:.1f}")

    # 打印持仓状态
    if position['holding']:
        direction_str = "多仓" if position['direction'] == 'long' else "空仓"
        logger.info(f"当前持仓: {direction_str}")
        logger.info(f"  入场价: {position['entry_price']:.2f}")
        logger.info(f"  止损价: {position['stop_loss']:.2f}")

        # 计算当前盈亏
        if position['direction'] == 'long':
            pnl_pct = (signal['price'] - position['entry_price']) / position['entry_price'] * 100
        else:
            pnl_pct = (position['entry_price'] - signal['price']) / position['entry_price'] * 100
        logger.info(f"  当前盈亏: {pnl_pct:+.2f}%")
    else:
        logger.info("当前持仓: 空仓")

    # 打印信号
    logger.info(f"\n{'='*60}")
    if signal.get('buy_signal'):
        logger.warning(f"做多信号: {signal['buy_reason']} ⭐")
    if signal.get('short_signal'):
        logger.warning(f"做空信号: {signal['short_reason']} ⭐")
    if signal.get('sell_long_signal'):
        logger.warning(f"平多信号: {signal['sell_long_reason']} ⭐")
    if signal.get('cover_short_signal'):
        logger.warning(f"平空信号: {signal['cover_short_reason']} ⭐")

    # 更新持仓
    logger.info(f"\n{'='*60}")
    logger.info("更新持仓状态...")
    position = update_position_bidirectional(signal, position)
    save_position(position)

    # 最终状态
    logger.info(f"\n{'='*60}")
    logger.info("持仓更新完成")
    if position['holding']:
        direction_str = "多仓" if position['direction'] == 'long' else "空仓"
        logger.info(f"当前状态: {direction_str}")
    else:
        logger.info("当前状态: 空仓")

    logger.info("=" * 80)

if __name__ == "__main__":
    run_monitoring_pta()
