# -*- coding: utf-8 -*-
"""
===================================
期货多品种策略监控系统（TOP 7优质信号）
===================================

功能:
1. 同时监控7个期货品种
2. 每个品种使用独立的最优参数
3. 独立的持仓状态管理
4. 每4小时K线收盘后30分钟运行（0:30, 8:30, 12:30, 20:30），确保数据已更新
5. 统一的信号推送和日志记录
6. 不记录具体金额，只记录持仓状态

监控品种（按信号质量排序）:
1. 沪镍      - 81.0分
2. 纯碱      - 78.6分
3. PVC      - 78.3分
4. 沪铜      - 77.0分
5. 沪锡      - 76.2分
6. 沪铅      - 73.3分
7. 玻璃      - 71.9分
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
from typing import Dict, List, Optional

# 导入本地模块
from china_futures_fetcher import ChinaFuturesFetcher

# 全局变量：优雅退出标志
shutdown_requested = False


def signal_handler(signum, frame):
    """处理退出信号，实现优雅退出"""
    global shutdown_requested
    logger.info(f"\n收到退出信号 {signum}，准备优雅退出...")
    shutdown_requested = True


# 注册信号处理器
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# TOP 7 品种配置（最优参数）
# ==========================================

TOP7_FUTURES_CONFIG = {
    '沪镍': {
        'name': '沪镍',
        'code': 'NI',  # 上期所代码
        'exchange': 'SHFE',
        'quality_score': 81.0,
        'params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 20,
            'RSI_FILTER': 55,
            'RATIO_TRIGGER': 1.10,
            'STC_SELL_ZONE': 75,
            'STOP_LOSS_PCT': 0.02
        },
        'contract_size': 1,
        'margin_rate': 0.12
    },
    '纯碱': {
        'name': '纯碱',
        'code': 'SA',  # 郑商所代码
        'exchange': 'CZCE',
        'quality_score': 78.6,
        'params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
            'STOP_LOSS_PCT': 0.02
        },
        'contract_size': 20,
        'margin_rate': 0.08
    },
    'PVC': {
        'name': 'PVC',
        'code': 'V',  # 大商所代码
        'exchange': 'DCE',
        'quality_score': 78.3,
        'params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 25,
            'RSI_FILTER': 55,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
            'STOP_LOSS_PCT': 0.02
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪铜': {
        'name': '沪铜',
        'code': 'CU',  # 上期所代码
        'exchange': 'SHFE',
        'quality_score': 77.0,
        'params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
            'STOP_LOSS_PCT': 0.02
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪锡': {
        'name': '沪锡',
        'code': 'SN',  # 上期所代码
        'exchange': 'SHFE',
        'quality_score': 76.2,
        'params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 10,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.25,
            'STC_SELL_ZONE': 75,
            'STOP_LOSS_PCT': 0.02
        },
        'contract_size': 1,
        'margin_rate': 0.13
    },
    '沪铅': {
        'name': '沪铅',
        'code': 'PB',  # 上期所代码
        'exchange': 'SHFE',
        'quality_score': 73.3,
        'params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 40,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
            'STOP_LOSS_PCT': 0.02
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '玻璃': {
        'name': '玻璃',
        'code': 'FG',  # 郑商所代码
        'exchange': 'CZCE',
        'quality_score': 71.9,
        'params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.10,
            'STC_SELL_ZONE': 75,
            'STOP_LOSS_PCT': 0.02
        },
        'contract_size': 20,
        'margin_rate': 0.08
    }
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
HISTORICAL_DAYS = 300  # 获取历史数据天数
RUN_INTERVAL_HOURS = 4  # 运行间隔（小时）

# 基础路径
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / 'logs'
CONFIG_DIR = BASE_DIR / 'config'

# 数据路径
BACKUP_DATA_DIR = BASE_DIR / 'data'
POSITIONS_FILE = LOGS_DIR / 'multi_positions.json'
SIGNAL_LOG_FILE = LOGS_DIR / 'multi_signals.json'
TRACKING_FILE = LOGS_DIR / 'multi_tracking.csv'
REPLAY_DATA_FILE = LOGS_DIR / 'multi_replay_data.csv'  # 详细复盘数据
LOG_FILE = LOGS_DIR / 'multi_monitor.log'

# 确保目录存在
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

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


# ==========================================
# 技术指标计算
# ==========================================

def calculate_indicators(df, params):
    """计算技术指标"""
    try:
        df = df.copy()

        # EMA
        df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

        # MACD & Ratio
        exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd_dif'] = exp1 - exp2
        df['macd_dea'] = df['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['ratio'] = np.where(df['macd_dea'].abs() > 1e-10, df['macd_dif'] / df['macd_dea'], 0)

        # RSI (避免除零)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
        # 防止除零：当loss接近0时使用小值
        loss_safe = loss.where(loss.abs() > 1e-10, 1e-10)
        rs = gain / loss_safe
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # STC (修复浮点数精度问题)
        stc_macd = df['close'].ewm(span=STC_FAST, adjust=False).mean() - \
                   df['close'].ewm(span=STC_SLOW, adjust=False).mean()
        stoch_period = STC_LENGTH
        min_macd = stc_macd.rolling(window=stoch_period).min()
        max_macd = stc_macd.rolling(window=stoch_period).max()
        # 使用where检查分母是否接近0，而非只替换精确的0
        macd_range = max_macd - min_macd
        stoch_k = 100 * (stc_macd - min_macd) / macd_range.where(macd_range.abs() > 1e-10, np.nan)
        stoch_k = stoch_k.fillna(50)
        stoch_d = stoch_k.rolling(window=3).mean()
        min_stoch_d = stoch_d.rolling(window=stoch_period).min()
        max_stoch_d = stoch_d.rolling(window=stoch_period).max()
        # 同样修复浮点数精度问题
        stoch_range = max_stoch_d - min_stoch_d
        stc_raw = 100 * (stoch_d - min_stoch_d) / stoch_range.where(stoch_range.abs() > 1e-10, np.nan)
        stc_raw = stc_raw.fillna(50)
        df['stc'] = stc_raw.rolling(window=3).mean()

        df['ratio_prev'] = df['ratio'].shift(1)
        df['stc_prev'] = df['stc'].shift(1)

        return df
    except Exception as e:
        logger.error(f"计算指标时出错: {e}")
        raise


# ==========================================
# 信号检测
# ==========================================

def check_signals(df, params, future_name):
    """
    检查交易信号

    Returns:
        dict: 信号信息
    """
    if len(df) < 200:
        return {'error': '数据不足，需要至少200根K线'}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 信号条件
    trend_up = latest['ema_fast'] > latest['ema_slow']
    ratio_safe = (0 < latest['ratio'] < params['RATIO_TRIGGER'])
    ratio_shrinking = latest['ratio'] < prev['ratio']
    turning_up = latest['macd_dif'] > prev['macd_dif']
    is_strong = latest['rsi'] > params['RSI_FILTER']

    ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (latest['ema_fast'] > latest['ema_slow'])

    sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
    chase_signal = ema_cross and is_strong

    buy_signal = sniper_signal or chase_signal
    buy_reason = 'sniper' if sniper_signal else ('chase' if chase_signal else None)

    # 卖出信号
    stc_exit = (df['stc_prev'].iloc[-1] > params['STC_SELL_ZONE']) and (latest['stc'] < df['stc_prev'].iloc[-1])
    trend_exit = latest['ema_fast'] < latest['ema_slow']
    sell_signal = stc_exit or trend_exit
    sell_reason = 'stc' if stc_exit else ('trend' if trend_exit else None)

    # 止损价
    stop_loss = latest['close'] * (1 - params['STOP_LOSS_PCT']) if buy_signal else None

    return {
        'future': future_name,
        'datetime': str(latest['datetime']),
        'price': float(latest['close']),
        'low': float(latest['low']),  # 最低价，用于止损检查
        'high': float(latest['high']),  # 最高价，记录完整信息
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
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'signal_type': buy_reason if buy_signal else (sell_reason if sell_signal else None),
        'stop_loss': stop_loss,
        'reason': {
            'buy': buy_reason,
            'sell': sell_reason
        },
        'trend': 'up' if trend_up else 'down',
        'strength': 'strong' if latest['ratio'] > 1.5 else ('normal' if latest['ratio'] > 1.0 else 'weak')
    }


# ==========================================
# 持仓管理
# ==========================================

def load_all_positions() -> Dict:
    """加载所有品种的持仓状态"""
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[持仓] 现有文件损坏，将创建新文件: {e}")
            # 备份损坏文件
            backup_path = POSITIONS_FILE.with_suffix('.json.bak')
            POSITIONS_FILE.rename(backup_path)
            logger.info(f"[持仓] 已备份损坏文件到: {backup_path}")

    # 初始化空持仓
    return {
        future_name: {
            'holding': False,
            'entry_price': None,
            'entry_datetime': None,
            'stop_loss': None,
            'signal_id': None
        }
        for future_name in TOP7_FUTURES_CONFIG.keys()
    }


def save_all_positions(positions: Dict):
    """保存所有品种的持仓状态"""
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


def update_position(future_name: str, signal: dict, positions: Dict) -> Dict:
    """
    更新单个品种的持仓状态

    Returns:
        更新后的positions字典
    """
    position = positions.get(future_name, {
        'holding': False,
        'entry_price': None,
        'entry_datetime': None,
        'stop_loss': None,
        'signal_id': None
    })

    if position['holding']:
        # 当前有持仓，检查是否需要平仓
        if signal['sell_signal']:
            logger.info(f"[{future_name}] 平仓信号: {signal['reason']['sell']}")

            # 计算盈亏（百分比）
            entry_price = position['entry_price']
            # 使用实际平仓价格（止损时使用市场价）
            exit_price = signal.get('actual_exit_price', signal['price'])
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            # 止损时额外记录止损价信息
            if signal['reason']['sell'] == 'stop_loss':
                stop_loss_price = signal.get('stop_loss_price', position['stop_loss'])
                logger.info(f"[{future_name}] 平仓: 入场{entry_price:.2f} → 止损价{stop_loss_price:.2f} → 实际出场{exit_price:.2f} | 盈亏{pnl_pct:+.2f}%")
            else:
                logger.info(f"[{future_name}] 平仓: 入场{entry_price:.2f} → 出场{exit_price:.2f} | 盈亏{pnl_pct:+.2f}%")

            # 清空持仓
            positions[future_name] = {
                'holding': False,
                'entry_price': None,
                'entry_datetime': None,
                'stop_loss': None,
                'signal_id': None
            }

            # 记录交易（传递实际平仓价格）
            log_trade(future_name, 'sell', signal, pnl_pct, exit_price)

    else:
        # 当前无持仓，检查是否需要开仓
        if signal['buy_signal']:
            logger.info(f"[{future_name}] 开仓信号: {signal['reason']['buy']}")

            positions[future_name] = {
                'holding': True,
                'entry_price': signal['price'],
                'entry_datetime': signal['datetime'],
                'stop_loss': signal['stop_loss'],
                'signal_id': f"{signal['datetime']}_{signal.get('signal_type', 'manual')}"
            }

            logger.info(f"[{future_name}] 开仓: 价格{signal['price']:.2f} | 止损{signal['stop_loss']:.2f}")

            # 记录交易
            log_trade(future_name, 'buy', signal, 0)

    return positions


def log_trade(future_name: str, action: str, signal: dict, pnl_pct: float, actual_price: float = None):
    """
    记录交易到日志

    Args:
        future_name: 品种名称
        action: 'buy' or 'sell'
        signal: 信号字典
        pnl_pct: 盈亏百分比（仅平仓时）
        actual_price: 实际交易价格（止损平仓时使用市场价）
    """
    log_path = Path(SIGNAL_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logs = []
    else:
        logs = []

    # 确定记录的价格
    trade_price = actual_price if actual_price is not None else signal['price']

    trade_entry = {
        'timestamp': datetime.now().isoformat(),
        'future': future_name,
        'action': action,  # 'buy' or 'sell'
        'signal_datetime': signal['datetime'],
        'price': trade_price,  # 实际交易价格
        'signal_type': signal.get('signal_type', 'unknown'),
        'pnl_pct': pnl_pct if action == 'sell' else None,
        'stop_loss': signal.get('stop_loss'),
        'stop_loss_price': signal.get('stop_loss_price'),  # 止损价（止损平仓时）
        'indicators': signal['indicators']
    }

    logs.append(trade_entry)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def save_replay_data(all_signals: dict, positions: dict, data_sources: dict):
    """
    保存详细复盘数据（OHLC + 技术指标）

    用于未来复盘分析，包含完整的价格和技术指标信息
    """
    replay_records = []

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for future_name, config in TOP7_FUTURES_CONFIG.items():
        signal = all_signals.get(future_name, {})
        position = positions.get(future_name, {})
        data_source = data_sources.get(future_name, 'Unknown')

        # 跳过错误数据
        if 'error' in signal:
            continue

        try:
            record = {
                'timestamp': timestamp,
                'future': future_name,
                'code': config['code'],
                'exchange': config['exchange'],
                'quality_score': config['quality_score'],

                # OHLC价格（完整K线数据）
                'open': signal.get('indicators', {}).get('open', 0),  # 需要从原始数据获取
                'high': signal.get('high', 0),
                'low': signal.get('low', 0),
                'close': signal.get('price', 0),

                # 技术指标值
                'ema_fast': signal.get('indicators', {}).get('ema_fast', 0),
                'ema_slow': signal.get('indicators', {}).get('ema_slow', 0),
                'macd_dif': signal.get('indicators', {}).get('macd_dif', 0),
                'macd_dea': signal.get('indicators', {}).get('macd_dea', 0),
                'ratio': signal.get('indicators', {}).get('ratio', 0),
                'ratio_prev': signal.get('indicators', {}).get('ratio_prev', 0),
                'rsi': signal.get('indicators', {}).get('rsi', 0),
                'stc': signal.get('indicators', {}).get('stc', 0),
                'stc_prev': signal.get('indicators', {}).get('stc_prev', 0),

                # 信号状态
                'trend': signal.get('trend', 'unknown'),
                'strength': signal.get('strength', 'unknown'),
                'buy_signal': signal.get('buy_signal', False),
                'sell_signal': signal.get('sell_signal', False),
                'signal_type': signal.get('signal_type', ''),

                # 持仓信息
                'holding': position.get('holding', False),
                'entry_price': position.get('entry_price', 0) if position.get('holding') else 0,
                'stop_loss': position.get('stop_loss', 0) if position.get('holding') else 0,

                # 数据来源
                'data_source': data_source,

                # 参数配置
                'param_ema_fast': config['params']['EMA_FAST'],
                'param_ema_slow': config['params']['EMA_SLOW'],
                'param_rsi': config['params']['RSI_FILTER'],
                'param_ratio': config['params']['RATIO_TRIGGER'],
                'param_stc': config['params']['STC_SELL_ZONE'],
                'param_stop_loss': config['params']['STOP_LOSS_PCT'],
            }

            replay_records.append(record)

        except Exception as e:
            logger.error(f"[{future_name}] 保存复盘数据失败: {e}")

    # 保存到CSV
    if replay_records:
        df_replay = pd.DataFrame(replay_records)

        # 追加模式
        if REPLAY_DATA_FILE.exists():
            df_existing = pd.read_csv(REPLAY_DATA_FILE)
            df_replay = pd.concat([df_existing, df_replay], ignore_index=True)

        df_replay.to_csv(REPLAY_DATA_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"复盘数据已保存: {REPLAY_DATA_FILE} ({len(replay_records)}个品种)")


# ==========================================
# 数据获取
# ==========================================

def load_market_data(future_name: str, future_code: str):
    """
    加载单个品种的市场数据

    Returns:
        (df, data_source) or (None, None)
    """
    # 尝试使用实时API
    logger.debug(f"[{future_name}] 尝试从API获取数据...")
    df = fetcher.get_historical_data(future_code, days=HISTORICAL_DAYS)

    if df is not None and not df.empty:
        logger.debug(f"[{future_name}] API成功获取 {len(df)} 条记录")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        return df, 'API'

    # API失败，尝试CSV备用
    logger.debug(f"[{future_name}] API失败，尝试CSV备用...")

    # 查找CSV文件
    csv_files = list(BACKUP_DATA_DIR.glob(f'*{future_name}*.csv'))
    csv_files.extend(list(BACKUP_DATA_DIR.glob(f'*{future_code}*.csv')))

    if csv_files:
        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.debug(f"[{future_name}] CSV成功加载 {len(df)} 条记录")
            return df, 'CSV'
        except Exception as e:
            logger.error(f"[{future_name}] CSV加载失败: {e}")

    return None, None


# ==========================================
# 主监控逻辑
# ==========================================

def monitor_single_future(future_name: str, config: dict, positions: Dict) -> Dict:
    """
    监控单个品种

    Returns:
        该品种的信号信息
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[{future_name}] 质量评分: {config['quality_score']}分")
    logger.info(f"[{future_name}] 参数: EMA({config['params']['EMA_FAST']},{config['params']['EMA_SLOW']}), "
                f"RSI={config['params']['RSI_FILTER']}, RATIO={config['params']['RATIO_TRIGGER']:.2f}, "
                f"STC={config['params']['STC_SELL_ZONE']}")
    logger.info(f"{'='*60}")

    # 加载数据
    df, data_source = load_market_data(future_name, config['code'])

    if df is None:
        logger.error(f"[{future_name}] 数据加载失败")
        return {'error': '数据加载失败'}

    logger.info(f"[{future_name}] 数据源: {data_source} | 数据量: {len(df)}条")

    # 计算指标
    df = calculate_indicators(df, config['params'])

    # 检查信号
    signal = check_signals(df, config['params'], future_name)

    if 'error' in signal:
        logger.error(f"[{future_name}] {signal['error']}")
        return signal, data_source

    # 获取当前持仓
    position = positions.get(future_name, {'holding': False})

    # 输出当前状态
    logger.info(f"[{future_name}] 价格: {signal['price']:.2f} | "
                f"趋势: {signal['trend']} ({signal['strength']}) | "
                f"Ratio: {signal['indicators']['ratio']:.2f} | "
                f"RSI: {signal['indicators']['rsi']:.1f} | "
                f"STC: {signal['indicators']['stc']:.1f}")

    if position['holding']:
        logger.info(f"[{future_name}] 当前持仓: 是 | "
                    f"入场价: {position['entry_price']:.2f} | "
                    f"止损价: {position['stop_loss']:.2f}")

        # 计算当前盈亏
        current_pnl_pct = (signal['price'] - position['entry_price']) / position['entry_price'] * 100
        logger.info(f"[{future_name}] 当前盈亏: {current_pnl_pct:+.2f}%")

        # 检查止损（实盘逻辑：使用最低价检查是否触及止损，按市场价平仓）
        if signal['low'] <= position['stop_loss']:
            # 止损已被触及（K线最低价触及止损价）
            actual_exit_price = signal['price']  # 实盘按当前市场价平仓
            logger.warning(f"[{future_name}] 止损触发! K线最低价 {signal['low']:.2f} <= 止损价 {position['stop_loss']:.2f}")
            logger.warning(f"[{future_name}] 立即平仓: 止损价 {position['stop_loss']:.2f} → 市场价 {actual_exit_price:.2f}")

            # 触发平仓信号
            signal['sell_signal'] = True
            signal['signal_type'] = 'stop_loss'
            signal['reason']['sell'] = 'stop_loss'
            # 记录实际平仓价格（市场价）
            signal['actual_exit_price'] = actual_exit_price
            signal['stop_loss_price'] = position['stop_loss']

    else:
        logger.info(f"[{future_name}] 当前持仓: 否")

    # 输出信号
    if signal['buy_signal']:
        logger.warning(f"[{future_name}] 买入信号: {signal['reason']['buy']} ⭐")

    if signal['sell_signal']:
        logger.warning(f"[{future_name}] 卖出信号: {signal['reason']['sell']} ⭐")

    # 更新持仓状态
    positions = update_position(future_name, signal, positions)

    # 添加数据源信息到signal中，用于复盘
    signal['data_source'] = data_source

    return signal, data_source


def run_monitoring():
    """运行多品种监控"""
    logger.info("=" * 80)
    logger.info("期货多品种策略监控系统（TOP 10优质信号）")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # 加载所有持仓状态
    positions = load_all_positions()

    # 监控结果
    all_signals = {}
    data_sources = {}  # 记录数据来源
    buy_signals = []
    sell_signals = []
    active_positions = []

    # 逐个监控
    for future_name, config in TOP7_FUTURES_CONFIG.items():
        try:
            signal, data_source = monitor_single_future(future_name, config, positions)
            all_signals[future_name] = signal
            data_sources[future_name] = data_source

            # 记录交易信号
            if signal.get('buy_signal'):
                buy_signals.append(future_name)
            if signal.get('sell_signal'):
                sell_signals.append(future_name)

            # 记录当前持仓
            if positions[future_name]['holding']:
                active_positions.append(future_name)

        except Exception as e:
            logger.error(f"[{future_name}] 监控异常: {e}")
            all_signals[future_name] = {'error': str(e)}
            data_sources[future_name] = 'Error'

    # 保存持仓状态
    save_all_positions(positions)

    # 保存追踪记录
    tracking_record = {
        'timestamp': datetime.now().isoformat(),
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'monitored_count': len(TOP7_FUTURES_CONFIG),
        'buy_signals': ','.join(buy_signals) if buy_signals else '',
        'sell_signals': ','.join(sell_signals) if sell_signals else '',
        'active_positions': ','.join(active_positions) if active_positions else '',
    }

    # 添加各品种状态
    for future_name, signal in all_signals.items():
        if 'error' not in signal:
            tracking_record[f'{future_name}_price'] = signal['price']
            tracking_record[f'{future_name}_trend'] = signal['trend']
            tracking_record[f'{future_name}_buy'] = signal['buy_signal']
            tracking_record[f'{future_name}_sell'] = signal['sell_signal']
            tracking_record[f'{future_name}_holding'] = positions[future_name]['holding']

    df_tracking = pd.DataFrame([tracking_record])
    if TRACKING_FILE.exists():
        df_existing = pd.read_csv(TRACKING_FILE)
        df_tracking = pd.concat([df_existing, df_tracking], ignore_index=True)

    df_tracking.to_csv(TRACKING_FILE, index=False, encoding='utf-8-sig')

    # 汇总报告
    logger.info("\n" + "=" * 80)
    logger.info("监控汇总")
    logger.info("=" * 80)
    logger.info(f"监控品种: {len(TOP7_FUTURES_CONFIG)}个")
    logger.info(f"当前持仓: {len(active_positions)}个 - {', '.join(active_positions) if active_positions else '无'}")

    if buy_signals:
        logger.warning(f"买入信号: {', '.join(buy_signals)} ⭐")

    if sell_signals:
        logger.warning(f"卖出信号: {', '.join(sell_signals)} ⭐")

    logger.info(f"追踪记录已保存: {TRACKING_FILE}")

    # 保存详细复盘数据
    try:
        save_replay_data(all_signals, positions, data_sources)
    except Exception as e:
        logger.error(f"保存复盘数据失败: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("监控完成")
    logger.info("=" * 80)

    return all_signals, positions


# ==========================================
# 定时运行
# ==========================================

def get_wait_seconds():
    """计算到下一个交易时段收盘后的等待时间

    4小时K线时间点: 01:00, 09:00, 13:00, 21:00
    程序运行时间: K线收盘后5分钟，确保数据已更新

    时间对应:
    - 01:00 K线收盘 -> 01:05 运行
    - 09:00 K线收盘 -> 09:05 运行
    - 13:00 K线收盘 -> 13:05 运行
    - 21:00 K线收盘 -> 21:05 运行
    """
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    # 运行时间点（K线收盘后5分钟）
    run_times = [
        (1, 5),    # 01:05 - 获取01:00 K线
        (9, 5),    # 09:05 - 获取09:00 K线
        (13, 5),   # 13:05 - 获取13:00 K线
        (21, 5)    # 21:05 - 获取21:00 K线
    ]

    # 找到下一个运行时间
    next_run = None
    for run_hour, run_minute in run_times:
        if run_hour > hour or (run_hour == hour and run_minute > minute):
            next_run = (run_hour, run_minute)
            break

    # 如果没找到（已过21:05），下一个是1:05（次日）
    if next_run is None:
        next_run = (1, 5)
        next_time = now.replace(hour=next_run[0], minute=next_run[1], second=0, microsecond=0)
        next_time += timedelta(days=1)
    else:
        next_time = now.replace(hour=next_run[0], minute=next_run[1], second=0, microsecond=0)

    wait_seconds = (next_time - now).total_seconds()
    return wait_seconds, next_time


def run_scheduled():
    """定时运行监控"""
    logger.info("=" * 80)
    logger.info("期货多品种监控系统 - 定时运行模式")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("运行时间: 4小时K线收盘后5分钟运行（最快获取数据）")
    logger.info("  - 01:05 (获取01:00 K线)")
    logger.info("  - 09:05 (获取09:00 K线)")
    logger.info("  - 13:05 (获取13:00 K线)")
    logger.info("  - 21:05 (获取21:00 K线)")
    logger.info("已注册信号处理器，支持优雅退出")
    logger.info("=" * 80)

    while not shutdown_requested:
        try:
            # 立即运行一次
            logger.info("\n开始执行监控...")
            run_monitoring()

            # 如果收到退出信号，退出循环
            if shutdown_requested:
                break

            # 计算等待时间
            wait_seconds, next_time = get_wait_seconds()
            wait_hours = wait_seconds / 3600

            logger.info(f"\n下次运行时间: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"等待时长: {wait_hours:.2f}小时 ({wait_seconds/60:.1f}分钟)")

            # 分段等待，每60秒检查一次退出信号
            wait_intervals = int(wait_seconds / 60)
            wait_remainder = wait_seconds % 60

            for i in range(wait_intervals):
                if shutdown_requested:
                    logger.info("检测到退出信号，中断等待...")
                    break
                time.sleep(60)

            if not shutdown_requested and wait_remainder > 0:
                time.sleep(wait_remainder)

        except Exception as e:
            logger.error(f"监控运行异常: {e}")
            # 异常时等待5分钟后重试
            if not shutdown_requested:
                logger.info("5分钟后重试...")
                time.sleep(300)

    logger.info("=" * 80)
    logger.info("服务已优雅退出")
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--scheduled':
        # 定时运行模式
        run_scheduled()
    else:
        # 单次运行模式
        run_monitoring()
