#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCI 铜期货 纸面交易系统
━━━━━━━━━━━━━━━━━━━━━━━
每日收盘后运行:
  1. 下载最新日线数据 (akshare)
  2. 计算 CCI + STC 指标
  3. 检查入场/出场信号
  4. 管理纸面持仓 (模拟下单)
  5. 记录交易到 CSV
  6. 保存状态到 JSON

监控品种: 玻璃/白银/铜/铝/锡 (OOS验证通过)
仓位: 1手起 (最小交易单位), 30%保证金上限
入场: 信号日收盘计算 → 次日开盘价成交
退出: 4%止损 / 20%止盈 / CCI超买 / CCI死叉
"""
import pandas as pd
import numpy as np
import os, sys, json, time
from datetime import datetime, timedelta

# ===== 配置 =====
DATA_DIR = os.environ.get('DATA_DIR', '/data')  # Zeabur 挂载卷路径
BASE_DIR = DATA_DIR
PAPER_CAPITAL = 500000  # 纸面本金 50万
MARGIN_RATE = 0.10      # 保证金比例 10%
SL_PCT = 0.04           # 止损 4%
TP_PCT = 0.20           # 止盈 20%
SLIPPAGE = 0.0002       # 滑点 0.02%
COMMISSION = 0.0003     # 手续费 0.03%

sys.path.insert(0, os.path.join(BASE_DIR, 'CCI策略系统'))
from cci_calculations import calculate_stc, f_normalize
from 最优参数配置 import OPTIMAL_PARAMS

import akshare as ak
import signal
import time as _time

# 监控品种 (OOS验证通过的)
WATCH_LIST = {
    '玻璃': {'code': 'FG0', 'exchange': 'CZCE', 'multiplier': 20, 'margin_rate': 0.12},
    '白银': {'code': 'AG0', 'exchange': 'SHFE', 'multiplier': 15, 'margin_rate': 0.12},
    '铜':   {'code': 'CU0', 'exchange': 'SHFE', 'multiplier': 5,  'margin_rate': 0.10},
    '铝':   {'code': 'AL0', 'exchange': 'SHFE', 'multiplier': 5,  'margin_rate': 0.10},
    '锡':   {'code': 'SN0', 'exchange': 'SHFE', 'multiplier': 1,  'margin_rate': 0.12},
}

# 路径
TRADE_LOG = os.path.join(BASE_DIR, 'paper_trade_log.csv')
STATE_FILE = os.path.join(BASE_DIR, 'paper_trade_state.json')

def calc_indicators(df):
    """计算CCI和STC"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(12).mean()
    mad = tp.rolling(12).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    cci_ma = cci.rolling(20).mean()
    stc_raw = calculate_stc(df.copy(), length=10, fast=23, slow=50, aaa=0.5)
    stc = f_normalize(stc_raw, 20, 80)
    df = df.copy()
    df['cci'] = cci
    df['cci_ma'] = cci_ma
    df['stc'] = stc
    return df

def check_entry_signal(row, params):
    """检查入场信号 (基于当前行的cci/stc)"""
    cci = row['cci']
    cci_ma = row['cci_ma']
    stc = row['stc']
    cci_prev = row.get('cci_prev', None)
    cci_ma_prev = row.get('cci_ma_prev', None)

    if cci_prev is None or cci_ma_prev is None:
        return False

    # 信号1: CCI超卖 + STC超卖
    if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
        return True
    # 信号2: CCI金叉 + CCI<=cci_cross_max + STC>=stc_cross
    if (cci_prev <= cci_ma_prev and cci > cci_ma and
        cci <= params['cci_cross_max'] and
        stc >= params['stc_cross']):
        return True
    return False

def check_exit_signal(row, params, entry_price, direction='LONG'):
    """检查出场信号"""
    cci = row['cci']
    cci_ma = row['cci_ma']
    cci_prev = row.get('cci_prev', None)
    cci_ma_prev = row.get('cci_ma_prev', None)

    price = row['close']
    pnl_pct = (price - entry_price) / entry_price if direction == 'LONG' else (entry_price - price) / entry_price

    # 止损4%
    if pnl_pct <= -SL_PCT:
        return 'SL', price
    # 止盈20%
    if pnl_pct >= TP_PCT:
        return 'TP', price
    # CCI超买
    if cci > params['cci_overbought']:
        return 'overbought', price
    # CCI死叉
    if cci_prev is not None and cci_ma_prev is not None:
        if cci_prev >= cci_ma_prev and cci < cci_ma:
            return 'death_cross', price
    return None, None

def get_latest_data(symbol, code):
    """通过akshare获取最新日线数据(需要足够warmup)"""
    # 获取最近1年数据 (足够计算CCI/STC)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')

    try:
        df = ak.futures_main_sina(symbol=code, start_date=start_date, end_date=end_date)
        df = df.rename(columns={
            '日期': 'date', '开盘价': 'open', '最高价': 'high',
            '最低价': 'low', '收盘价': 'close', '成交量': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        return df
    except Exception as e:
        print(f"  [ERROR] {symbol} 数据获取失败: {e}")
        return None

def load_state():
    """加载纸面交易状态"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'positions': {}, 'trades': [], 'equity': [], 'balance': PAPER_CAPITAL}

def save_state(state):
    """保存纸面交易状态"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)

def load_trade_log():
    """加载交易记录"""
    if os.path.exists(TRADE_LOG):
        return pd.read_csv(TRADE_LOG)
    return pd.DataFrame()

def append_trade_log(trade):
    """追加交易记录"""
    df = pd.DataFrame([trade])
    if os.path.exists(TRADE_LOG):
        df.to_csv(TRADE_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(TRADE_LOG, index=False)

def run_daily_scan():
    """每日扫描: 检查所有品种的信号"""
    state = load_state()
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*70}")
    print(f"纸面交易扫描 — {today}")
    print(f"本金: {state['balance']:,.0f} | 当前持仓: {len(state['positions'])}个")
    print(f"{'='*70}")

    new_trades = 0

    for name, cfg in WATCH_LIST.items():
        params = OPTIMAL_PARAMS[name]
        print(f"\n--- {name} ({cfg['code']}) ---")

        # 获取数据
        df = get_latest_data(name, cfg['code'])
        if df is None or len(df) < 40:
            print(f"  数据不足, 跳过")
            continue

        # 计算指标
        df = calc_indicators(df)
        df['cci_prev'] = df['cci'].shift(1)
        df['cci_ma_prev'] = df['cci_ma'].shift(1)

        # 检查是否有持仓
        pos = state['positions'].get(name, None)

        if pos:
            # --- 管理现有持仓 ---
            print(f"  持仓中: 入场{pos['entry_date']} 价格{pos['entry_price']:.0f} "
                  f"方向{pos['direction']} 止损{pos['stop_loss']:.0f}")

            # 用最新bar检查出场信号
            latest_row = df.iloc[-1]
            exit_signal, exit_price = check_exit_signal(
                latest_row, params, pos['entry_price'], pos['direction'])

            if exit_signal:
                # 平仓
                pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] if pos['direction'] == 'LONG' else (pos['entry_price'] - exit_price) / pos['entry_price']
                pnl = pnl_pct * cfg['multiplier'] * pos['lots']  # 简化: 1手
                state['balance'] += pnl

                trade = {
                    'date': today, 'symbol': name, 'side': pos['direction'],
                    'action': 'EXIT', 'reason': exit_signal,
                    'entry_date': pos['entry_date'], 'exit_date': today,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'pnl_pct': pnl_pct * 100, 'pnl': pnl,
                    'lots': pos['lots'], 'multiplier': cfg['multiplier']
                }
                append_trade_log(trade)
                state['trades'].append(trade)
                state['equity'].append(state['balance'])

                print(f"  [平仓] {exit_signal} @ {exit_price:.0f} PnL {pnl_pct*100:+.1f}%")
                del state['positions'][name]
                new_trades += 1
            else:
                # 检查是否需要移动止损 (如果盈利>5%, 止损上移)
                current_price = df.iloc[-1]['close']
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] if pos['direction'] == 'LONG' else (pos['entry_price'] - current_price) / pos['entry_price']
                if pnl_pct > 0.05:
                    new_sl = pos['entry_price'] * (1 + pnl_pct * 0.5) if pos['direction'] == 'LONG' else pos['entry_price'] * (1 - pnl_pct * 0.5)
                    pos['stop_loss'] = new_sl

        else:
            # --- 检查入场信号 ---
            latest_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            # 构造prev_row的属性用于check_entry_signal
            latest_row = latest_row.copy()
            latest_row['cci_prev'] = prev_row['cci']
            latest_row['cci_ma_prev'] = prev_row['cci_ma']

            if check_entry_signal(latest_row, params):
                # 入场: 次日开盘价 (今天收盘价确定信号, 明天开盘成交)
                # 用最新bar的close作为信号确认, entry = 明天的open
                entry_price = df.iloc[-1]['close'] * (1 + SLIPPAGE)

                # 计算仓位: 30%保证金上限
                margin_per_lot = entry_price * cfg['multiplier'] * cfg['margin_rate']
                cap = state['balance'] * 0.30  # 30%保证金上限
                lots = int(cap / margin_per_lot)
                lots = max(0, lots)

                if lots >= 1:
                    # 入场
                    state['positions'][name] = {
                        'direction': 'LONG',
                        'entry_date': today,
                        'entry_price': entry_price,
                        'lots': lots,
                        'stop_loss': entry_price * (1 - SL_PCT),
                        'multiplier': cfg['multiplier']
                    }
                    fee = entry_price * cfg['multiplier'] * lots * COMMISSION * 2
                    state['balance'] -= fee

                    trade = {
                        'date': today, 'symbol': name, 'side': 'LONG',
                        'action': 'ENTRY', 'reason': 'cci_oversold/golden_cross',
                        'entry_date': today, 'exit_date': '',
                        'entry_price': entry_price, 'exit_price': '',
                        'pnl_pct': 0, 'pnl': 0,
                        'lots': lots, 'multiplier': cfg['multiplier']
                    }
                    append_trade_log(trade)
                    state['trades'].append(trade)
                    state['equity'].append(state['balance'])

                    print(f"  [入场] 信号触发! 价格{entry_price:.0f} "
                          f"手数{lots} 止损{entry_price*(1-SL_PCT):.0f}")
                    new_trades += 1
                else:
                    need_pct = min(margin_per_lot / state['balance'], 1.0) * 100
                    print(f"  [无仓位] 信号触发但保证金不足 (需要{need_pct:.0f}%保证金)")
            else:
                print(f"  [等待] 无信号")

    # 保存状态
    state['last_scan'] = today
    save_state(state)

    # 汇总
    print(f"\n{'='*70}")
    print(f"扫描完成 — 新增交易: {new_trades} | 当前持仓: {len(state['positions'])}个")
    print(f"当前权益: {state['balance']:,.0f}")

    # 统计历史交易
    if state['trades']:
        closes = [t for t in state['trades'] if t['action'] == 'EXIT']
        if closes:
            wins = sum(1 for t in closes if t['pnl'] > 0)
            total_pnl = sum(t['pnl'] for t in closes)
            print(f"历史: {len(closes)}笔平仓, 胜率{wins/len(closes)*100:.0f}%, "
                  f"累计PnL {total_pnl:+,.0f}")

    return state

def show_status():
    """显示当前状态"""
    state = load_state()
    print(f"\n{'='*70}")
    print(f"纸面交易状态")
    print(f"{'='*70}")
    print(f"本金: {state['balance']:,.0f}")
    print(f"持仓: {len(state['positions'])}个")
    for name, pos in state['positions'].items():
        print(f"  {name}: {pos['direction']} 入场{pos['entry_date']} "
              f"价格{pos['entry_price']:.0f} 止损{pos['stop_loss']:.0f}")

    if state['trades']:
        exits = [t for t in state['trades'] if t['action'] == 'EXIT']
        if exits:
            wins = sum(1 for t in exits if t['pnl'] > 0)
            total_pnl = sum(t['pnl'] for t in exits)
            print(f"\n交易统计: {len(exits)}笔平仓, 胜率{wins/len(exits)*100:.1f}%")
            print(f"累计PnL: {total_pnl:+,.0f}")
            print(f"\n最近5笔:")
            for t in state['trades'][-5:]:
                if t['action'] == 'EXIT':
                    print(f"  {t['date']} {t['symbol']} {t['reason']:<12} "
                          f"入{t['entry_price']:.0f} 出{t['exit_price']:.0f} "
                          f"PnL {t['pnl']:+,.0f}")
    print(f"{'='*70}")

def reset():
    """重置纸面交易"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    if os.path.exists(TRADE_LOG):
        os.remove(TRADE_LOG)
    print("纸面交易状态已重置")

def run_daemon():
    """常驻模式: 工作日 15:35 自动扫描，24小时保持运行"""
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"[daemon] 数据目录: {BASE_DIR}")
    print(f"[daemon] 工作日 15:35 自动扫描，Ctrl-C 退出")

    def _scan_once():
        try:
            run_daily_scan()
        except Exception as e:
            print(f"[daemon] 扫描异常: {e}")
            import traceback; traceback.print_exc()

    _scan_once()  # 启动先跑一次

    while True:
        now = datetime.now()
        # 工作日(周一~周五) 15:35 触发
        if now.weekday() < 5 and now.hour == 15 and now.minute == 35:
            _scan_once()
            _time.sleep(61)  # 避免同分钟内重复触发
        _time.sleep(30)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'scan':
            run_daily_scan()
        elif cmd == 'status':
            show_status()
        elif cmd == 'reset':
            reset()
        elif cmd == 'daemon':
            run_daemon()
        else:
            print(f"用法: python paper_trade.py [scan|status|reset|daemon]")
    else:
        run_daily_scan()
