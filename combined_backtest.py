"""
COMBINED FOREX NEWS BACKTEST
============================
One script that runs BOTH phases so you never have to re-run the second one.

PHASE 1  (window grid search - logic unchanged from backtesting_window_grid_atr.py)
    Paste news data -> grid search every Train/Forward window size -> simulate the
    picked strategy on the next event (walk-forward) -> report per-window:
        Actual_Win_Rate_Percent, Actual_Total_PnL_Percent, Actual_EV_Percent ...
    Then choose the BEST window:
        actual win rate >= minimum win rate (input)
        AND actual EV > 0.05 %
    If no window satisfies both -> no window -> stop.

PHASE 2  (analysis - logic from Analyzing_v20_atr.py, applied to the chosen window)
    Use the chosen Train/Forward window on the newest events.
    Cut the merged strategies by the lowest |Avg PnL Delta| (default: half).
    Sort the kept strategies by FORWARD total PnL (highest first).
    Roll down from the top and pick the FIRST strategy where:
        train win rate  > minimum win rate (input)   (both windows > 70 %)
        fwd   win rate  > minimum win rate
        train total PnL > 0  and  fwd total PnL > 0
        train average PnL > 0.05 %  and  fwd average PnL > 0.05 %
    That strategy is the one you backtest / trade.

Output: one Excel file with the final strategy, the window grid, heatmap,
the best-window trade log, all results, best-per-pair, ATR reference and news.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
from io import StringIO
import re
import time

DATA_DIR = Path(".")
OUTPUT_DIR = DATA_DIR / "results"

# ----------------------------------------------------------------------------
# Shared settings
# ----------------------------------------------------------------------------
ATR_PERIOD = 14
ATR_TIMEFRAME = "1h"
ATR_SMOOTHING = "wilder"

# PHASE 1 grid multipliers (unchanged from backtesting_window_grid_atr.py)
ATR_MIN_MULTIPLIER = 0.5
ATR_MAX_MULTIPLIER = 6.0
ATR_INTERVAL = 0.5
ATR_MULTIPLIERS = [
    round(ATR_MIN_MULTIPLIER + i * ATR_INTERVAL, 4)
    for i in range(int(round((ATR_MAX_MULTIPLIER - ATR_MIN_MULTIPLIER) / ATR_INTERVAL)) + 1)
]

# PHASE 2 analysis multipliers (unchanged from Analyzing_v20_atr.py)
ANALYSIS_ATR_MIN_MULTIPLIER = 1.0
ANALYSIS_ATR_MAX_MULTIPLIER = 6.0
ANALYSIS_ATR_INTERVAL = 0.5
ANALYSIS_ATR_MULTIPLIERS = [
    round(ANALYSIS_ATR_MIN_MULTIPLIER + i * ANALYSIS_ATR_INTERVAL, 4)
    for i in range(int(round((ANALYSIS_ATR_MAX_MULTIPLIER - ANALYSIS_ATR_MIN_MULTIPLIER) / ANALYSIS_ATR_INTERVAL)) + 1)
]

SAME_CANDLE_RULE = "sl_first"
ESTIMATED_ROLLOVER_FEE_PERCENT_PER_DAY = 0.01
ROLLOVER_HOUR_UTC = 0

# ---- acceptance thresholds ----
EV_MIN_PERCENT = 0.05          # PHASE 1: window's actual EV must be ABOVE this
MIN_AVG_PNL_PERCENT = 0.05     # PHASE 2: both average PnLs must be ABOVE this
DEFAULT_FORWARD_PNL_MIN = 0.0  # PHASE 2: both total PnLs must be ABOVE this (positive)

MET_PNL = 0
MET_COUNT = 1
MET_WINS = 2
MET_TOTAL = 3

ASSET_TIMEZONES = {
    "USD": "America/New_York",
    "GBP": "Europe/London",
    "EUR": "Europe/Berlin",
    "JPY": "Asia/Tokyo",
    "AUD": "Australia/Sydney",
    "NZD": "Pacific/Auckland",
    "CHF": "Europe/Zurich",
    "CAD": "America/Toronto",
}

pair_sets = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"],
    "EUR": ["EURUSD", "EURJPY", "EURGBP", "EURAUD", "EURNZD", "EURCHF", "EURCAD"],
    "GBP": ["GBPUSD", "GBPJPY", "EURGBP", "GBPAUD", "GBPNZD", "GBPCHF", "GBPCAD"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
    "AUD": ["AUDUSD", "AUDJPY", "EURAUD", "GBPAUD", "AUDNZD", "AUDCHF", "AUDCAD"],
    "NZD": ["NZDUSD", "NZDJPY", "EURNZD", "GBPNZD", "AUDNZD", "NZDCHF", "NZDCAD"],
    "CHF": ["USDCHF", "EURCHF", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF", "CHFJPY"],
    "CAD": ["USDCAD", "EURCAD", "GBPCAD", "AUDCAD", "NZDCAD", "CADCHF", "CADJPY"],
}


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
def choose_trading_mode():
    print("\nSelect Trading Mode:")
    print("1. Surprise (Actual vs Forecast)")
    print("2. Candle Colour (Tests Trend and Fade)")
    choice = input("Choice (1/2): ").strip()
    if choice == "1":
        return "surprise"
    if choice == "2":
        return "candle_colour"
    print("Invalid trading mode.")
    return None


def choose_currency_group():
    print("\nAvailable groups:", ", ".join(pair_sets.keys()))
    choice = input("Type currency group (example: GBP): ").upper().strip()
    if choice in pair_sets:
        return choice
    print("Invalid currency group.")
    return None


def choose_entry_time():
    return input("Enter trade entry time in local event timezone HH:MM: ").strip()


def choose_inverted_event():
    print("\nIs higher Actual worse for the currency? (y/n)")
    choice = input("Choice: ").lower().strip()
    return choice == "y"


def choose_grid_settings():
    print("\n--- Window Grid Search Settings ---")
    min_train = int(input("Min Train Window Size (default 1): ").strip() or 1)
    max_train = int(input("Max Train Window Size (default 35): ").strip() or 35)

    min_fwd = int(input("Min Forward Test Window Size (default 1): ").strip() or 1)
    max_fwd = int(input("Max Forward Test Window Size (default 35): ").strip() or 35)

    cutoff_str = input("Top Lowest Delta Percentage Cutoff (e.g., 15 or 30) (default 50): ").strip()
    top_pct = float(cutoff_str) / 100.0 if cutoff_str.replace(".", "", 1).isdigit() else 0.50

    min_wr_str = input("Minimum Acceptable Win Rate % (e.g., 50) (default 70): ").strip()
    min_win_rate = float(min_wr_str) if min_wr_str.replace(".", "", 1).isdigit() else 70.0

    return min_train, max_train, min_fwd, max_fwd, top_pct, min_win_rate


def choose_data_cutoff():
    print("\n--- Phase 2: Results Cutoff ---")
    pct_str = input("Keep top % of strategies by lowest |Avg PnL Delta| (default 50 = cut in half): ").strip()
    try:
        pct = float(pct_str)
        if not 0 < pct <= 100:
            raise ValueError
    except ValueError:
        return 50.0
    return pct


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def add_atr_column(df, period=ATR_PERIOD):
    out = df.copy()
    prev_close = out["Close"].shift(1)
    true_range = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    if ATR_SMOOTHING == "sma":
        out["ATR"] = true_range.rolling(period, min_periods=period).mean()
    else:
        out["ATR"] = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return out


def load_csv(pair):
    file_path = DATA_DIR / f"{pair}60.csv"
    if not file_path.exists():
        print(f"Missing CSV for {pair}: {file_path}")
        return None
    df = pd.read_csv(file_path, sep="\t", header=None,
                     names=["Datetime", "Open", "High", "Low", "Close", "Volume"], engine="c")
    required_cols = ["Datetime", "Open", "High", "Low", "Close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"{pair} missing columns: {missing_cols}")
        return None
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce").dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df.dropna(subset=required_cols)
        .sort_values("Datetime")
        .drop_duplicates("Datetime")
        .set_index("Datetime")
    )
    return add_atr_column(df)


def build_pair_arrays(pair, df):
    return {
        "pair": pair,
        "index": df.index,
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
        "atr": df["ATR"].to_numpy(),
    }


def index_position(index, timestamp):
    pos = index.searchsorted(timestamp, side="left")
    if pos < len(index) and index[pos] == timestamp:
        return int(pos)
    return -1


def has_candle(pair_arrays, timestamp):
    return index_position(pair_arrays["index"], timestamp) >= 0


def get_atr_at_entry(pair_arrays, entry_time):
    pos = index_position(pair_arrays["index"], entry_time)
    if pos <= 0:
        return None
    atr = float(pair_arrays["atr"][pos - 1])
    if not np.isfinite(atr) or atr <= 0.0:
        return None
    return atr


def get_direction(mode, pair, surprise, event_currency, candle=None, sub_mode=None):
    if mode == "surprise":
        if surprise == "neutral":
            return None
        if pair.startswith(event_currency):
            return "long" if surprise == "positive" else "short"
        if pair.endswith(event_currency):
            return "short" if surprise == "positive" else "long"
    elif mode == "candle_colour":
        if candle is None:
            return None
        if candle["Close"] > candle["Open"]:
            candle_colour = "green"
        elif candle["Close"] < candle["Open"]:
            candle_colour = "red"
        else:
            return None
        if sub_mode == "trend":
            return "long" if candle_colour == "green" else "short"
        if sub_mode == "fade":
            return "short" if candle_colour == "green" else "long"
    return None


def paste_news_data(event_currency, entry_time_str, inverted_event, mode):
    print("\n--- Paste News Data ---")
    print("Paste your tab-separated news data below (with header row).")
    print("When done pasting, enter an empty line:\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)
    if not lines:
        print("No data pasted.")
        return None
    raw = "\n".join(lines)
    df = pd.read_csv(StringIO(raw), sep="\t")
    if "History" not in df.columns:
        print("Missing 'History' column.")
        return None
    try:
        hour, minute = map(int, entry_time_str.split(":"))
    except ValueError:
        print("Invalid time format. Use HH:MM.")
        return None

    def parse_history_date(date_str):
        date_str = str(date_str).strip()
        date_str = re.sub(r'(\w+\s+\d+)-(\d+),\s*(\d{4})', r'\1, \3', date_str)
        return pd.to_datetime(date_str, format="%b %d, %Y", errors="coerce")

    df["History"] = df["History"].astype(str).apply(parse_history_date)

    if mode == "surprise":
        if "Actual" not in df.columns or "Forecast" not in df.columns:
            print("Surprise mode requires Actual and Forecast columns.")
            return None

        def parse_value(val):
            val = str(val).replace("%", "").replace(",", "").strip().lower()
            if val.endswith("k") or val.endswith("m"):
                return val[:-1]
            return val

        for col in ["Actual", "Forecast"]:
            df[col] = pd.to_numeric(df[col].astype(str).apply(parse_value), errors="coerce")
        df = df.dropna(subset=["History", "Actual", "Forecast"]).copy()
    else:
        df = df.dropna(subset=["History"]).copy()

    local_tz = ZoneInfo(ASSET_TIMEZONES[event_currency])
    utc_tz = ZoneInfo("UTC")

    df["UTC_Time"] = [
        pd.Timestamp(
            year=d.year, month=d.month, day=d.day,
            hour=hour, minute=minute, tz=local_tz,
        ).tz_convert(utc_tz).tz_localize(None)
        for d in df["History"]
    ]

    if mode == "surprise":
        adjusted_surprise = (df["Actual"] - df["Forecast"]) * (-1 if inverted_event else 1)
        df["Surprise_Type"] = adjusted_surprise.apply(
            lambda value: "positive" if value > 0 else "negative" if value < 0 else "neutral"
        )
    else:
        df["Surprise_Type"] = "neutral"

    return df.sort_values("UTC_Time").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Trade simulation
# ----------------------------------------------------------------------------
def calculate_pnl(entry_price, exit_price, direction):
    if direction == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def calculate_rollover_fee_percent(entry_time, exit_time):
    entry_day = (entry_time - pd.Timedelta(hours=ROLLOVER_HOUR_UTC)).date()
    exit_day = (exit_time - pd.Timedelta(hours=ROLLOVER_HOUR_UTC)).date()
    rollover_count = max(0, (exit_day - entry_day).days)
    return rollover_count * ESTIMATED_ROLLOVER_FEE_PERCENT_PER_DAY


def build_trade_result(entry_price, exit_price, direction, entry_time, exit_time,
                       tp_distance_percent, sl_distance_percent):
    hold_hours = (exit_time - entry_time).total_seconds() / 3600
    gross_pnl_pct = calculate_pnl(entry_price, exit_price, direction)
    rollover_fee_percent = calculate_rollover_fee_percent(entry_time, exit_time)
    rollover_fee_pct = rollover_fee_percent / 100
    return {
        "pnl_pct": gross_pnl_pct - rollover_fee_pct,
        "gross_pnl_pct": gross_pnl_pct,
        "rollover_fee_percent": rollover_fee_percent,
        "hold_hours": hold_hours,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "tp_distance_percent": tp_distance_percent,
        "sl_distance_percent": sl_distance_percent,
    }


def get_end_of_week_cutoff(entry_time):
    days_ahead = 4 - entry_time.weekday()
    if days_ahead < 0 or (days_ahead == 0 and entry_time.hour >= 18):
        days_ahead += 7
    target = entry_time + pd.Timedelta(days=days_ahead)
    return target.replace(hour=18, minute=0, second=0, microsecond=0)


def build_trade_window(pair_arrays, entry_time):
    index = pair_arrays["index"]
    pos = index_position(index, entry_time)
    if pos <= 0:
        return None
    atr = float(pair_arrays["atr"][pos - 1])
    if not np.isfinite(atr) or atr <= 0.0:
        return None
    cutoff = get_end_of_week_cutoff(entry_time)
    end = index.searchsorted(cutoff, side="right")
    if end <= pos:
        return None
    return {
        "entry_time": entry_time,
        "entry_price": float(pair_arrays["open"][pos]),
        "atr": atr,
        "times": index[pos:end],
        "highs": pair_arrays["high"][pos:end],
        "lows": pair_arrays["low"][pos:end],
        "closes": pair_arrays["close"][pos:end],
    }


def simulate_trade(window, direction, tp_atr, sl_atr):
    entry_price = window["entry_price"]
    atr = window["atr"]

    if direction == "long":
        tp_price = entry_price + tp_atr * atr
        sl_price = entry_price - sl_atr * atr
        tp_hits = window["highs"] >= tp_price
        sl_hits = window["lows"] <= sl_price
    else:
        tp_price = entry_price - tp_atr * atr
        sl_price = entry_price + sl_atr * atr
        tp_hits = window["lows"] <= tp_price
        sl_hits = window["highs"] >= sl_price

    tp_distance_percent = abs(tp_price - entry_price) / entry_price * 100
    sl_distance_percent = abs(entry_price - sl_price) / entry_price * 100

    tp_idx = np.flatnonzero(tp_hits)
    sl_idx = np.flatnonzero(sl_hits)
    first_tp = int(tp_idx[0]) if tp_idx.size else -1
    first_sl = int(sl_idx[0]) if sl_idx.size else -1

    if first_tp < 0 and first_sl < 0:
        res = build_trade_result(
            entry_price, float(window["closes"][-1]), direction,
            window["entry_time"], window["times"][-1],
            tp_distance_percent, sl_distance_percent,
        )
        res["exit_reason"] = "Friday Cutoff"
        return res

    if first_sl < 0 or (0 <= first_tp < first_sl):
        exit_idx, exit_price, reason = first_tp, tp_price, "TP"
    elif first_tp < 0 or first_sl < first_tp:
        exit_idx, exit_price, reason = first_sl, sl_price, "SL"
    elif SAME_CANDLE_RULE == "sl_first":
        exit_idx, exit_price, reason = first_sl, sl_price, "SL (Same Candle)"
    else:
        exit_idx, exit_price, reason = first_tp, tp_price, "TP (Same Candle)"

    res = build_trade_result(
        entry_price, exit_price, direction, window["entry_time"],
        window["times"][exit_idx], tp_distance_percent, sl_distance_percent,
    )
    res["exit_reason"] = reason
    return res


def calculate_max_drawdown(pnl_series):
    if len(pnl_series) == 0:
        return 0.0
    cum_pnl = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    return float(np.max(drawdowns))


def build_atr_reference(multipliers, unit_sum, unit_count):
    rows = []
    mean_unit = (unit_sum / unit_count) if unit_count > 0 else 0.0
    for mult in multipliers:
        rows.append({
            "atr_multiplier": mult,
            "atr_period": ATR_PERIOD,
            "atr_timeframe": ATR_TIMEFRAME,
            "avg_distance_percent": mult * mean_unit,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# PHASE 1  -  window grid search (logic unchanged)
# ----------------------------------------------------------------------------
def run_window_grid_search(news_df, pair_arrays_map, pair_dfs, mode, group,
                           min_train, max_train, min_fwd, max_fwd,
                           top_pct_cutoff, min_acceptable_win_rate):
    total_events = len(news_df)
    sub_modes = ["trend", "fade"] if mode == "candle_colour" else ["N/A"]

    candidate_keys = []
    for pair in pair_dfs:
        for sub_mode in sub_modes:
            for tp in ATR_MULTIPLIERS:
                for sl in ATR_MULTIPLIERS:
                    candidate_keys.append((pair, sub_mode, tp, sl))
    candidate_pos = {key: i for i, key in enumerate(candidate_keys)}
    cand_tp = np.array([key[2] for key in candidate_keys], dtype=float)
    cand_sl = np.array([key[3] for key in candidate_keys], dtype=float)
    cand_rr = cand_tp / cand_sl
    n_candidates = len(candidate_keys)

    print(f"\nPre-computing ATR trade simulation lookup table "
          f"({n_candidates:,} parameter combinations x {total_events} events)...")

    metrics = np.zeros((MET_TOTAL, n_candidates, total_events), dtype=float)
    trade_lookup = {}
    unit_sum = 0.0
    unit_count = 0
    news_rows = news_df.to_dict("records")

    for event_idx, nr in enumerate(news_rows):
        entry_time_utc = nr["UTC_Time"]
        surprise_type = nr["Surprise_Type"]

        for pair, pair_arrays in pair_arrays_map.items():
            window = build_trade_window(pair_arrays, entry_time_utc)
            if window is None:
                continue

            unit_sum += window["atr"] / window["entry_price"] * 100
            unit_count += 1

            if mode == "candle_colour":
                prev_pos = index_position(pair_arrays["index"], entry_time_utc - pd.Timedelta(hours=1))
                if prev_pos < 0:
                    continue
                candle = {
                    "Open": float(pair_arrays["open"][prev_pos]),
                    "Close": float(pair_arrays["close"][prev_pos]),
                }
            else:
                candle = None

            for sub_mode in sub_modes:
                direction = get_direction(
                    mode=mode, pair=pair, surprise=surprise_type,
                    event_currency=group, candle=candle,
                    sub_mode=sub_mode if sub_mode != "N/A" else None,
                )
                if direction is None:
                    continue

                for tp in ATR_MULTIPLIERS:
                    for sl in ATR_MULTIPLIERS:
                        res = simulate_trade(window, direction, tp, sl)
                        if res is None:
                            continue
                        ci = candidate_pos[(pair, sub_mode, tp, sl)]
                        metrics[MET_PNL, ci, event_idx] = res["pnl_pct"] * 100
                        metrics[MET_COUNT, ci, event_idx] = 1.0
                        if res["pnl_pct"] > 0:
                            metrics[MET_WINS, ci, event_idx] = 1.0
                        trade_lookup[(event_idx, ci)] = (
                            res["pnl_pct"] * 100,
                            res["exit_reason"],
                            res["tp_distance_percent"],
                            res["sl_distance_percent"],
                            res["hold_hours"],
                        )

    simulated_combinations = int(metrics[MET_COUNT].sum())
    print(f"Pre-computation complete! Total simulated trades cached: {simulated_combinations:,}")

    prefix = np.empty((MET_TOTAL, n_candidates, total_events + 1), dtype=float)
    prefix[:, :, 0] = 0.0
    np.cumsum(metrics, axis=2, out=prefix[:, :, 1:])
    del metrics

    pre_pnl = prefix[MET_PNL]
    pre_n = prefix[MET_COUNT]
    pre_w = prefix[MET_WINS]
    zeros_c = np.zeros(n_candidates, dtype=float)

    grid_results = []
    best_overall_log = []
    best_overall_pnl = -99999.0
    best_overall_window = None

    print("\nStarting Window Grid Search over specified bounds...")

    for train_size in range(min_train, max_train + 1):
        for fwd_size in range(min_fwd, max_fwd + 1):
            total_window = train_size + fwd_size
            min_req = total_window + 1

            if total_events < min_req:
                continue

            num_iterations = total_events - total_window
            current_trade_logs = []
            skipped_count = 0

            for i in range(num_iterations):
                train_start, train_end = i, i + train_size
                fwd_start, fwd_end = i + train_size, i + total_window
                target_idx = i + total_window

                target_event = news_df.iloc[target_idx]
                history_date = target_event["History"]

                n_tr = pre_n[:, train_end] - pre_n[:, train_start]
                n_fw = pre_n[:, fwd_end] - pre_n[:, fwd_start]
                pool = np.flatnonzero((n_tr > 0) | (n_fw > 0))

                if pool.size == 0:
                    skipped_count += 1
                    current_trade_logs.append({
                        "Iteration": i + 1,
                        "Date": history_date.strftime("%Y-%m-%d"),
                        "Pair": "None", "PnL_Percent": 0.0, "Is_Win": 0,
                        "Reason": "No candidate strategies",
                    })
                    continue

                tr_pnl = pre_pnl[:, train_end] - pre_pnl[:, train_start]
                fw_pnl = pre_pnl[:, fwd_end] - pre_pnl[:, fwd_start]
                tr_avg = np.divide(tr_pnl, n_tr, out=zeros_c.copy(), where=n_tr > 0)
                fw_avg = np.divide(fw_pnl, n_fw, out=zeros_c.copy(), where=n_fw > 0)
                tr_wr = np.divide(pre_w[:, train_end] - pre_w[:, train_start], n_tr,
                                  out=zeros_c.copy(), where=n_tr > 0) * 100
                fw_wr = np.divide(pre_w[:, fwd_end] - pre_w[:, fwd_start], n_fw,
                                  out=zeros_c.copy(), where=n_fw > 0) * 100

                avg_pnl_delta = np.abs(tr_avg - fw_avg)
                cutoff_count = max(1, int(np.ceil(pool.size * top_pct_cutoff)))
                by_delta = pool[np.argsort(avg_pnl_delta[pool], kind="stable")][:cutoff_count]
                ordered = by_delta[np.lexsort((-cand_rr[by_delta], -fw_avg[by_delta]))]

                acceptable = (
                    (tr_wr >= min_acceptable_win_rate)
                    & (fw_wr >= min_acceptable_win_rate)
                    & (tr_pnl > 0)
                    & (fw_pnl > 0)
                )

                picked = -1
                if acceptable[ordered].any():
                    picked = int(ordered[int(np.argmax(acceptable[ordered]))])
                else:
                    fallback = pool[np.lexsort((-cand_rr[pool], -fw_avg[pool]))]
                    if acceptable[fallback].any():
                        picked = int(fallback[int(np.argmax(acceptable[fallback]))])

                if picked < 0:
                    skipped_count += 1
                    current_trade_logs.append({
                        "Iteration": i + 1,
                        "Date": history_date.strftime("%Y-%m-%d"),
                        "Pair": "None", "PnL_Percent": 0.0, "Is_Win": 0,
                        "Reason": "No strategy with positive train & fwd PnL",
                    })
                    continue

                pair, sub_mode, tp, sl = candidate_keys[picked]
                cached = trade_lookup.get((target_idx, picked))
                if cached is None:
                    skipped_count += 1
                    current_trade_logs.append({
                        "Iteration": i + 1,
                        "Date": history_date.strftime("%Y-%m-%d"),
                        "Pair": "None", "PnL_Percent": 0.0, "Is_Win": 0,
                        "Reason": "Strategy outcome not cached",
                    })
                    continue

                pnl_val, exit_reason, tp_pct, sl_pct, hold_hours = cached
                current_trade_logs.append({
                    "Iteration": i + 1,
                    "Date": history_date.strftime("%Y-%m-%d"),
                    "Pair": pair, "Sub_Mode": sub_mode,
                    "TP_ATR": tp, "SL_ATR": sl,
                    "TP_Percent": tp_pct, "SL_Percent": sl_pct,
                    "Hold_Hours": hold_hours,
                    "PnL_Percent": pnl_val,
                    "Is_Win": 1 if pnl_val > 0 else 0,
                    "Reason": exit_reason,
                })

            trades_executed = [tl["PnL_Percent"] for tl in current_trade_logs if tl["Pair"] != "None"]
            total_trades = len(trades_executed)
            actual_total_pnl = sum(trades_executed)
            actual_win_rate = (sum([1 for pnl in trades_executed if pnl > 0]) / total_trades * 100) if total_trades > 0 else 0.0
            actual_ev = (actual_total_pnl / total_trades) if total_trades > 0 else 0.0
            max_dd = calculate_max_drawdown(trades_executed)

            grid_results.append({
                "Train_Size": train_size,
                "Forward_Size": fwd_size,
                "Train_Ratio_Percent": (train_size / total_window) * 100,
                "Total_Window": total_window,
                "Total_Trades": total_trades,
                "Skipped_Trades": skipped_count,
                "Actual_Win_Rate_Percent": actual_win_rate,
                "Actual_Total_PnL_Percent": actual_total_pnl,
                "Actual_EV_Percent": actual_ev,
                "Max_Drawdown_Percent": max_dd,
            })

            if actual_total_pnl > best_overall_pnl:
                best_overall_pnl = actual_total_pnl
                best_overall_log = current_trade_logs
                best_overall_window = (train_size, fwd_size)

    return grid_results, best_overall_log, best_overall_window, unit_sum, unit_count


# ----------------------------------------------------------------------------
# PHASE 2  -  analysis on the chosen window (logic from Analyzing_v20_atr.py)
# ----------------------------------------------------------------------------
METRIC_COLS = [
    "win_rate_percent", "total_pnl_percent", "average_pnl_percent",
    "average_hold_hours", "total_rollover_fee_percent",
    "average_rollover_fee_percent", "average_tp_distance_percent",
    "average_sl_distance_percent",
]

HIDDEN_FROM_DISPLAY = [
    "average_hold_hours", "total_rollover_fee_percent", "average_rollover_fee_percent",
]


def summarize_trades(trades, mode, sub_mode, pair, tp_atr, sl_atr):
    trade_count = len(trades)
    pnl_arr = np.empty(trade_count)
    hold_arr = np.empty(trade_count)
    rollover_arr = np.empty(trade_count)
    tp_dist_arr = np.empty(trade_count)
    sl_dist_arr = np.empty(trade_count)
    for i, t in enumerate(trades):
        pnl_arr[i] = t["pnl_pct"]
        hold_arr[i] = t["hold_hours"]
        rollover_arr[i] = t["rollover_fee_percent"]
        tp_dist_arr[i] = t["tp_distance_percent"]
        sl_dist_arr[i] = t["sl_distance_percent"]

    return {
        "mode": mode,
        "sub_mode": sub_mode or "N/A",
        "pair": pair,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "risk_to_reward_ratio": tp_atr / sl_atr,
        "win_rate_percent": float((pnl_arr > 0).sum()) / trade_count * 100,
        "total_pnl_percent": pnl_arr.sum() * 100,
        "average_pnl_percent": pnl_arr.mean() * 100,
        "average_hold_hours": hold_arr.mean(),
        "total_rollover_fee_percent": rollover_arr.sum(),
        "average_rollover_fee_percent": rollover_arr.mean(),
        "average_tp_distance_percent": tp_dist_arr.mean(),
        "average_sl_distance_percent": sl_dist_arr.mean(),
    }


def grid_search(pair, pair_arrays, mode, event_currency, news_df):
    results = []
    sub_modes = ["trend", "fade"] if mode == "candle_colour" else [None]
    news_rows = news_df.to_dict("records")

    for sub_mode in sub_modes:
        valid_entries = []
        for nr in news_rows:
            entry_time = nr["UTC_Time"]
            window = build_trade_window(pair_arrays, entry_time)
            if window is None:
                continue
            if mode == "candle_colour":
                prev_pos = index_position(pair_arrays["index"], entry_time - pd.Timedelta(hours=1))
                if prev_pos < 0:
                    continue
                candle = {
                    "Open": float(pair_arrays["open"][prev_pos]),
                    "Close": float(pair_arrays["close"][prev_pos]),
                }
            else:
                candle = None
            direction = get_direction(
                mode=mode, pair=pair, surprise=nr["Surprise_Type"],
                event_currency=event_currency, candle=candle, sub_mode=sub_mode,
            )
            if direction is not None:
                valid_entries.append((direction, window))

        for tp_atr in ANALYSIS_ATR_MULTIPLIERS:
            for sl_atr in ANALYSIS_ATR_MULTIPLIERS:
                trades = []
                for direction, window in valid_entries:
                    trade = simulate_trade(window, direction, tp_atr, sl_atr)
                    if trade is not None:
                        trades.append(trade)
                if trades:
                    results.append(summarize_trades(trades, mode, sub_mode, pair, tp_atr, sl_atr))
    return pd.DataFrame(results)


def merge_train_fwd(train_df, fwd_df, group_cols):
    merged = train_df.merge(
        fwd_df, on=group_cols, suffixes=("_TRAIN", "_FWD"), how="outer",
    )
    for col in METRIC_COLS:
        t_col = f"{col}_TRAIN"
        f_col = f"{col}_FWD"
        if t_col not in merged.columns:
            merged[t_col] = np.nan
        if f_col not in merged.columns:
            merged[f_col] = np.nan
    return merged


def build_display_cols(source, group_cols):
    cols = list(group_cols)
    for col in METRIC_COLS:
        if col in HIDDEN_FROM_DISPLAY:
            continue
        t_col = f"{col}_TRAIN"
        f_col = f"{col}_FWD"
        if t_col in source.columns:
            cols.append(t_col)
        if f_col in source.columns:
            cols.append(f_col)
        if col == "average_pnl_percent":
            cols.append("avg_pnl_delta")
    return cols


def run_analysis(news_df, pair_dfs, pair_arrays_map, mode, group,
                 train_window_size, fwd_window_size, keep_pct, min_win_rate):
    total_usable = len(news_df)
    total_window = train_window_size + fwd_window_size

    if total_usable < total_window:
        print(f"\nError: requested {train_window_size} train + {fwd_window_size} forward "
              f"= {total_window} events, but only {total_usable} usable events available.")
        return None

    recent_news_df = news_df.iloc[-total_window:].reset_index(drop=True)
    train_news = recent_news_df.iloc[:train_window_size].reset_index(drop=True)
    fwd_news = recent_news_df.iloc[train_window_size:].reset_index(drop=True)

    print(f"\nTotal usable events available: {total_usable}")
    print(f"Putting oldest {total_usable - total_window} events aside.")
    print(f"Using newest {total_window} events:")
    print(f"  - Train Window: {len(train_news)} events "
          f"({train_news['UTC_Time'].iloc[0].date()} to {train_news['UTC_Time'].iloc[-1].date()})")
    print(f"  - Forward Test Window: {len(fwd_news)} events "
          f"({fwd_news['UTC_Time'].iloc[0].date()} to {fwd_news['UTC_Time'].iloc[-1].date()})")

    print(f"\nRunning {mode} mode on TRAIN and FORWARD TEST splits...")
    all_train_results = []
    all_fwd_results = []
    unit_sum = 0.0
    unit_count = 0

    for pair, df in pair_dfs.items():
        print(f"Processing {pair}...")
        pair_arrays = pair_arrays_map[pair]
        train_results = grid_search(pair, pair_arrays, mode, group, train_news)
        fwd_results = grid_search(pair, pair_arrays, mode, group, fwd_news)
        if not train_results.empty:
            all_train_results.append(train_results)
        if not fwd_results.empty:
            all_fwd_results.append(fwd_results)

        for entry_row in pd.concat([train_news, fwd_news]).to_dict("records"):
            entry_time_utc = entry_row["UTC_Time"]
            pos = index_position(pair_arrays["index"], entry_time_utc)
            atr = get_atr_at_entry(pair_arrays, entry_time_utc)
            if atr is None or pos < 0:
                continue
            unit_sum += atr / float(pair_arrays["open"][pos]) * 100
            unit_count += 1

    if not all_train_results and not all_fwd_results:
        print("No results generated.")
        return None

    train_all = pd.concat(all_train_results, ignore_index=True) if all_train_results else pd.DataFrame()
    fwd_all = pd.concat(all_fwd_results, ignore_index=True) if all_fwd_results else pd.DataFrame()

    group_cols = ["mode", "sub_mode", "pair", "tp_atr", "sl_atr", "risk_to_reward_ratio"]

    train_all_sorted = (train_all.sort_values("total_pnl_percent", ascending=False)
                        if not train_all.empty else pd.DataFrame(columns=group_cols + METRIC_COLS))
    fwd_all_sorted = (fwd_all.sort_values("total_pnl_percent", ascending=False)
                      if not fwd_all.empty else pd.DataFrame(columns=group_cols + METRIC_COLS))

    merged_all = merge_train_fwd(train_all_sorted, fwd_all_sorted, group_cols)
    merged_all["avg_pnl_delta"] = (merged_all["average_pnl_percent_TRAIN"]
                                   - merged_all["average_pnl_percent_FWD"]).abs()

    merged_cut = merged_all
    if not merged_all.empty and keep_pct < 100.0:
        total_rows = len(merged_all)
        keep_count = max(1, int(np.ceil(total_rows * keep_pct / 100.0)))
        merged_cut = (merged_all.sort_values("avg_pnl_delta", ascending=True)
                      .head(keep_count).reset_index(drop=True))
        print(f"\n|Avg PnL Delta| cutoff ({keep_pct:g}%): kept {keep_count} of {total_rows} strategies.")

    display_cols = build_display_cols(merged_cut, group_cols)

    ranked = merged_cut.copy().sort_values(["avg_pnl_delta", "pair"], ascending=[True, True]).reset_index(drop=True)
    ranked["number"] = range(1, len(ranked) + 1)
    ranked = ranked[["number"] + [c for c in display_cols if c in ranked.columns]]

    best_group_cols = ["pair", "sub_mode"] if mode == "candle_colour" else ["pair"]
    best_merged = (
        merged_cut.sort_values(best_group_cols + ["avg_pnl_delta"],
                               ascending=[True] * len(best_group_cols) + [True])
        .groupby(best_group_cols, as_index=False).first()
    )
    best_display = build_display_cols(best_merged, best_group_cols)
    best_merged = best_merged[[c for c in best_display if c in best_merged.columns]]

    atr_reference_df = build_atr_reference(ANALYSIS_ATR_MULTIPLIERS, unit_sum, unit_count)

    return {
        "merged_cut": merged_cut,
        "ranked": ranked,
        "best_merged": best_merged,
        "train_news": train_news,
        "fwd_news": fwd_news,
        "atr_reference": atr_reference_df,
        "group_cols": group_cols,
        "display_cols": display_cols,
    }


def select_final_strategy(merged_cut, min_win_rate):
    """Sort kept strategies by FORWARD total PnL (highest first) and roll down
    to the first strategy passing every gate."""
    if merged_cut is None or merged_cut.empty:
        return None, None

    df = merged_cut.sort_values("total_pnl_percent_FWD", ascending=False).reset_index(drop=True)

    for i, row in df.iterrows():
        wr_t = row.get("win_rate_percent_TRAIN", np.nan)
        wr_f = row.get("win_rate_percent_FWD", np.nan)
        tp_t = row.get("total_pnl_percent_TRAIN", np.nan)
        tp_f = row.get("total_pnl_percent_FWD", np.nan)
        ap_t = row.get("average_pnl_percent_TRAIN", np.nan)
        ap_f = row.get("average_pnl_percent_FWD", np.nan)

        if (pd.notna(wr_t) and pd.notna(wr_f)
                and wr_t > min_win_rate and wr_f > min_win_rate
                and pd.notna(tp_t) and pd.notna(tp_f)
                and tp_t > DEFAULT_FORWARD_PNL_MIN and tp_f > DEFAULT_FORWARD_PNL_MIN
                and pd.notna(ap_t) and pd.notna(ap_f)
                and ap_t > MIN_AVG_PNL_PERCENT and ap_f > MIN_AVG_PNL_PERCENT):
            return i, row
    return None, None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    mode = choose_trading_mode()
    if not mode:
        return

    group = choose_currency_group()
    if not group:
        return

    entry_time = choose_entry_time()
    inverted_event = choose_inverted_event() if mode == "surprise" else False

    min_train, max_train, min_fwd, max_fwd, top_pct_cutoff, min_win_rate = choose_grid_settings()
    keep_pct = choose_data_cutoff()

    print("\n--- ATR Grid Search Settings ---")
    print(f"ATR({ATR_PERIOD}) on {ATR_TIMEFRAME} candles ({ATR_SMOOTHING} smoothing), last closed bar before entry")
    print(f"Phase 1 TP/SL multipliers: {ATR_MULTIPLIERS}")
    print(f"Phase 2 TP/SL multipliers: {ANALYSIS_ATR_MULTIPLIERS}")

    news_df = paste_news_data(group, entry_time, inverted_event, mode)
    if news_df is None:
        return

    pair_dfs = {}
    pair_arrays_map = {}
    for pair in pair_sets[group]:
        df = load_csv(pair)
        if df is not None:
            pair_dfs[pair] = df
            pair_arrays_map[pair] = build_pair_arrays(pair, df)

    if not pair_dfs:
        print("No valid pair data found.")
        return

    total_pasted = len(news_df)
    price_fit_mask = []
    for _, row in news_df.iterrows():
        entry_time_utc = row["UTC_Time"]
        has_price = any(has_candle(pair_arrays_map[p], entry_time_utc) for p in pair_arrays_map)
        price_fit_mask.append(has_price)
    news_df = news_df[price_fit_mask].reset_index(drop=True)
    out_of_range = total_pasted - len(news_df)
    print(f"\nPasted data rows: {total_pasted}")
    print(f"Out of price data range: {out_of_range}")
    print(f"Events matching price data: {len(news_df)}")

    if mode == "surprise":
        news_df = news_df[news_df["Surprise_Type"] != "neutral"].reset_index(drop=True)
        print(f"Filtered out neutral surprise events. Usable events remaining: {len(news_df)}")
    elif mode == "candle_colour":
        valid_mask = []
        for _, row in news_df.iterrows():
            entry_time_utc = row["UTC_Time"]
            prev_time_utc = entry_time_utc - pd.Timedelta(hours=1)
            is_valid = any(
                has_candle(pair_arrays_map[p], entry_time_utc) and has_candle(pair_arrays_map[p], prev_time_utc)
                for p in pair_arrays_map
            )
            valid_mask.append(is_valid)
        news_df = news_df[valid_mask].reset_index(drop=True)
        print(f"Filtered out events with no valid candle. Usable events remaining: {len(news_df)}")

    if len(news_df) == 0:
        print("No usable events.")
        return

    # ================= PHASE 1 =================
    print("\n" + "=" * 70)
    print("PHASE 1: WINDOW GRID SEARCH")
    print("=" * 70)

    grid_results, best_overall_log, best_overall_window, unit_sum, unit_count = run_window_grid_search(
        news_df, pair_arrays_map, pair_dfs, mode, group,
        min_train, max_train, min_fwd, max_fwd, top_pct_cutoff, min_win_rate,
    )

    if not grid_results:
        print("No valid grid results produced.")
        return

    grid_df = pd.DataFrame(grid_results).sort_values("Actual_Total_PnL_Percent", ascending=False).reset_index(drop=True)
    heatmap_df = pd.DataFrame(grid_results).pivot(
        index="Train_Size", columns="Forward_Size", values="Actual_Total_PnL_Percent")
    atr_reference_1 = build_atr_reference(ATR_MULTIPLIERS, unit_sum, unit_count)

    # ---- choose best window: actual WR >= min AND actual EV > 0.05 ----
    qualified = grid_df[
        (grid_df["Actual_Win_Rate_Percent"] >= min_win_rate)
        & (grid_df["Actual_EV_Percent"] > EV_MIN_PERCENT)
    ].copy()

    print("\n--- Window Selection ---")
    print(f"Windows meeting BOTH gates (WR >= {min_win_rate:g}% and EV > {EV_MIN_PERCENT:g}%): "
          f"{len(qualified)} of {len(grid_df)}")

    if qualified.empty:
        print("\nNo window met the required conditions -> NO WINDOW -> no strategy.")
        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{group}_{ts}_combined_NO_WINDOW.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            grid_df.to_excel(writer, sheet_name="Window_Grid_Summary", index=False)
            heatmap_df.to_excel(writer, sheet_name="PnL_Heatmap")
            pd.DataFrame(best_overall_log).to_excel(writer, sheet_name="Best_Window_Trade_Log", index=False)
            atr_reference_1.to_excel(writer, sheet_name="ATR_Grid_Reference", index=False)
        print(f"Diagnostic file saved: {out_path}")
        return

    qualified = qualified.sort_values("Actual_Total_PnL_Percent", ascending=False).reset_index(drop=True)
    best_row = qualified.iloc[0]
    train_size = int(best_row["Train_Size"])
    fwd_size = int(best_row["Forward_Size"])

    print("\nChosen window (best total PnL among qualifying):")
    print(f"  Train = {train_size}, Forward = {fwd_size}")
    print(f"  Actual Win Rate = {best_row['Actual_Win_Rate_Percent']:.2f}%")
    print(f"  Actual EV       = {best_row['Actual_EV_Percent']:.4f}%")
    print(f"  Actual Total PnL= {best_row['Actual_Total_PnL_Percent']:.2f}%")

    # ================= PHASE 2 =================
    print("\n" + "=" * 70)
    print(f"PHASE 2: ANALYSIS ON WINDOW Train={train_size} / Forward={fwd_size}")
    print("=" * 70)

    analysis = run_analysis(news_df, pair_dfs, pair_arrays_map, mode, group,
                            train_size, fwd_size, keep_pct, min_win_rate)
    if analysis is None:
        return

    merged_cut = analysis["merged_cut"]
    idx, strategy_row = select_final_strategy(merged_cut, min_win_rate)

    final_strategy_df = pd.DataFrame()
    if strategy_row is None:
        print("\nNo strategy passed all gates in the cut list -> NO TRADE.")
    else:
        print(f"\nFINAL STRATEGY FOUND at rank {idx + 1} of the kept (forward-PnL sorted) list:")
        cols_show = [c for c in analysis["display_cols"] if c in strategy_row.index]
        print(strategy_row[cols_show].to_string())
        final_strategy_df = pd.DataFrame([strategy_row[cols_show]])

    # ================= BACKTEST TRACKING (append to log, keep terminal open) =================
    BACKTEST_LOG = OUTPUT_DIR / "backtest_log.xlsx"
    LOG_SHEET = "Results"
    news_name = input("\nNews name (for log; press Enter to skip): ").strip()
    result_str = input("Trade result in pips (- means lost; press Enter to skip): ").strip()
    result_val = float(result_str) if result_str.replace("-", "").replace(".", "").isdigit() else None

    new_row = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Group": group, "Mode": mode,
        "Train_Size": train_size, "Forward_Size": fwd_size,
        "Keep_Pct": keep_pct, "Min_Win_Rate": min_win_rate,
        "News_Name": news_name,
        "Pair_Chosen": strategy_row.get("pair", "") if strategy_row is not None else "",
        "TP_ATR": strategy_row.get("tp_atr", "") if strategy_row is not None else "",
        "SL_ATR": strategy_row.get("sl_atr", "") if strategy_row is not None else "",
        "WR_Train": strategy_row.get("win_rate_percent_TRAIN", "") if strategy_row is not None else "",
        "WR_Fwd": strategy_row.get("win_rate_percent_FWD", "") if strategy_row is not None else "",
        "AvgPnL_Train": strategy_row.get("average_pnl_percent_TRAIN", "") if strategy_row is not None else "",
        "AvgPnL_Fwd": strategy_row.get("average_pnl_percent_FWD", "") if strategy_row is not None else "",
        "Result_Pips": result_val,
        "Result_Text": "WIN" if (result_val is not None and result_val > 0) else ("LOSS" if (result_val is not None and result_val < 0) else "SKIPPED"),
    }
    if BACKTEST_LOG.exists():
        existing = pd.read_excel(BACKTEST_LOG, sheet_name=LOG_SHEET)
        existing = pd.concat([pd.DataFrame([new_row]), existing], ignore_index=True)
    else:
        existing = pd.DataFrame([new_row])
    existing.to_excel(BACKTEST_LOG, sheet_name=LOG_SHEET, index=False)
    print(f"Backtest log updated: {BACKTEST_LOG} ({len(existing)} entries)")

    # Keep terminal open so the result stays visible
    print("\nPress Enter to close this terminal ...")
    try:
        input()
    except EOFError:
        pass

    # ================= OUTPUT =================
    ts = time.strftime("%Y%m%d_%H%M%S")
    clean_time = entry_time.replace(":", "")
    out_path = OUTPUT_DIR / f"{group}_{clean_time}_{ts}_combined_backtest.xlsx"
    OUTPUT_DIR.mkdir(exist_ok=True)

    window_choice_df = pd.DataFrame([{
        "Train_Size": train_size,
        "Forward_Size": fwd_size,
        "Actual_Win_Rate_Percent": best_row["Actual_Win_Rate_Percent"],
        "Actual_EV_Percent": best_row["Actual_EV_Percent"],
        "Actual_Total_PnL_Percent": best_row["Actual_Total_PnL_Percent"],
        "Min_Win_Rate_Required": min_win_rate,
        "EV_Threshold_Percent": EV_MIN_PERCENT,
        "Min_Avg_PnL_Percent": MIN_AVG_PNL_PERCENT,
        "Cutoff_Percent": keep_pct,
    }])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        final_strategy_df.to_excel(writer, sheet_name="FINAL_STRATEGY", index=False)
        window_choice_df.to_excel(writer, sheet_name="Chosen_Window", index=False)
        qualified.to_excel(writer, sheet_name="Qualified_Windows", index=False)
        grid_df.to_excel(writer, sheet_name="Window_Grid_Summary", index=False)
        heatmap_df.to_excel(writer, sheet_name="PnL_Heatmap")
        pd.DataFrame(best_overall_log).to_excel(writer, sheet_name="Best_Window_Trade_Log", index=False)
        analysis["ranked"].to_excel(writer, sheet_name="All_Results_Cut", index=False)
        analysis["best_merged"].to_excel(writer, sheet_name="Best_Per_Pair", index=False)
        atr_reference_1.to_excel(writer, sheet_name="ATR_Grid_Reference", index=False)
        analysis["atr_reference"].to_excel(writer, sheet_name="ATR_Reference_Analysis", index=False)
        analysis["train_news"].to_excel(writer, sheet_name="Train_News", index=False)
        analysis["fwd_news"].to_excel(writer, sheet_name="Fwd_News", index=False)

        for sheet_name in writer.sheets:
            writer.sheets[sheet_name].freeze_panes = "A2"

        from openpyxl.formatting.rule import ColorScaleRule
        import openpyxl.utils
        ws = writer.sheets["PnL_Heatmap"]
        col_letter = openpyxl.utils.get_column_letter(ws.max_column)
        cell_range = f"B2:{col_letter}{ws.max_row}"
        ws.conditional_formatting.add(
            cell_range,
            ColorScaleRule(
                start_type='num', start_value=-10, start_color='F8696B',
                mid_type='num', mid_value=0, mid_color='FFFFFF',
                end_type='num', end_value=10, end_color='63BE7B',
            )
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"File saved: {out_path}")
    if strategy_row is None:
        print("Result: NO TRADE (no strategy met every gate).")
    else:
        print("Result: TRADE the FINAL_STRATEGY sheet (row shown above).")


if __name__ == "__main__":
    main()
