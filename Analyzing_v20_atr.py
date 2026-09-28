import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import time

DATA_DIR = Path(".")
OUTPUT_DIR = DATA_DIR / "results"

ATR_PERIOD = 14
ATR_TIMEFRAME = "1h"
ATR_SMOOTHING = "wilder"
ATR_MIN_MULTIPLIER = 1.0
ATR_MAX_MULTIPLIER = 6.0
ATR_INTERVAL = 0.5
ATR_MULTIPLIERS = [
    round(ATR_MIN_MULTIPLIER + i * ATR_INTERVAL, 4)
    for i in range(int(round((ATR_MAX_MULTIPLIER - ATR_MIN_MULTIPLIER) / ATR_INTERVAL)) + 1)
]

SAME_CANDLE_RULE = "sl_first"
ESTIMATED_ROLLOVER_FEE_PERCENT_PER_DAY = 0.01
ROLLOVER_HOUR_UTC = 0

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


def choose_custom_windows():
    print("\n--- Custom Window Sizes ---")
    train_str = input("Train Window Size (e.g., 10): ").strip()
    train_size = int(train_str) if train_str.isdigit() else 10

    fwd_str = input("Forward Test Window Size (e.g., 5): ").strip()
    fwd_size = int(fwd_str) if fwd_str.isdigit() else 5

    return train_size, fwd_size


def choose_data_cutoff():
    print("\n--- Results Display Filter ---")
    pct_str = input("Keep top % of strategies by lowest |Avg PnL Delta| (e.g., 50 cuts the worst half): ").strip()
    try:
        pct = float(pct_str)
        if not 0 < pct <= 100:
            raise ValueError
    except ValueError:
        print("Invalid percentage. Keeping 100% of results.")
        return 100.0
    return pct


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
    df = pd.read_csv(file_path, sep="\t", header=None, names=["Datetime", "Open", "High", "Low", "Close", "Volume"], engine="c")
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
        line = input()
        if line.strip() == "" and lines:
            break
        lines.append(line)
    if not lines:
        print("No data pasted.")
        return None
    from io import StringIO
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


def calculate_pnl(entry_price, exit_price, direction):
    if direction == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def calculate_rollover_fee_percent(entry_time, exit_time):
    entry_day = (entry_time - pd.Timedelta(hours=ROLLOVER_HOUR_UTC)).date()
    exit_day = (exit_time - pd.Timedelta(hours=ROLLOVER_HOUR_UTC)).date()
    rollover_count = max(0, (exit_day - entry_day).days)
    return rollover_count * ESTIMATED_ROLLOVER_FEE_PERCENT_PER_DAY


def build_trade_result(entry_price, exit_price, direction, entry_time, exit_time, tp_distance_percent, sl_distance_percent):
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

        for tp_atr in ATR_MULTIPLIERS:
            for sl_atr in ATR_MULTIPLIERS:
                trades = []
                for direction, window in valid_entries:
                    trade = simulate_trade(window, direction, tp_atr, sl_atr)
                    if trade is not None:
                        trades.append(trade)
                if trades:
                    results.append(summarize_trades(trades, mode, sub_mode, pair, tp_atr, sl_atr))
    return pd.DataFrame(results)


METRIC_COLS = [
    "win_rate_percent", "total_pnl_percent", "average_pnl_percent",
    "average_hold_hours", "total_rollover_fee_percent",
    "average_rollover_fee_percent", "average_tp_distance_percent",
    "average_sl_distance_percent",
]

HIDDEN_FROM_DISPLAY = [
    "average_hold_hours", "total_rollover_fee_percent", "average_rollover_fee_percent",
]


def build_atr_reference(unit_sum, unit_count):
    rows = []
    mean_unit = (unit_sum / unit_count) if unit_count > 0 else 0.0
    for mult in ATR_MULTIPLIERS:
        rows.append({
            "atr_multiplier": mult,
            "atr_period": ATR_PERIOD,
            "atr_timeframe": ATR_TIMEFRAME,
            "avg_distance_percent": mult * mean_unit,
        })
    return pd.DataFrame(rows)


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


def main():
    mode = choose_trading_mode()
    if not mode:
        return

    group = choose_currency_group()
    if not group:
        return

    entry_time = choose_entry_time()
    inverted_event = choose_inverted_event() if mode == "surprise" else False

    train_window_size, fwd_window_size = choose_custom_windows()

    keep_pct = choose_data_cutoff()

    print(f"\n--- ATR Grid Search Settings ---")
    print(f"ATR({ATR_PERIOD}) computed on {ATR_TIMEFRAME} candles ({ATR_SMOOTHING} smoothing), read from last closed bar before entry")
    print(f"TP multipliers: {ATR_MULTIPLIERS}")
    print(f"SL multipliers: {ATR_MULTIPLIERS}")

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
    print(f"\nPasted data: {total_pasted}")
    print(f"Out of price data range: {out_of_range}")
    print(f"Fitting price data: {len(news_df)}")

    if mode == "surprise":
        news_df = news_df[news_df["Surprise_Type"] != "neutral"].reset_index(drop=True)
        print(f"Filtered out neutral events. Usable surprise events: {len(news_df)}")
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
        print(f"Filtered out events with no valid candle. Usable events: {len(news_df)}")

    total_usable = len(news_df)
    total_window = train_window_size + fwd_window_size

    if total_usable < total_window:
        print(f"\nError: You requested {train_window_size} train + {fwd_window_size} forward = {total_window} total events, but only {total_usable} usable events are available.")
        return

    recent_news_df = news_df.iloc[-total_window:].reset_index(drop=True)

    train_news = recent_news_df.iloc[:train_window_size].reset_index(drop=True)
    fwd_news = recent_news_df.iloc[train_window_size:].reset_index(drop=True)

    print(f"\nTotal usable events available: {total_usable}")
    print(f"Putting oldest {total_usable - total_window} events aside.")
    print(f"Using newest {total_window} events:")
    print(f"  - Train Window: {len(train_news)} events ({train_news['UTC_Time'].iloc[0].date()} to {train_news['UTC_Time'].iloc[-1].date()})")
    print(f"  - Forward Test Window: {len(fwd_news)} events ({fwd_news['UTC_Time'].iloc[0].date()} to {fwd_news['UTC_Time'].iloc[-1].date()})")

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
        return

    train_all = pd.concat(all_train_results, ignore_index=True) if all_train_results else pd.DataFrame()
    fwd_all = pd.concat(all_fwd_results, ignore_index=True) if all_fwd_results else pd.DataFrame()

    group_cols = ["mode", "sub_mode", "pair", "tp_atr", "sl_atr", "risk_to_reward_ratio"]

    train_all_sorted = train_all.sort_values("total_pnl_percent", ascending=False) if not train_all.empty else pd.DataFrame(columns=group_cols + METRIC_COLS)
    fwd_all_sorted = fwd_all.sort_values("total_pnl_percent", ascending=False) if not fwd_all.empty else pd.DataFrame(columns=group_cols + METRIC_COLS)

    merged_all = merge_train_fwd(train_all_sorted, fwd_all_sorted, group_cols)

    merged_all["avg_pnl_delta"] = (merged_all["average_pnl_percent_TRAIN"] - merged_all["average_pnl_percent_FWD"]).abs()

    if not merged_all.empty and keep_pct < 100.0:
        total_rows = len(merged_all)
        keep_count = max(1, int(np.ceil(total_rows * keep_pct / 100.0)))
        merged_all = (
            merged_all.sort_values("avg_pnl_delta", ascending=True)
            .head(keep_count)
            .reset_index(drop=True)
        )
        print(f"\n|Avg PnL Delta| cutoff ({keep_pct:g}%): kept {keep_count} of {total_rows} strategies.")

    def build_display_cols(source):
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

    display_cols = build_display_cols(merged_all)

    ranked = merged_all.copy()
    ranked = ranked.sort_values(["avg_pnl_delta", "pair"], ascending=[True, True]).reset_index(drop=True)
    ranked["number"] = range(1, len(ranked) + 1)

    all_res_display = ["number"] + [c for c in display_cols if c in ranked.columns]
    ranked = ranked[all_res_display]

    best_group_cols = ["pair", "sub_mode"] if mode == "candle_colour" else ["pair"]
    best_merged = (
        merged_all.sort_values(
            best_group_cols + ["avg_pnl_delta"],
            ascending=[True] * len(best_group_cols) + [True],
        )
        .groupby(best_group_cols, as_index=False)
        .first()
    )

    best_display = build_display_cols(best_merged)
    best_merged = best_merged[[c for c in best_display if c in best_merged.columns]]

    atr_reference_df = build_atr_reference(unit_sum, unit_count)

    clean_time = entry_time.replace(":", "")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{group}_{clean_time}_{timestamp}_v20_atr.xlsx"
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / output_file

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        best_merged.to_excel(writer, sheet_name="Best_Per_Pair", index=False)
        ranked.to_excel(writer, sheet_name="All_Results", index=False)
        atr_reference_df.to_excel(writer, sheet_name="ATR_Grid_Reference", index=False)
        train_news.to_excel(writer, sheet_name="Train_News", index=False)
        fwd_news.to_excel(writer, sheet_name="Fwd_News", index=False)
        for sheet_name in writer.sheets:
            writer.sheets[sheet_name].freeze_panes = "A2"

    print(f"\nDone! Results saved to {output_path}")
    print(f"\n=== Best Strategies (Train vs Forward Test) ===")
    print(best_merged.to_string(index=False))
    print(f"\n=== ATR Grid Reference (average realized distance per multiplier) ===")
    print(atr_reference_df.to_string(index=False))


if __name__ == "__main__":
    main()
