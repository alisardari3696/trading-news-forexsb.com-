import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
import re

DATA_DIR = Path(".")
OUTPUT_DIR = DATA_DIR / "results"

# ---- Dollar-based trade model -------------------------------------------------
TP_VALUES = [float(x) for x in range(1, 11)]   # take profit in USD: $1 ... $10
SL_VALUES = [float(x) for x in range(1, 11)]   # stop loss in USD:   $1 ... $10

LOT_SIZE = 0.01                                # every trade uses 0.01 lot
UNITS_PER_LOT = 100_000
POSITION_UNITS = LOT_SIZE * UNITS_PER_LOT      # 1,000 units of the base currency
INITIAL_FEE_USD = 0.05                         # charged once on every trade
ROLLOVER_FEE_USD_PER_NIGHT = 0.05              # charged for every night the trade stays open

# Only used if a quote-currency -> USD price file is missing (approximate, warns once)
FALLBACK_QUOTE_TO_USD = {"USD": 1.0, "EUR": 1.15, "GBP": 1.30, "JPY": 0.0067,
                         "AUD": 0.65, "NZD": 0.60, "CHF": 1.25, "CAD": 0.73}

SAME_CANDLE_RULE = "sl_first"
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


def choose_grid_settings():
    print("\n--- Window Grid Search Settings ---")
    min_train = int(input("Min Train Window Size (default 1): ").strip() or 1)
    max_train = int(input("Max Train Window Size (default 100): ").strip() or 100)
    
    min_fwd = int(input("Min Forward Test Window Size (default 1): ").strip() or 1)
    max_fwd = int(input("Max Forward Test Window Size (default 100): ").strip() or 100)

    cutoff_str = input("Top Lowest Delta Percentage Cutoff (e.g., 15 or 30) (default 50): ").strip()
    top_pct = float(cutoff_str) if cutoff_str.replace(".", "", 1).isdigit() else 50.0

    min_wr_str = input("Minimum Acceptable Win Rate % (e.g., 50) (default 50): ").strip()
    min_win_rate = float(min_wr_str) if min_wr_str.replace(".", "", 1).isdigit() else 50.0

    return min_train, max_train, min_fwd, max_fwd, top_pct, min_win_rate


_CSV_CACHE = {}


def load_csv(pair):
    if pair not in _CSV_CACHE:
        _CSV_CACHE[pair] = _load_csv_uncached(pair)
    return _CSV_CACHE[pair]


def _load_csv_uncached(pair):
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
    return df


def build_pair_arrays(pair, df):
    return {
        "pair": pair,
        "index": df.index,
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
    }


def index_position(index, timestamp):
    pos = index.searchsorted(timestamp, side="left")
    if pos < len(index) and index[pos] == timestamp:
        return int(pos)
    return -1


def has_candle(pair_arrays, timestamp):
    return index_position(pair_arrays["index"], timestamp) >= 0


FX_ARRAYS = {}          # pair -> price arrays (filled in main, extended lazily)
_FX_MISSING_WARNED = set()


def _fx_rate(pair, timestamp):
    """Open price of `pair` at/after timestamp, or None if the pair has no data."""
    if pair not in FX_ARRAYS:
        df = load_csv(pair) if (DATA_DIR / f"{pair}60.csv").exists() else None
        FX_ARRAYS[pair] = build_pair_arrays(pair, df) if df is not None else None
    arrays = FX_ARRAYS[pair]
    if arrays is None:
        return None
    pos = min(int(arrays["index"].searchsorted(timestamp, side="left")), len(arrays["index"]) - 1)
    return float(arrays["open"][pos])


def quote_to_usd(quote_ccy, timestamp):
    """USD value of 1 unit of the quote currency at the trade's entry time."""
    if quote_ccy == "USD":
        return 1.0
    rate = _fx_rate(f"{quote_ccy}USD", timestamp)
    if rate:
        return rate
    rate = _fx_rate(f"USD{quote_ccy}", timestamp)
    if rate:
        return 1.0 / rate
    if quote_ccy not in _FX_MISSING_WARNED:
        _FX_MISSING_WARNED.add(quote_ccy)
        print(f"WARNING: no {quote_ccy}USD / USD{quote_ccy} price file found - "
              f"using approximate {quote_ccy}->USD rate {FALLBACK_QUOTE_TO_USD[quote_ccy]}")
    return FALLBACK_QUOTE_TO_USD[quote_ccy]


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


def read_pasted_news():
    """Read the pasted news table once (all three runs reuse it)."""
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
    return "\n".join(lines)


def parse_news_data(raw, event_currency, entry_time_str, inverted_event, mode):
    from io import StringIO
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


def get_end_of_week_cutoff(entry_time):
    days_ahead = 4 - entry_time.weekday()
    if days_ahead < 0 or (days_ahead == 0 and entry_time.hour >= 18):
        days_ahead += 7
    target = entry_time + pd.Timedelta(days=days_ahead)
    return target.replace(hour=18, minute=0, second=0, microsecond=0)


def dollars_to_chart_units(pair, usd):
    """Convert a dollar TP/SL (0.01 lot) into pips, % of price and price distance, using the
    latest bar in the data as the reference price / FX rate (approximate for the next entry)."""
    arrays = FX_ARRAYS[pair]
    ref_price = float(arrays["close"][-1])
    q2usd = quote_to_usd(pair[3:], arrays["index"][-1])
    price_dist = usd / (POSITION_UNITS * q2usd)
    pip_size = 0.01 if pair.endswith("JPY") else 0.0001
    return {
        "pips": price_dist / pip_size,
        "pct": price_dist / ref_price * 100,
        "price_dist": price_dist,
        "ref_price": ref_price,
        "pip_value_usd": pip_size * POSITION_UNITS * q2usd,
        "decimals": 3 if pair.endswith("JPY") else 5,
    }


def build_trade_window(pair_arrays, entry_time):
    index = pair_arrays["index"]
    pos = index_position(index, entry_time)
    if pos < 0:
        return None
    cutoff = get_end_of_week_cutoff(entry_time)
    end = index.searchsorted(cutoff, side="right")
    if end <= pos:
        return None
    times = index[pos:end]
    # Rollover fee (USD) if the trade exits on each bar of the window
    shifted = times.values - np.timedelta64(ROLLOVER_HOUR_UTC, "h")
    day_num = shifted.astype("datetime64[D]").astype(np.int64)
    rollover_count = np.maximum(0, day_num - day_num[0])
    rollover_fee = rollover_count * ROLLOVER_FEE_USD_PER_NIGHT
    # USD profit/loss for a 1.0 price move on this pair with the fixed 0.01 lot position
    usd_per_price = POSITION_UNITS * quote_to_usd(pair_arrays["pair"][3:], entry_time)
    return {
        "entry_time": entry_time,
        "entry_price": float(pair_arrays["open"][pos]),
        "usd_per_price": usd_per_price,
        "times": times,
        "highs": pair_arrays["high"][pos:end],
        "lows": pair_arrays["low"][pos:end],
        "closes": pair_arrays["close"][pos:end],
        "rollover_fee": rollover_fee,
    }


TIE_DECIMALS = 6      # PnL / delta values equal to 6 decimals count as TIED (kills float noise)
ONE_SIDED_DELTA = 1e300  # delta rank for strategies with trades in only one of train / forward
EXIT_REASONS = ["TP", "SL", "SL (Same Candle)", "TP (Same Candle)", "Friday Cutoff"]
TP_ARR = np.array(TP_VALUES, dtype=float)
SL_ARR = np.array(SL_VALUES, dtype=float)


def simulate_trade_grid(window, direction):
    """Simulate every (TP $, SL $) combination for one trade window at once.

    TP / SL are dollar amounts for the fixed 0.01 lot position, converted to a price
    distance with this trade's USD-per-price value. The first bar that touches a TP level
    only depends on the TP, and the first bar that touches an SL level only depends on the SL,
    so each is found with one running max/min plus a binary search.
    Net PnL = gross USD - initial fee - rollover fee. Returns (net_pnl_usd, exit_code), both
    shaped (n_tp, n_sl).
    """
    entry = window["entry_price"]
    usd_per_price = window["usd_per_price"]
    highs, lows, closes = window["highs"], window["lows"], window["closes"]
    n = len(highs)

    tp_dist = TP_ARR / usd_per_price
    sl_dist = SL_ARR / usd_per_price

    if direction == "long":
        tp_prices = entry + tp_dist
        sl_prices = entry - sl_dist
        first_tp = np.searchsorted(np.maximum.accumulate(highs), tp_prices, side="left")
        first_sl = np.searchsorted(-np.minimum.accumulate(lows), -sl_prices, side="left")
        close_gross = (closes[-1] - entry) * usd_per_price
    else:
        tp_prices = entry - tp_dist
        sl_prices = entry + sl_dist
        first_tp = np.searchsorted(-np.minimum.accumulate(lows), -tp_prices, side="left")
        first_sl = np.searchsorted(np.maximum.accumulate(highs), sl_prices, side="left")
        close_gross = (entry - closes[-1]) * usd_per_price

    first_tp = np.where(first_tp >= n, -1, first_tp)
    first_sl = np.where(first_sl >= n, -1, first_sl)

    ft = first_tp[:, None]
    fs = first_sl[None, :]
    none_hit = (ft < 0) & (fs < 0)
    tp_wins = ~none_hit & ((fs < 0) | ((ft >= 0) & (ft < fs)))
    sl_wins = ~none_hit & ~tp_wins & ((ft < 0) | (fs < ft))
    same = ~none_hit & ~tp_wins & ~sl_wins
    sl_first = SAME_CANDLE_RULE == "sl_first"

    codes = np.full(none_hit.shape, 4, dtype=np.int8)
    codes[tp_wins] = 0
    codes[sl_wins] = 1
    codes[same] = 2 if sl_first else 3

    exit_idx = np.where(tp_wins, ft, fs)
    exit_idx = np.where(none_hit, n - 1, exit_idx)

    uses_tp_price = tp_wins | (same & (not sl_first))
    gross_usd = np.where(
        none_hit, close_gross,
        np.where(uses_tp_price, TP_ARR[:, None], -SL_ARR[None, :]),
    )
    pnl = gross_usd - INITIAL_FEE_USD - window["rollover_fee"][exit_idx]
    return pnl, codes


def _lex_pick(valid, tot, rr, delta=None):
    """Per row: among `valid` candidates pick highest total PnL, then highest TP/SL ratio,
    then (optionally) lowest delta, then lowest index. Returns -1 where nothing is valid."""
    any_valid = valid.any(axis=1)
    best = np.where(valid, tot, -np.inf).max(axis=1, keepdims=True)
    cand = valid & (tot == best)
    best_rr = np.where(cand, rr, -np.inf).max(axis=1, keepdims=True)
    cand &= (rr == best_rr)
    if delta is not None:
        best_d = np.where(cand, delta, np.inf).min(axis=1, keepdims=True)
        cand &= (delta == best_d)
    return np.where(any_valid, cand.argmax(axis=1), -1)


def select_strategies(pre_pnl, pre_n, pre_w, cand_rr, train_size, total_window,
                      c0, c1, top_pct, min_win_rate):
    """Pick one strategy for each iteration in [c0, c1) of a (train, forward) window pair.

    Returns (picked, status): picked is a candidate index or -1; status is 0 = picked,
    1 = no candidate strategies, 2 = no acceptable strategy.
    """
    a = slice(c0, c1)
    tr_end = slice(c0 + train_size, c1 + train_size)
    fw_end = slice(c0 + total_window, c1 + total_window)

    n_tr = pre_n[tr_end] - pre_n[a]
    n_fw = pre_n[fw_end] - pre_n[tr_end]
    has_tr = n_tr > 0
    has_fw = n_fw > 0
    pool = has_tr | has_fw
    pool_size = pool.sum(axis=1)

    tr_pnl = pre_pnl[tr_end] - pre_pnl[a]
    fw_pnl = pre_pnl[fw_end] - pre_pnl[tr_end]
    tot_pnl = np.round(tr_pnl + fw_pnl, TIE_DECIMALS)  # train + forward total PnL (noise-free for ties)

    # Reject a strategy only if its combined train + forward PnL is not positive
    acceptable = tot_pnl > 0
    if min_win_rate > 0:
        tr_wr = np.divide(pre_w[tr_end] - pre_w[a], n_tr, out=np.zeros_like(tr_pnl), where=has_tr) * 100
        fw_wr = np.divide(pre_w[fw_end] - pre_w[tr_end], n_fw, out=np.zeros_like(tr_pnl), where=has_fw) * 100
        acceptable &= (tr_wr >= min_win_rate) & (fw_wr >= min_win_rate)

    tr_avg = np.divide(tr_pnl, n_tr, out=np.zeros_like(tr_pnl), where=has_tr)
    fw_avg = np.divide(fw_pnl, n_fw, out=np.zeros_like(tr_pnl), where=has_fw)
    both = has_tr & has_fw
    delta = np.where(both, np.round(np.abs(tr_avg - fw_avg), TIE_DECIMALS),
                     np.where(pool, ONE_SIDED_DELTA, np.inf))

    # Shortlist = the K lowest-delta strategies in the pool (ties resolved by index order)
    k_per_row = np.ceil(pool_size * top_pct / 100.0).astype(np.int64)
    k_per_row = np.minimum(np.maximum(k_per_row, 1), np.maximum(pool_size, 1))
    short = np.zeros(pool.shape, dtype=bool)
    for k in np.unique(k_per_row):
        rows = np.flatnonzero(k_per_row == k)
        d = delta[rows]
        kth = np.partition(d, k - 1, axis=1)[:, k - 1:k]
        lt = d < kth
        need = k - lt.sum(axis=1, keepdims=True)
        eq = d == kth
        short[rows] = lt | (eq & (np.cumsum(eq, axis=1) <= need))

    picked = _lex_pick(short & acceptable, tot_pnl, cand_rr, delta)

    # Fallback: nothing acceptable in the shortlist -> search the whole pool
    miss = np.flatnonzero(picked < 0)
    if miss.size:
        picked[miss] = _lex_pick((pool & acceptable)[miss], tot_pnl[miss], cand_rr)

    status = np.where(pool_size == 0, 1, 2).astype(np.int8)
    status[picked >= 0] = 0
    return picked, status


def build_next_candidates_table(best_window, pre_pnl, pre_n, pre_w, cand_rr, candidate_keys,
                                top_pct, min_win_rate, top_n=15):
    """Top strategies of the lowest-delta shortlist for the NEXT event, ranked by train + forward
    total PnL, with the reason any of them was rejected. Returns (DataFrame, shortlist index set)."""
    train_size, fwd_size = best_window
    total_events = pre_pnl.shape[0] - 1
    c0 = total_events - (train_size + fwd_size)
    te = c0 + train_size

    n_tr = pre_n[te] - pre_n[c0]
    n_fw = pre_n[total_events] - pre_n[te]
    tr_pnl = pre_pnl[te] - pre_pnl[c0]
    fw_pnl = pre_pnl[total_events] - pre_pnl[te]
    tot = np.round(tr_pnl + fw_pnl, TIE_DECIMALS)
    has_tr, has_fw = n_tr > 0, n_fw > 0
    pool = has_tr | has_fw
    zeros = np.zeros_like(tr_pnl)
    tr_avg = np.divide(tr_pnl, n_tr, out=zeros.copy(), where=has_tr)
    fw_avg = np.divide(fw_pnl, n_fw, out=zeros.copy(), where=has_fw)
    tr_wr = np.divide(pre_w[te] - pre_w[c0], n_tr, out=zeros.copy(), where=has_tr) * 100
    fw_wr = np.divide(pre_w[total_events] - pre_w[te], n_fw, out=zeros.copy(), where=has_fw) * 100
    delta = np.where(has_tr & has_fw, np.round(np.abs(tr_avg - fw_avg), TIE_DECIMALS),
                     np.where(pool, ONE_SIDED_DELTA, np.inf))

    pool_idx = np.flatnonzero(pool)
    k = max(1, int(np.ceil(pool_idx.size * top_pct / 100.0)))
    shortlist = pool_idx[np.argsort(delta[pool_idx], kind="stable")][:k]
    ranked = shortlist[np.lexsort((-cand_rr[shortlist], -tot[shortlist]))]

    rows = []
    selected_done = False
    for rank, ci in enumerate(ranked[:top_n], start=1):
        reasons = []
        if not tot[ci] > 0:
            reasons.append("total PnL <= 0")
        if min_win_rate > 0 and tr_wr[ci] < min_win_rate:
            reasons.append("train win rate < min")
        if min_win_rate > 0 and fw_wr[ci] < min_win_rate:
            reasons.append("fwd win rate < min")
        passes = not reasons
        mark = ""
        if passes and not selected_done:
            mark, selected_done = "<- NEXT TRADE", True
        pair, sub_mode, tp, sl = candidate_keys[ci]
        rows.append({
            "Rank_By_Total_PnL": rank,
            "Pair": pair,
            "Sub_Mode": sub_mode,
            "TP_$": round(tp, 2),
            "SL_$": round(sl, 2),
            "TP_pips": round(dollars_to_chart_units(pair, tp)["pips"], 1),
            "SL_pips": round(dollars_to_chart_units(pair, sl)["pips"], 1),
            "Train_PnL_$": round(float(tr_pnl[ci]), 4),
            "Fwd_PnL_$": round(float(fw_pnl[ci]), 4),
            "Total_PnL_$": round(float(tot[ci]), 4),
            "Avg_PnL_Train_$": round(float(tr_avg[ci]), 4),
            "Avg_PnL_Fwd_$": round(float(fw_avg[ci]), 4),
            "Avg_PnL_Delta": round(float(delta[ci]), 4) if delta[ci] < ONE_SIDED_DELTA else None,
            "Train_WR_%": round(float(tr_wr[ci]), 1),
            "Fwd_WR_%": round(float(fw_wr[ci]), 1),
            "Passes_Filters": "yes" if passes else "no: " + ", ".join(reasons),
            "Selected": mark,
        })
    return pd.DataFrame(rows), set(int(x) for x in shortlist)


def build_next_trade_report(mode, group, entry_time_str, inverted_event, best_window, best_row,
                            pre_pnl, pre_n, pre_w, cand_rr, candidate_keys, history_strs,
                            top_pct, min_win_rate, shortlist):
    """Apply the best (train, forward) window to the most recent events and pick the
    strategy for the NEXT event. Returns a list of (field, value) rows."""
    train_size, fwd_size = best_window
    total_window = train_size + fwd_size
    total_events = pre_pnl.shape[0] - 1
    c0 = total_events - total_window
    train_end = c0 + train_size

    rows = [
        ("Best window (Train / Forward)", f"{train_size} / {fwd_size} events"),
        ("Backtest of this window: total PnL $", round(best_row["Actual_Total_PnL_USD"], 4)),
        ("Backtest of this window: trades", best_row["Total_Trades"]),
        ("Backtest of this window: win rate %", round(best_row["Actual_Win_Rate_Percent"], 2)),
        ("Backtest of this window: max drawdown $", round(best_row["Max_Drawdown_USD"], 4)),
        ("Position size", f"{LOT_SIZE} lot ({POSITION_UNITS:,.0f} units)"),
        ("Fees assumed", f"${INITIAL_FEE_USD:.2f} per trade + ${ROLLOVER_FEE_USD_PER_NIGHT:.2f} per night held"),
        ("Latest event in data", history_strs[-1]),
        ("Train events used", f"{history_strs[c0]} -> {history_strs[train_end - 1]} ({train_size})"),
        ("Forward events used", f"{history_strs[train_end]} -> {history_strs[-1]} ({fwd_size})"),
    ]

    picked, _ = select_strategies(
        pre_pnl, pre_n, pre_w, cand_rr, train_size, total_window,
        c0, c0 + 1, top_pct, min_win_rate,
    )
    pk = int(picked[0])
    if pk < 0:
        rows.append(("NEXT TRADE", "NO TRADE - no strategy passes the filters (positive total train + fwd PnL, min win rate) in this window"))
        return rows

    pair, sub_mode, tp, sl = candidate_keys[pk]

    # Stats of the chosen strategy over the train / forward events
    n_tr = pre_n[train_end, pk] - pre_n[c0, pk]
    n_fw = pre_n[total_events, pk] - pre_n[train_end, pk]
    tr_pnl = pre_pnl[train_end, pk] - pre_pnl[c0, pk]
    fw_pnl = pre_pnl[total_events, pk] - pre_pnl[train_end, pk]
    tr_wr = (pre_w[train_end, pk] - pre_w[c0, pk]) / n_tr * 100 if n_tr > 0 else 0.0
    fw_wr = (pre_w[total_events, pk] - pre_w[train_end, pk]) / n_fw * 100 if n_fw > 0 else 0.0

    # Direction rules for the next event
    if mode == "surprise":
        pos_dir = get_direction("surprise", pair, "positive", group)
        neg_dir = get_direction("surprise", pair, "negative", group)
        better = "Actual < Forecast" if inverted_event else "Actual > Forecast"
        worse = "Actual > Forecast" if inverted_event else "Actual < Forecast"
        direction_lines = [
            f"{better} -> {pos_dir.upper()} {pair}",
            f"{worse} -> {neg_dir.upper()} {pair}",
            "Actual = Forecast -> no trade",
        ]
        strategy_type = "Surprise (Actual vs Forecast)"
    else:
        green_dir = get_direction("candle_colour", pair, None, group, {"Open": 1.0, "Close": 2.0}, sub_mode)
        red_dir = get_direction("candle_colour", pair, None, group, {"Open": 2.0, "Close": 1.0}, sub_mode)
        direction_lines = [
            f"Previous 1h candle GREEN -> {green_dir.upper()} {pair}",
            f"Previous 1h candle RED -> {red_dir.upper()} {pair}",
            "Doji (open = close) -> no trade",
        ]
        strategy_type = f"Candle colour - {sub_mode.upper()}"

    cu_tp = dollars_to_chart_units(pair, tp)
    cu_sl = dollars_to_chart_units(pair, sl)
    same_candle = "SL assumed hit first" if SAME_CANDLE_RULE == "sl_first" else "TP assumed hit first"
    rows += [
        ("NEXT TRADE", f"{pair} | {strategy_type} | TP ${tp:.2f} | SL ${sl:.2f}"),
        ("Pair", pair),
        ("Strategy type", strategy_type),
        ("Direction rule", " ; ".join(direction_lines)),
        ("Take profit $", round(tp, 2)),
        ("Stop loss $", round(sl, 2)),
        ("Take profit pips", round(cu_tp["pips"], 1)),
        ("Stop loss pips", round(cu_sl["pips"], 1)),
        ("Take profit % of price", round(cu_tp["pct"], 4)),
        ("Stop loss % of price", round(cu_sl["pct"], 4)),
        ("Take profit / stop loss price distance", f"{cu_tp['price_dist']:.{cu_tp['decimals']}f} / {cu_sl['price_dist']:.{cu_sl['decimals']}f}"),
        ("Pip value for 0.01 lot (USD)", round(cu_tp["pip_value_usd"], 4)),
        ("Pips / % basis", f"Latest {pair} close in data = {cu_tp['ref_price']:.{cu_tp['decimals']}f} and current quote->USD rate. Dollar TP/SL are fixed; re-check pips/% at the real entry price. TP/SL are gross (fees not included in distance)."),
        ("TP / SL ratio", round(tp / sl, 3)),
        ("Entry", f"{entry_time_str} {ASSET_TIMEZONES[group]} time on the next event day (open of that 1h candle)"),
        ("Exit", f"TP / SL from entry price; if neither is hit, closed Friday 18:00 (price-data time). Same-candle hit: {same_candle}"),
        ("Strategy train: trades / PnL $ / win rate %", f"{int(n_tr)} / {tr_pnl:.4f} / {tr_wr:.1f}"),
        ("Strategy forward: trades / PnL $ / win rate %", f"{int(n_fw)} / {fw_pnl:.4f} / {fw_wr:.1f}"),
        ("Strategy total (train + fwd) PnL $", round(float(tr_pnl + fw_pnl), 4)),
        ("Picked from lowest-delta shortlist", "yes" if pk in shortlist else
         "NO - fallback: nothing in the shortlist passed the filters, so the whole pool was searched"),
    ]
    return rows


def find_best_3x3_block(heatmap_df):
    """Slide a 3x3 square over the PnL heatmap (Train_Size x Forward_Size).
    Every square whose 9 cells all exist is scored by the SUM of its 9 window PnLs.
    The square with the highest sum wins and the HIGHEST-PnL window inside it is the
    chosen window (on a tie inside the square the middle cell is preferred, then the
    first cell reading top-left to bottom-right).
    Returns (best_block, all_blocks). If the grid is too small for any full 3x3
    square, falls back to the single highest cell (best_block['fallback'] = True)."""
    vals = heatmap_df.to_numpy(dtype=float)
    rows = [int(r) for r in heatmap_df.index]
    cols = [int(c) for c in heatmap_df.columns]
    n_r, n_c = vals.shape

    blocks = []
    for i in range(1, n_r - 1):
        if rows[i + 1] - rows[i - 1] != 2:      # neighbouring train sizes must be adjacent
            continue
        for j in range(1, n_c - 1):
            if cols[j + 1] - cols[j - 1] != 2:  # neighbouring forward sizes must be adjacent
                continue
            sub = vals[i - 1:i + 2, j - 1:j + 2]
            if np.isnan(sub).any():
                continue
            peak = sub.max()
            if sub[1, 1] == peak:
                pi, pj = 1, 1
            else:
                pi, pj = (int(x) for x in np.argwhere(sub == peak)[0])
            blocks.append({
                "i": i, "j": j,                                   # middle of the square
                "center_train": rows[i], "center_fwd": cols[j],
                "center_pnl": float(vals[i, j]),
                "pick_i": i - 1 + pi, "pick_j": j - 1 + pj,       # highest window inside it
                "pick_train": rows[i - 1 + pi], "pick_fwd": cols[j - 1 + pj],
                "pick_pnl": float(peak),
                "block_sum": float(sub.sum()),
                "block_avg": float(sub.mean()),
                "block_min": float(sub.min()),
                "fallback": False,
            })

    if blocks:
        best = max(blocks, key=lambda b: (b["block_sum"], b["center_pnl"]))
        return best, blocks

    flat = int(np.nanargmax(vals))
    i, j = divmod(flat, n_c)
    best = {
        "i": i, "j": j,
        "center_train": rows[i], "center_fwd": cols[j],
        "center_pnl": float(vals[i, j]),
        "pick_i": i, "pick_j": j,
        "pick_train": rows[i], "pick_fwd": cols[j],
        "pick_pnl": float(vals[i, j]),
        "block_sum": float(vals[i, j]), "block_avg": float(vals[i, j]),
        "block_min": float(vals[i, j]),
        "fallback": True,
    }
    return best, []


def style_heatmap_sheet(ws, best, draw_block=True, notes=None, box="pick"):
    """Mark the chosen window on a heatmap sheet written by DataFrame.to_excel
    (headers in row 1 / column A, data starting at B2).
      - thick black box + bold = chosen window (highest window inside the best 3x3 square)
      - blue outline          = the best 3x3 square
    box="center" puts the thick box on the middle of the square instead (used on the sum sheet).
      - yellow axis headers   = chosen Train / Forward size"""
    from openpyxl.styles import Border, Side, Font, PatternFill

    bi, bj = best["i"], best["j"]                                   # middle of the square
    pi, pj = (best["pick_i"], best["pick_j"]) if box == "pick" else (bi, bj)
    br, bc = bi + 2, bj + 2      # Excel row / col of the middle of the square
    cr, cc = pi + 2, pj + 2      # Excel row / col of the boxed (chosen) window
    edges = {}

    def set_edge(r, c, side, style, color):
        edges.setdefault((r, c), {})[side] = Side(style=style, color=color)

    if draw_block:
        top, bottom, left, right = br - 1, br + 1, bc - 1, bc + 1
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if r == top:
                    set_edge(r, c, "top", "medium", "1F4E79")
                if r == bottom:
                    set_edge(r, c, "bottom", "medium", "1F4E79")
                if c == left:
                    set_edge(r, c, "left", "medium", "1F4E79")
                if c == right:
                    set_edge(r, c, "right", "medium", "1F4E79")

    # thick box around the chosen cell, set on the neighbours' shared edges too
    set_edge(cr, cc, "top", "thick", "000000")
    set_edge(cr, cc, "bottom", "thick", "000000")
    set_edge(cr, cc, "left", "thick", "000000")
    set_edge(cr, cc, "right", "thick", "000000")
    set_edge(cr - 1, cc, "bottom", "thick", "000000")
    set_edge(cr + 1, cc, "top", "thick", "000000")
    set_edge(cr, cc - 1, "right", "thick", "000000")
    set_edge(cr, cc + 1, "left", "thick", "000000")

    for (r, c), sides in edges.items():
        ws.cell(row=r, column=c).border = Border(**sides)

    ws.cell(row=cr, column=cc).font = Font(bold=True)
    yellow = PatternFill(start_color="FFD966", end_color="FFD966", fill_type="solid")
    for cell in (ws.cell(row=1, column=cc), ws.cell(row=cr, column=1)):
        cell.fill = yellow
        cell.font = Font(bold=True)

    ws.freeze_panes = "B2"

    if notes:
        note_row = ws.max_row + 2
        for k, text in enumerate(notes):
            ws.cell(row=note_row + k, column=1, value=text)


def calculate_max_drawdown(pnl_series):
    if len(pnl_series) == 0:
        return 0.0
    cum_pnl = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    return float(np.max(drawdowns))


RUN_LABEL = ""


def run_analysis(mode, group, entry_time, inverted_event, grid_settings, raw_news, label):
    """One full window-grid analysis for a single mode. Writes its own Excel report."""
    global RUN_LABEL
    RUN_LABEL = label
    min_train, max_train, min_fwd, max_fwd, top_pct_cutoff, min_acceptable_win_rate = grid_settings

    news_df = parse_news_data(raw_news, group, entry_time, inverted_event, mode)
    if news_df is None:
        return None

    pair_dfs = {}
    pair_arrays_map = {}
    FX_ARRAYS.clear()
    for pair in pair_sets[group]:
        df = load_csv(pair)
        if df is not None:
            pair_dfs[pair] = df
            pair_arrays_map[pair] = build_pair_arrays(pair, df)
            FX_ARRAYS[pair] = pair_arrays_map[pair]

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

    total_events = len(news_df)
    if total_events == 0:
        print("No usable events.")
        return

    # =========================================================================
    # PRE-CALCULATE ALL SINGLE-EVENT TRADES FOR SPEED
    # =========================================================================
    print("\nPre-computing trade simulation lookup table for all events & parameter combinations...")

    sub_modes = ["trend", "fade"] if mode == "candle_colour" else ["N/A"]

    candidate_keys = []
    for pair in pair_dfs:
        for sub_mode in sub_modes:
            for tp in TP_VALUES:
                for sl in SL_VALUES:
                    candidate_keys.append((pair, sub_mode, tp, sl))
    candidate_pos = {key: i for i, key in enumerate(candidate_keys)}
    cand_rr = np.array([key[2] / key[3] for key in candidate_keys], dtype=float)
    n_candidates = len(candidate_keys)

    metrics = np.zeros((3, total_events, n_candidates), dtype=float)
    exit_code = np.full((total_events, n_candidates), -1, dtype=np.int8)
    n_per_combo = len(TP_VALUES) * len(SL_VALUES)
    combo_base = {
        (pair, sub_mode): candidate_pos[(pair, sub_mode, TP_VALUES[0], SL_VALUES[0])]
        for pair in pair_dfs for sub_mode in sub_modes
    }
    news_rows = news_df.to_dict("records")

    for event_idx, nr in enumerate(news_rows):
        entry_time_utc = nr["UTC_Time"]
        surprise_type = nr["Surprise_Type"]

        for pair, pair_arrays in pair_arrays_map.items():
            window = build_trade_window(pair_arrays, entry_time_utc)
            if window is None:
                continue

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

                pnl_frac, codes = simulate_trade_grid(window, direction)  # net USD per (TP, SL)
                pnl_frac = pnl_frac.ravel()
                base = combo_base[(pair, sub_mode)]
                cols = slice(base, base + n_per_combo)
                metrics[0, event_idx, cols] = pnl_frac
                metrics[1, event_idx, cols] = 1.0
                metrics[2, event_idx, cols] = pnl_frac > 0
                exit_code[event_idx, cols] = codes.ravel()

    print(f"Pre-computation complete! Total simulated combinations cached: {int((exit_code >= 0).sum()):,}")

    pnl_cache = metrics[0].copy()  # single-event net PnL $ per (event, candidate)
    prefix = np.zeros((3, total_events + 1, n_candidates), dtype=float)
    np.cumsum(metrics, axis=1, out=prefix[:, 1:, :])
    del metrics

    pre_pnl, pre_n, pre_w = prefix[0], prefix[1], prefix[2]
    history_strs = [d.strftime("%Y-%m-%d") for d in news_df["History"]]
    chunk_rows = max(1, 400_000 // n_candidates)

    grid_results = []

    def evaluate_window(train_size, fwd_size):
        """Walk-forward run for one (train, forward) pair.
        Returns (picked_all, status_all, trades_executed, total_window)."""
        total_window = train_size + fwd_size
        num_iterations = total_events - total_window
        picked_all = np.empty(num_iterations, dtype=np.int64)
        status_all = np.empty(num_iterations, dtype=np.int8)

        for c0 in range(0, num_iterations, chunk_rows):
            c1 = min(num_iterations, c0 + chunk_rows)
            picked_all[c0:c1], status_all[c0:c1] = select_strategies(
                pre_pnl, pre_n, pre_w, cand_rr, train_size, total_window,
                c0, c1, top_pct_cutoff, min_acceptable_win_rate,
            )

        # Picked strategies whose trade was never simulated on the target event
        target_idx = np.arange(total_window, total_window + num_iterations)
        picked_pos = np.flatnonzero(picked_all >= 0)
        uncached = exit_code[target_idx[picked_pos], picked_all[picked_pos]] < 0
        status_all[picked_pos[uncached]] = 3
        picked_all[picked_pos[uncached]] = -1

        traded = picked_all >= 0
        trades_executed = pnl_cache[target_idx[traded], picked_all[traded]]
        return picked_all, status_all, trades_executed, total_window

    print("\nStarting Window Grid Search over specified bounds...")

    # Grid search loop over window sizes
    for train_size in range(min_train, max_train + 1):
        for fwd_size in range(min_fwd, max_fwd + 1):
            total_window = train_size + fwd_size
            min_req = total_window + 1

            if total_events < min_req:
                continue

            num_iterations = total_events - total_window
            _, _, trades_executed, _ = evaluate_window(train_size, fwd_size)
            total_trades = int(trades_executed.size)
            skipped_count = num_iterations - total_trades

            # Calculate Performance for this specific window size
            if total_trades > 0:
                actual_total_pnl = float(np.cumsum(trades_executed)[-1])
                actual_win_rate = (np.count_nonzero(trades_executed > 0) / total_trades) * 100
                actual_ev = actual_total_pnl / total_trades
            else:
                actual_total_pnl = 0.0
                actual_win_rate = 0.0
                actual_ev = 0.0
            max_dd = calculate_max_drawdown(trades_executed)

            grid_results.append({
                "Train_Size": train_size,
                "Forward_Size": fwd_size,
                "Train_Ratio_Percent": (train_size / total_window) * 100,
                "Total_Window": total_window,
                "Total_Trades": total_trades,
                "Skipped_Trades": skipped_count,
                "Actual_Win_Rate_Percent": actual_win_rate,
                "Actual_Total_PnL_USD": actual_total_pnl,
                "Actual_EV_USD": actual_ev,
                "Max_Drawdown_USD": max_dd
            })

    if not grid_results:
        print("No valid grid results produced.")
        return

    # SAVE RESULTS
    grid_df = pd.DataFrame(grid_results).sort_values("Actual_Total_PnL_USD", ascending=False)

    # Heatmap data: ForwardSize on X (Columns), TrainSize on Y (Index)
    heatmap_df = pd.DataFrame(grid_results).pivot(index="Train_Size", columns="Forward_Size", values="Actual_Total_PnL_USD")

    # WINDOW SELECTION: best 3x3 square (highest combined PnL of 9 neighbouring windows);
    # the highest-PnL window inside that square is the chosen window.
    best_block, all_blocks = find_best_3x3_block(heatmap_df)
    best_overall_window = (int(best_block["pick_train"]), int(best_block["pick_fwd"]))
    best_picked, best_status, _, best_total_window = evaluate_window(*best_overall_window)

    # Block tables: sum of every 3x3 square (placed on its middle cell) + ranking
    block_sum_df = pd.DataFrame(np.nan, index=heatmap_df.index, columns=heatmap_df.columns)
    for b in all_blocks:
        block_sum_df.loc[b["center_train"], b["center_fwd"]] = b["block_sum"]
    block_rank_df = pd.DataFrame(all_blocks)
    if not block_rank_df.empty:
        block_rank_df = (block_rank_df
                         .sort_values(["block_sum", "center_pnl"], ascending=False)
                         .reset_index(drop=True))
        block_rank_df.insert(0, "Rank", block_rank_df.index + 1)
        block_rank_df = block_rank_df.rename(columns={
            "center_train": "Center_Train_Size",
            "center_fwd": "Center_Forward_Size",
            "center_pnl": "Center_PnL_USD",
            "pick_train": "Best_Window_Train_Size",
            "pick_fwd": "Best_Window_Forward_Size",
            "pick_pnl": "Best_Window_PnL_USD",
            "block_sum": "Block_Sum_PnL_USD",
            "block_avg": "Block_Avg_PnL_USD",
            "block_min": "Block_Worst_Window_PnL_USD",
        })[["Rank", "Center_Train_Size", "Center_Forward_Size", "Center_PnL_USD",
            "Best_Window_Train_Size", "Best_Window_Forward_Size", "Best_Window_PnL_USD",
            "Block_Sum_PnL_USD", "Block_Avg_PnL_USD", "Block_Worst_Window_PnL_USD"]]

    status_messages = {
        1: "No candidate strategies",
        2: "No strategy with positive total (train + fwd) PnL",
        3: "Strategy outcome not cached",
    }
    log_rows = []
    for i in range(len(best_picked)):
        target = i + best_total_window
        pk = int(best_picked[i])
        if pk >= 0:
            pnl_val = float(pnl_cache[target, pk])
            log_rows.append({
                "Iteration": i + 1,
                "Date": history_strs[target],
                "Pair": candidate_keys[pk][0],
                "PnL_USD": pnl_val,
                "Is_Win": 1 if pnl_val > 0 else 0,
                "Reason": EXIT_REASONS[int(exit_code[target, pk])],
            })
        else:
            log_rows.append({
                "Iteration": i + 1,
                "Date": history_strs[target],
                "Pair": "None",
                "PnL_USD": 0.0,
                "Is_Win": 0,
                "Reason": status_messages[int(best_status[i])],
            })
    log_df = pd.DataFrame(log_rows)

    best_row = next(r for r in grid_results
                    if (r["Train_Size"], r["Forward_Size"]) == best_overall_window)
    candidates_df, shortlist = build_next_candidates_table(
        best_overall_window, pre_pnl, pre_n, pre_w, cand_rr, candidate_keys,
        top_pct_cutoff, min_acceptable_win_rate,
    )
    next_trade_rows = build_next_trade_report(
        mode, group, entry_time, inverted_event, best_overall_window, best_row,
        pre_pnl, pre_n, pre_w, cand_rr, candidate_keys, history_strs,
        top_pct_cutoff, min_acceptable_win_rate, shortlist,
    )
    if best_block["fallback"]:
        next_trade_rows[1:1] = [
            ("Window selection", "FALLBACK - grid too small for a full 3x3 square; used the single highest-PnL window"),
        ]
    else:
        t, f = int(best_block["center_train"]), int(best_block["center_fwd"])
        next_trade_rows[1:1] = [
            ("Window selection", "Highest-PnL window inside the 3x3 heatmap square with the highest combined total PnL"),
            ("Best 3x3 square: combined PnL $ (9 windows)", round(best_block["block_sum"], 4)),
            ("Best 3x3 square: Train sizes / Forward sizes", f"{t - 1}-{t + 1} / {f - 1}-{f + 1}"),
        ]
    next_trade_df = pd.DataFrame(next_trade_rows, columns=["Field", "Value"])

    # Sheets are NOT written here any more - they are returned so main() can put all runs
    # into ONE workbook.
    payload = {
        "Next_Trade": next_trade_df,
        "Next_Trade_Candidates": candidates_df,
        "Window_Grid_Summary": grid_df,
        "PnL_Heatmap": heatmap_df,
        "PnL_Heatmap_3x3_Sum": block_sum_df if all_blocks else None,
        "Block_Ranking": block_rank_df if all_blocks else None,
        "Best_Window_Trade_Log": log_df,
        "_best_block": best_block,
        "_best_window": best_overall_window,
        "_has_blocks": bool(all_blocks),
    }

    print(f"\n--- GRID SEARCH FINISHED ---")
    print(f"Best Window configuration found: {best_overall_window} (Train/Forward)")
    if best_block["fallback"]:
        print("Grid too small for a full 3x3 square - used the single highest-PnL window.")
    else:
        bi, bj = best_block["i"], best_block["j"]
        print(f"Best 3x3 square combined PnL $: {best_block['block_sum']:.4f} "
              f"(square centred on Train {best_block['center_train']} / Forward {best_block['center_fwd']})")
        print(f"Chosen window = highest inside the square: Train {best_overall_window[0]} / "
              f"Forward {best_overall_window[1]} (PnL $ {best_block['pick_pnl']:.4f})")
        print("Best 3x3 square (rows = Train size, cols = Forward size):")
        print(heatmap_df.iloc[bi - 1:bi + 2, bj - 1:bj + 2].round(2).to_string())
    print(f"Total entries in grid: {len(grid_df)}")
    total_skipped = sum(r["Skipped_Trades"] for r in grid_results)
    print(f"Total trades skipped (no strategy with positive total PnL): {total_skipped}")

    print("\n=== NEXT TRADE (best window applied to the latest events) ===")
    for field, value in next_trade_rows:
        print(f"{field}: {value}")
    print("\nTop strategies in the lowest-delta shortlist, ranked by train + forward total PnL:")
    print(candidates_df.head(10).to_string(index=False))

    return {
        "payload": payload,
        "best_window": f"{best_overall_window[0]} / {best_overall_window[1]}",
        "block_sum": None if best_block["fallback"] else round(best_block["block_sum"], 2),
        "next_trade": next((v for f, v in next_trade_rows if f == "NEXT TRADE"), ""),
        "backtest_pnl": round(float(best_row["Actual_Total_PnL_USD"]), 2),
        "trades": int(best_row["Total_Trades"]),
    }


SHEET_ORDER = [
    "Next_Trade", "Next_Trade_Candidates", "Window_Grid_Summary", "PnL_Heatmap",
    "PnL_Heatmap_3x3_Sum", "Block_Ranking", "Best_Window_Trade_Log",
]
RUN_PREFIX = {
    "surprise_higher_actual_better": "S_Better",
    "surprise_higher_actual_worse": "S_Worse",
    "candle_colour": "Candle",
}


def write_combined_workbook(group, results):
    """ONE Excel file. Sheets are grouped by type, so the three Next_Trade (signal) sheets
    sit right behind each other, then the three candidate sheets, and so on."""
    from openpyxl.formatting.rule import ColorScaleRule
    import openpyxl.utils
    import time

    ok = [(label, res) for label, res in results if res is not None]
    if not ok:
        print("No results to write.")
        return None

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{group}_{time.strftime('%Y%m%d_%H%M%S')}_ALL_RUNS_window_grid_search.xlsx"

    summary_df = pd.DataFrame([{
        "Run": RUN_PREFIX.get(label, label),
        "Mode": label,
        "Best_Window_Train_Forward": res["best_window"],
        "Backtest_PnL_USD": res["backtest_pnl"],
        "Backtest_Trades": res["trades"],
        "Best_3x3_Square_PnL_USD": res["block_sum"],
        "NEXT_TRADE": res["next_trade"] or "NO TRADE",
    } for label, res in ok])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ws = writer.sheets["Summary"]
        for col, w in zip("ABCDEFG", (16, 32, 26, 18, 16, 24, 70)):
            ws.column_dimensions[col].width = w

        for base in SHEET_ORDER:
            for label, res in ok:
                p = res["payload"]
                df = p[base]
                if df is None:
                    continue
                name = f"{RUN_PREFIX.get(label, label)}_{base}"[:31]
                keep_index = base in ("PnL_Heatmap", "PnL_Heatmap_3x3_Sum")
                df.to_excel(writer, sheet_name=name, index=keep_index)
                ws = writer.sheets[name]
                best_block = p["_best_block"]
                best_window = p["_best_window"]

                if base == "Next_Trade":
                    ws.column_dimensions["A"].width = 46
                    ws.column_dimensions["B"].width = 110

                elif base == "PnL_Heatmap":
                    col_letter = openpyxl.utils.get_column_letter(ws.max_column)
                    scale_lim = float(np.nanmax(np.abs(df.to_numpy(dtype=float)))) or 1.0
                    ws.conditional_formatting.add(
                        f"B2:{col_letter}{ws.max_row}",
                        ColorScaleRule(
                            start_type="num", start_value=-scale_lim, start_color="F8696B",
                            mid_type="num", mid_value=0, mid_color="FFFFFF",
                            end_type="num", end_value=scale_lim, end_color="63BE7B"))
                    if best_block["fallback"]:
                        notes = [
                            "Thick black box = chosen window (grid too small for a 3x3 square, highest single window used)",
                            f"Chosen window: Train {best_window[0]} / Forward {best_window[1]}",
                        ]
                    else:
                        notes = [
                            "Blue outline = best 3x3 square (highest combined PnL of its 9 windows)",
                            "Thick black box = chosen window (highest PnL inside that square)",
                            "Yellow headers = chosen Train / Forward size",
                            f"Chosen window: Train {best_window[0]} / Forward {best_window[1]}"
                            f" | 3x3 combined PnL $ = {best_block['block_sum']:.4f}",
                        ]
                    style_heatmap_sheet(ws, best_block, draw_block=not best_block["fallback"], notes=notes)

                elif base == "PnL_Heatmap_3x3_Sum":
                    col_letter2 = openpyxl.utils.get_column_letter(ws.max_column)
                    ws.conditional_formatting.add(
                        f"B2:{col_letter2}{ws.max_row}",
                        ColorScaleRule(
                            start_type="min", start_color="F8696B",
                            mid_type="percentile", mid_value=50, mid_color="FFFFFF",
                            end_type="max", end_color="63BE7B"))
                    style_heatmap_sheet(ws, best_block, draw_block=False, notes=[
                        "Each cell = combined PnL $ of the 3x3 square centred on that window (edges have no full square)",
                        "Thick black box = middle of the best square (its highest window, shown on PnL_Heatmap, is the chosen one)",
                    ], box="center")

    print(f"\nONE combined file saved: {out_path}")
    return out_path


# The three analyses that run from one set of answers and one news paste
RUNS = [
    ("surprise", False, "surprise_higher_actual_better"),
    ("surprise", True, "surprise_higher_actual_worse"),
    ("candle_colour", False, "candle_colour"),
]


def main():
    group = choose_currency_group()
    if not group:
        return

    entry_time = choose_entry_time()
    grid_settings = choose_grid_settings()

    raw_news = read_pasted_news()
    if raw_news is None:
        return

    results = []
    for mode, inverted, label in RUNS:
        print("\n" + "=" * 78)
        print(f"RUN: {label}")
        print("=" * 78)
        try:
            res = run_analysis(mode, group, entry_time, inverted, grid_settings, raw_news, label)
        except Exception:
            import traceback
            traceback.print_exc()
            res = None
            print(f"\n!! Run '{label}' failed - continuing with the next one.")
        results.append((label, res))

    write_combined_workbook(group, results)

    print("\n" + "=" * 78)
    print("SUMMARY OF ALL RUNS")
    print("=" * 78)
    for label, res in results:
        if res is None:
            print(f"\n[{label}] no result (see messages above)")
            continue
        print(f"\n[{label}]")
        print(f"  Best window (Train / Forward): {res['best_window']}")
        print(f"  Backtest of that window: ${res['backtest_pnl']} over {res['trades']} trades")
        print(f"  Best 3x3 square combined PnL $: {res['block_sum']}")
        print(f"  NEXT TRADE: {res['next_trade'] or 'n/a'}")


if __name__ == "__main__":
    main()
