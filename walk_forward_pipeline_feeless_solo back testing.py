"""
WALK-FORWARD RUNNER - SOLO MODES  (engine = combined_pipeline_feeless.py, feeless)
==================================================================================
Same walk-forward as walk_forward_pipeline_feeless.py, BUT instead of letting the
engine pick ONE winning mode out of a combined set, FOUR solo modes are tracked
INDEPENDENTLY inside the SAME walk-forward:

    1) Surprise_Better : higher Actual is BETTER for the currency
    2) Surprise_Worse  : higher Actual is WORSE for the currency
    3) Candle_Fade     : candle colour only - fade the move
    4) Candle_Trend    : candle colour only - trade with the move

How the walk-forward rolls (same idea as the original runner):

    rows  1..N  -> run the engine -> strategy -> trade it on row N+1 -> save result
    rows  2..N+1 -> run the engine -> strategy -> trade it on row N+2 -> save result
    ...
    until the last row has been traded.

At EVERY step each of the four solo modes does its OWN full pipeline:
    Stage 1: window grid search RESTRICTED to that single mode
             (surprise modes: one direction of surprise;
              candle modes : only 'fade' OR only 'trend')
             -> top 5 windows by total PnL -> highest EV wins (train/fwd sizes)
    Stage 2: Analyzing on the newest train+fwd events of the window
             -> cut by lowest |Avg PnL Delta| (default 50%) -> highest total PnL
             -> final strategy for that solo mode
    Trade  : the final strategy is applied to the NEXT unseen event.

Each solo mode therefore builds its OWN PnL curve, its own win/loss record, and
it never competes with the other modes. One walk-forward -> FOUR track records.

Row order: rows are sorted oldest -> newest by event time, so "row 1" is the
OLDEST news no matter in which order you pasted them.

Per-mode skip reasons (logged, walk-forward rolls on):
    NO_STRATEGY   that mode could not produce a strategy for the window
    SKIP_NEUTRAL  surprise mode: target has Actual == Forecast (or no values)
    SKIP_DOJI     candle mode: previous 1h candle has Open == Close
    SKIP_NO_PRICE no price data for the strategy pair at the target time
    ENGINE_ERROR  the engine crashed on that window (message is logged)

The terminal stays open: progress is printed after every solo-mode result, the
final report is printed, and you press Enter to exit. Ctrl+C stops early and
still saves + reports what is done.

Output (./results): one Excel file (Summary, Summary_By_Solo_Mode, Log,
By_Solo_Mode, By_Pair) + a CSV of the log (rewritten after every step).

PAPER-TRADING MODE (after the final report, answer Enter at the prompt):
    date of the event -> the 4 solo strategies are rebuilt from the news BEFORE that event
    -> you pick one (or skip) -> it is traded on that event and logged -> next round.
    Rounds 2+ ask asset + entry time again (Enter keeps them) and let you paste new news;
    grid settings / cut % / window are reused. A strategy with NEGATIVE total PnL is traded
    in reverse (see REVERSE_SWAPS_TP_SL). Ctrl+C or q finishes and saves
    ./results/PAPER_<stamp>_paper_trades.xlsx (+ CSV rewritten after every decision).

Needs in the SAME folder:
    combined_pipeline_feeless.py   (the engine)
    backtesting_window_grid.py
    backtesting_three_modes_feeless.py
    Analyzing_v20_feeless.py
    <PAIR>60.csv price files
"""
import contextlib
import io
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtesting_window_grid as bw            # noqa: E402
import combined_pipeline_feeless as cp          # noqa: E402  (the engine; also applies the feeless override)

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_WINDOW = 20
MAX_CONSECUTIVE_ERRORS = 5

# Paper-trading mode: when a chosen strategy has a NEGATIVE total PnL the trade is reversed.
# True  -> mirror trade: direction flipped AND TP/SL swapped (new TP = old SL, new SL = old TP)
# False -> direction flipped, same TP/SL
REVERSE_SWAPS_TP_SL = True

PAPER_COLUMNS = [
    "Trade_No", "Logged_At", "Round", "Currency_Group", "Entry_Time_Setting",
    "Target_Date", "Target_UTC", "Raw_Diff", "Actual", "Forecast",
    "Chosen_Solo_Mode", "Status", "Note",
    "Strategy_Mode", "Sub_Mode", "Pair",
    "Candle_Colour", "Candle_Open", "Candle_Close", "Candle_Body_Pct",
    "Reversed", "Strategy_Direction", "Traded_Direction",
    "Strategy_TP_Percent", "Strategy_SL_Percent", "Used_TP_Percent", "Used_SL_Percent", "Risk_Reward",
    "PnL_Percent", "Is_Win", "Exit_Reason", "Cumulative_PnL_Percent", "Cumulative_PnL_Mode",
    "Window_From", "Window_To", "Train_Size", "Forward_Size",
    "S1_Window_WinRate", "S1_Window_EV", "S1_Window_TotalPnL",
    "Strat_Train_WinRate", "Strat_Fwd_WinRate", "Strat_Train_PnL", "Strat_Fwd_PnL",
    "Strat_Total_PnL", "Strat_Avg_Delta",
    "All_Strategies",
]

# ----------------------------------------------------------------------------
# The four SOLO modes tracked independently by this walk-forward.
#   label        : name shown in the report / saved Excel
#   mode         : engine mode ("surprise" or "candle_colour")
#   inverted     : for surprise modes, True = higher Actual is WORSE
#   sub_modes    : for candle modes, restrict to one sub-mode ("fade" / "trend")
# ----------------------------------------------------------------------------
SOLO_MODES = [
    {"label": "Surprise_Better", "mode": "surprise",      "inverted": False, "sub_modes": None},
    {"label": "Surprise_Worse",  "mode": "surprise",      "inverted": True,  "sub_modes": None},
    {"label": "Candle_Fade",     "mode": "candle_colour", "inverted": False, "sub_modes": ["fade"]},
    {"label": "Candle_Trend",    "mode": "candle_colour", "inverted": False, "sub_modes": ["trend"]},
]

LOG_COLUMNS = [
    "Step", "Solo_Mode", "Target_Date", "Target_UTC", "Window_From", "Window_To",
    "Status", "Note",
    "Mode", "Inverted", "Train_Size", "Forward_Size",
    "Pair", "Candle_Body_Pct", "Sub_Mode", "Direction", "TP_Percent", "SL_Percent", "Risk_Reward",
    "PnL_Percent", "Is_Win", "Exit_Reason", "Cumulative_PnL_Percent",
    "S1_Window_WinRate", "S1_Window_EV", "S1_Window_TotalPnL",
    "Strat_Train_WinRate", "Strat_Fwd_WinRate",
    "Strat_Train_PnL", "Strat_Fwd_PnL", "Strat_Total_PnL", "Strat_Avg_Delta",
    "Engine_Seconds",
]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def ask_window(n_rows):
    raw = input(f"\nWalk-forward window size in news rows (default {DEFAULT_WINDOW}): ").strip()
    if not raw:
        return DEFAULT_WINDOW
    try:
        w = int(raw)
        if w < 10:
            raise ValueError
        return w
    except ValueError:
        print(f"Invalid window size. Using {DEFAULT_WINDOW}.")
        return DEFAULT_WINDOW


@contextlib.contextmanager
def quiet():
    """Silence the engine's own prints (it prints a lot per run)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def fmt_secs(s):
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def to_fraction(value, grid):
    """Engine reports TP/SL as percent (0.3 = 0.3%); the simulator wants a fraction (0.003).
    Match against the simulator's own TP/SL grid so the conversion is never a guess."""
    v = float(value)
    for cand in (v, v / 100.0):
        if any(abs(cand - g) < 1e-9 for g in grid):
            return cand
    return v / 100.0 if v >= 0.05 else v


def normalise_sub_mode(mode, raw):
    if mode != "candle_colour":
        return None
    s = str(raw).strip().lower()
    if "trend" in s:
        return "trend"
    if "fade" in s:
        return "fade"
    return None


def get_num(series, key):
    try:
        v = series[key]
        return float(v) if pd.notna(v) else np.nan
    except (KeyError, TypeError, ValueError):
        return np.nan


def candle_body_pct(pair_arrays, t, direction=None):
    """Body size in percent of the 1h candle just before the entry candle:
    entry 04:00 -> the 03:00-04:00 candle.
    Signed by the trade direction: positive when the candle colour agrees with
    the trade (buy + green / sell + red), negative when it fights it."""
    pos = bw.index_position(pair_arrays["index"], t - pd.Timedelta(hours=1))
    if pos < 0:
        return np.nan
    o = float(pair_arrays["open"][pos])
    c = float(pair_arrays["close"][pos])
    if not np.isfinite(o) or o == 0:
        return np.nan
    body = abs(c - o) / o * 100.0
    if direction is None or c == o:
        return body
    candle_up = c > o
    agrees = (direction == "long") == candle_up
    return body if agrees else -body


# ----------------------------------------------------------------------------
# Engine call: rows of ONE window, ONE solo mode -> strategy (or None + reason)
# ----------------------------------------------------------------------------
def run_engine(window_news, group, has_surprise, pair_dfs, pair_arrays_map,
               settings, keep_pct, solo):
    mode = solo["mode"]
    inverted = solo["inverted"]
    label = solo["label"]

    if mode == "surprise" and not has_surprise:
        return None, "no Actual/Forecast columns for surprise mode"

    mode_tuple = (mode, inverted, label)
    with quiet():
        results = cp.run_stage1(window_news, has_surprise, group, pair_arrays_map,
                               settings, modes=[mode_tuple], sub_modes=solo["sub_modes"])
        if not results:
            return None, "Stage 1 produced no window results"

        top = cp.pick_top_candidates(results)
        winner = top.iloc[0]
        win_mode = winner["Mode"]
        win_inverted = bool(winner["Inverted"])
        train_size = int(winner["Train_Size"])
        fwd_size = int(winner["Forward_Size"])

        news2 = cp.build_stage2_events(window_news, win_mode, win_inverted)
        s2 = cp.run_stage2(news2, win_mode, group, train_size, fwd_size, keep_pct,
                           pair_dfs, pair_arrays_map, sub_modes=solo["sub_modes"])
    if s2 is None:
        return None, "Stage 2 produced no results"

    return {
        "label": winner["Mode_Label"],
        "mode": win_mode,
        "inverted": win_inverted,
        "train_size": train_size,
        "fwd_size": fwd_size,
        "winner": winner,
        "final": s2["final"],
    }, ""


# ----------------------------------------------------------------------------
# Apply the engine's strategy to the next (unseen) event
# ----------------------------------------------------------------------------
def trade_target(info, target, group, pair_arrays_map, reverse=False, swap_tp_sl=True):
    final = info["final"]
    mode = info["mode"]
    pair = str(final["pair"])
    pa = pair_arrays_map.get(pair)
    out = {"Pair": pair, "Candle_Body_Pct": np.nan}
    if pa is None:
        return {**out, "Status": "SKIP_NO_PRICE", "Note": f"no price data loaded for {pair}"}

    t = target["UTC_Time"]
    out["Candle_Body_Pct"] = candle_body_pct(pa, t)
    sub_mode = normalise_sub_mode(mode, final.get("sub_mode", None))
    out["Sub_Mode"] = sub_mode if sub_mode else "N/A"

    # ---- direction ----
    if mode == "surprise":
        raw_diff = target.get("Raw_Diff", np.nan)
        if pd.isna(raw_diff):
            return {**out, "Status": "SKIP_NEUTRAL", "Note": "target has no Actual/Forecast"}
        adj = raw_diff * (-1 if info["inverted"] else 1)
        stype = "positive" if adj > 0 else ("negative" if adj < 0 else "neutral")
        if stype == "neutral":
            return {**out, "Status": "SKIP_NEUTRAL", "Note": "target Actual == Forecast"}
        direction = bw.get_direction(mode="surprise", pair=pair, surprise=stype,
                                     event_currency=group)
    else:
        if sub_mode is None:
            return {**out, "Status": "ENGINE_ERROR",
                    "Note": f"unknown candle sub_mode '{final.get('sub_mode', None)}'"}
        pp = bw.index_position(pa["index"], t - pd.Timedelta(hours=1))
        if pp < 0:
            return {**out, "Status": "SKIP_NO_PRICE", "Note": "previous 1h candle missing"}
        candle = {"Open": float(pa["open"][pp]), "Close": float(pa["close"][pp])}
        direction = bw.get_direction(mode="candle_colour", pair=pair, surprise="neutral",
                                     event_currency=group, candle=candle, sub_mode=sub_mode)
        if direction is None:
            return {**out, "Status": "SKIP_DOJI", "Note": "previous 1h candle Open == Close"}

    if direction is None:
        return {**out, "Status": "SKIP_NEUTRAL", "Note": "no direction"}

    tp_frac = to_fraction(final["tp_percent"], bw.TP_VALUES)
    sl_frac = to_fraction(final["sl_percent"], bw.SL_VALUES)
    strat_direction, strat_tp, strat_sl = direction, tp_frac, sl_frac
    if reverse:  # strategy had negative total PnL -> trade the opposite side
        direction = "short" if str(direction).lower() == "long" else "long"
        if swap_tp_sl:
            tp_frac, sl_frac = sl_frac, tp_frac
    out.update({"Reversed": bool(reverse), "Strategy_Direction": strat_direction,
                "Strategy_TP_Percent": strat_tp * 100, "Strategy_SL_Percent": strat_sl * 100})
    out["Candle_Body_Pct"] = candle_body_pct(pa, t, direction)

    # ---- simulate with the same simulator the engine used ----
    window = bw.build_trade_window(pa, t)
    if window is None:
        return {**out, "Status": "SKIP_NO_PRICE", "Note": "no price data at target entry time"}

    res = bw.simulate_trade(window, direction, tp_frac, sl_frac)
    pnl = res["pnl_pct"] * 100
    return {
        **out,
        "Status": "TRADED", "Note": "",
        "Direction": direction,
        "TP_Percent": tp_frac * 100, "SL_Percent": sl_frac * 100,
        "Risk_Reward": tp_frac / sl_frac if sl_frac else np.nan,
        "PnL_Percent": pnl, "Is_Win": int(pnl > 0), "Exit_Reason": res["exit_reason"],
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def group_table(log_df, col):
    traded = log_df[log_df["Status"] == "TRADED"]
    if traded.empty:
        return pd.DataFrame()
    g = traded.groupby(col)["PnL_Percent"]
    t = pd.DataFrame({
        "Trades": g.size(),
        "Wins": traded.groupby(col)["Is_Win"].sum(),
        "Total_PnL_Percent": g.sum(),
        "Avg_PnL_Percent": g.mean(),
    })
    t["Win_Rate_Percent"] = t["Wins"] / t["Trades"] * 100
    return t.sort_values("Total_PnL_Percent", ascending=False).reset_index()


def build_summary(log_df, window, n_rows, steps_total, interrupted):
    traded = log_df[log_df["Status"] == "TRADED"]
    pnl = traded["PnL_Percent"].to_numpy(dtype=float)
    n = len(pnl)
    wins = int((pnl > 0).sum())
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    s = {
        "News rows": n_rows,
        "Window size": window,
        "Steps planned": steps_total,
        "Solo-mode rows": len(log_df),
        "Trades taken": n,
        "Wins": wins,
        "Losses": n - wins,
        "Win rate %": (wins / n * 100) if n else 0.0,
        "Total PnL %": float(pnl.sum()) if n else 0.0,
        "Average PnL % per trade (EV)": float(pnl.mean()) if n else 0.0,
        "Best trade %": float(pnl.max()) if n else 0.0,
        "Worst trade %": float(pnl.min()) if n else 0.0,
        "Profit factor": (gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0),
        "Max drawdown %": bw.calculate_max_drawdown(list(pnl)),
    }
    for status, cnt in log_df["Status"].value_counts().items():
        if status != "TRADED":
            s[f"Not traded - {status}"] = int(cnt)
    return s


def build_summary_by_mode(log_df, window, n_rows, steps_total, interrupted):
    """One summary row per solo mode."""
    rows = []
    for label in log_df["Solo_Mode"].dropna().unique():
        sub = log_df[log_df["Solo_Mode"] == label]
        s = build_summary(sub, window, n_rows, steps_total, interrupted)
        s = {"Solo_Mode": label, **s}
        rows.append(s)
    return pd.DataFrame(rows)


def get_next_trade_signal(last_window_news, group, has_surprise, pair_dfs, pair_arrays_map,
                           settings, keep_pct, solo):
    """Run engine on the last window to get the strategy for the NEXT trade."""
    mode = solo["mode"]
    inverted = solo["inverted"]
    label = solo["label"]

    if mode == "surprise" and not has_surprise:
        return None, "no Actual/Forecast columns for surprise mode"

    mode_tuple = (mode, inverted, label)
    with quiet():
        results = cp.run_stage1(last_window_news, has_surprise, group, pair_arrays_map,
                               settings, modes=[mode_tuple], sub_modes=solo["sub_modes"])
        if not results:
            return None, "Stage 1 produced no window results"

        top = cp.pick_top_candidates(results)
        winner = top.iloc[0]
        win_mode = winner["Mode"]
        win_inverted = bool(winner["Inverted"])
        train_size = int(winner["Train_Size"])
        fwd_size = int(winner["Forward_Size"])

        news2 = cp.build_stage2_events(last_window_news, win_mode, win_inverted)
        s2 = cp.run_stage2(news2, win_mode, group, train_size, fwd_size, keep_pct,
                           pair_dfs, pair_arrays_map, sub_modes=solo["sub_modes"])
    if s2 is None:
        return None, "Stage 2 produced no results"

    return {
        "label": winner["Mode_Label"],
        "mode": win_mode,
        "inverted": win_inverted,
        "train_size": train_size,
        "fwd_size": fwd_size,
        "winner": winner,
        "final": s2["final"],
    }, ""


def format_next_signal(info, solo_label):
    """Format the next trade signal for display."""
    if info is None:
        return f"  {solo_label}: NO STRATEGY"
    
    final = info["final"]
    pair = str(final["pair"])
    sub_mode = normalise_sub_mode(info["mode"], final.get("sub_mode", None))
    
    tp_pct = float(final["tp_percent"])
    sl_pct = float(final["sl_percent"])
    
    # Convert to fractions for display consistency
    tp_frac = to_fraction(tp_pct, bw.TP_VALUES) * 100
    sl_frac = to_fraction(sl_pct, bw.SL_VALUES) * 100
    
    if info["mode"] == "surprise":
        direction_note = "higher Actual = BETTER" if not info["inverted"] else "higher Actual = WORSE"
        return (f"  {solo_label}: {pair} | Surprise ({direction_note}) | "
                f"Train={info['train_size']} Fwd={info['fwd_size']} | "
                f"TP={tp_frac:.2f}% SL={sl_frac:.2f}%")
    else:
        sub_display = sub_mode.capitalize() if sub_mode else "N/A"
        return (f"  {solo_label}: {pair} | Candle {sub_display} | "
                f"Train={info['train_size']} Fwd={info['fwd_size']} | "
                f"TP={tp_frac:.2f}% SL={sl_frac:.2f}%")


def print_report(summary, by_mode_summ, log_df, next_signals=None):
    print("\n" + "=" * 72)
    print("WALK-FORWARD RESULT (SOLO MODES)")
    print("=" * 72)
    print("--- Overall (all solo modes pooled) ---")
    for k, v in summary.items():
        if isinstance(v, float):
            v = "inf" if np.isinf(v) else f"{v:.4f}"
        print(f"  {k:<34}{v}")

    if not by_mode_summ.empty:
        cols = ["Solo_Mode", "Trades taken", "Wins", "Losses", "Win rate %",
                "Total PnL %", "Average PnL % per trade (EV)", "Profit factor", "Max drawdown %"]
        print("\n--- By solo mode ---")
        show = by_mode_summ[[c for c in cols if c in by_mode_summ.columns]].copy()
        for c in show.columns:
            if c != "Solo_Mode":
                show[c] = pd.to_numeric(show[c], errors="coerce")
        print(show.round(3).to_string(index=False))

    by_pair = group_table(log_df, "Pair")
    if not by_pair.empty:
        print("\n--- By pair ---")
        print(by_pair.round(3).to_string(index=False))

    if next_signals:
        print("\n" + "=" * 72)
        print("NEXT TRADE SIGNALS (run on last window for next news event)")
        print("=" * 72)
        for label, (info, reason) in next_signals.items():
            if info is None:
                print(f"  {label}: NO STRATEGY - {reason}")
            else:
                print(format_next_signal(info, label))


def save_outputs(log_df, summary, by_mode_summ, group, stamp, csv_path, next_signals=None):
    log_df.to_csv(csv_path, index=False)
    xlsx_path = bw.OUTPUT_DIR / f"{group}_{stamp}_walk_forward_solo_feeless.xlsx"
    summary_df = pd.DataFrame(
        [(k, ("inf" if isinstance(v, float) and np.isinf(v) else v)) for k, v in summary.items()],
        columns=["Metric", "Value"])
    
    # Build next signals sheet
    next_rows = []
    if next_signals:
        for label, (info, reason) in next_signals.items():
            if info is None:
                next_rows.append({"Solo_Mode": label, "Status": "NO_STRATEGY", "Note": reason,
                                  "Pair": "", "Mode": "", "Train_Size": "", "Forward_Size": "",
                                  "Sub_Mode": "", "Direction_Logic": "", "TP_Percent": "", "SL_Percent": ""})
            else:
                final = info["final"]
                pair = str(final["pair"])
                sub_mode = normalise_sub_mode(info["mode"], final.get("sub_mode", None))
                tp_pct = float(final["tp_percent"])
                sl_pct = float(final["sl_percent"])
                tp_frac = to_fraction(tp_pct, bw.TP_VALUES) * 100
                sl_frac = to_fraction(sl_pct, bw.SL_VALUES) * 100
                
                if info["mode"] == "surprise":
                    direction_logic = "higher Actual = BETTER" if not info["inverted"] else "higher Actual = WORSE"
                else:
                    direction_logic = f"Candle {sub_mode.capitalize() if sub_mode else 'N/A'}"
                
                next_rows.append({"Solo_Mode": label, "Status": "READY", "Note": "",
                                  "Pair": pair, "Mode": info["mode"], 
                                  "Train_Size": info["train_size"], "Forward_Size": info["fwd_size"],
                                  "Sub_Mode": sub_mode if sub_mode else "N/A", 
                                  "Direction_Logic": direction_logic,
                                  "TP_Percent": round(tp_frac, 2), "SL_Percent": round(sl_frac, 2)})
    
    next_df = pd.DataFrame(next_rows) if next_rows else pd.DataFrame()
    
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        by_mode_summ.to_excel(writer, sheet_name="Summary_By_Solo_Mode", index=False)
        log_df.to_excel(writer, sheet_name="Log", index=False)
        group_table(log_df, "Solo_Mode").to_excel(writer, sheet_name="By_Solo_Mode", index=False)
        group_table(log_df, "Pair").to_excel(writer, sheet_name="By_Pair", index=False)
        if not next_df.empty:
            next_df.to_excel(writer, sheet_name="Next_Trade_Signals", index=False)
        for name in writer.sheets:
            writer.sheets[name].freeze_panes = "A2"
    return xlsx_path


# ----------------------------------------------------------------------------
# PAPER-TRADING MODE (interactive; starts after the walk-forward report)
# ----------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_user_date(text):
    """'1 jul' / 'jul 1' / '2026-07-01' / '01/07' (day first) -> (year|None, month, day) or None."""
    s = re.sub(r"\s+", " ", text.strip().lower().replace(",", " "))
    y = mo = d = None
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})(?:[-/.](\d{2,4}))?", s)
        if m:
            d, mo = int(m[1]), int(m[2])
            if m[3]:
                y = int(m[3])
                y += 2000 if y < 100 else 0
        else:
            m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)? ?([a-z]{3,9}) ?(\d{4})?", s)
            if m and m[2][:3] in _MONTHS:
                d, mo = int(m[1]), _MONTHS[m[2][:3]]
                y = int(m[3]) if m[3] else None
            else:
                m = re.fullmatch(r"([a-z]{3,9}) ?(\d{1,2})(?:st|nd|rd|th)? ?(\d{4})?", s)
                if m and m[1][:3] in _MONTHS:
                    mo, d = _MONTHS[m[1][:3]], int(m[2])
                    y = int(m[3]) if m[3] else None
    if mo is None or not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return y, mo, d


def find_events_by_date(news_df, parsed):
    y, mo, d = parsed
    h = pd.to_datetime(news_df["History"])
    mask = (h.dt.month == mo) & (h.dt.day == d)
    if y is not None:
        mask &= (h.dt.year == y)
    return news_df.index[mask.to_numpy()].tolist()


def merge_news(old_df, new_df):
    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["UTC_Time"], keep="last")
    return df.sort_values("UTC_Time").reset_index(drop=True)


def paste_into(st, replace):
    """Paste news for the current asset/time. replace=True -> new news replaces the old."""
    parsed = cp.t3.paste_news(st["group"], st["entry_time"])
    if parsed is None:
        return False
    new_df, new_has = parsed
    new_df = new_df.sort_values("UTC_Time").reset_index(drop=True)
    if replace:
        st["news_df"], st["has_surprise"] = new_df, new_has
    else:
        st["news_df"] = merge_news(st["news_df"], new_df)
        st["has_surprise"] = bool(st["has_surprise"] and new_has)
    return True


def new_round_setup(st):
    """Rounds 2+: ask asset + entry time again (grid settings are NOT asked again), then news.
    Returns False when the user wants to finish."""
    print("\n" + "-" * 72)
    ans = input(f"Asset: {st['group']} | entry time: {st['entry_time']}.  "
                "Enter = keep, c = change, q = finish: ").strip().lower()
    if ans in ("q", "quit", "exit"):
        return False
    if ans.startswith("c"):
        group = bw.choose_currency_group()
        entry_time = bw.choose_entry_time() if group else None
        if not group or not entry_time:
            print("No new asset/time chosen - keeping the previous ones.")
        elif group != st["group"] or entry_time != st["entry_time"]:
            pair_dfs, pam = cp.load_pairs(group)
            if not pam:
                print("No valid pair data for that asset - keeping the previous ones.")
            else:
                old = dict(st)
                st.update({"group": group, "entry_time": entry_time,
                           "pair_dfs": pair_dfs, "pair_arrays_map": pam})
                print("\nDifferent asset/time -> paste the news for it (replaces the old news).")
                if not paste_into(st, replace=True):
                    st.clear()
                    st.update(old)
                    print("Nothing pasted - keeping the previous asset/time and news.")
                return True
    ans = input("Paste new news now? Enter = yes, n = no (use the news already loaded): ").strip().lower()
    if not ans.startswith("n"):
        if paste_into(st, replace=False):
            print(f"News loaded: {len(st['news_df'])} rows in total.")
        else:
            print("Nothing pasted - using the news already loaded.")
    return True


def ask_target(st, window):
    """Ask for the event date. Returns the row index in st['news_df'] or 'quit'."""
    pending = None
    while True:
        raw = pending or input("\nEvent date to trade (e.g. 1 Jul, 2026-07-01, 01/07)  [q = finish]: ").strip()
        pending = None
        if raw.lower() in ("q", "quit", "exit"):
            return "quit"
        if not raw:
            continue
        parsed = parse_user_date(raw)
        if parsed is None:
            print("  Could not read that date. Try '1 Jul', '2026-07-01' or '01/07' (day first).")
            continue
        hits = find_events_by_date(st["news_df"], parsed)
        if not hits:
            ans = input("  No event on that date in the loaded news. "
                        "Paste news now? Enter = yes, n = type another date: ").strip().lower()
            if ans.startswith("n"):
                continue
            if paste_into(st, replace=False):
                pending = raw          # search again with the same date
            else:
                print("  Nothing pasted.")
            continue
        idx = hits[0]
        if len(hits) > 1:
            print("  Several events match:")
            for k, i in enumerate(hits, 1):
                print(f"    {k}) {st['news_df']['History'].iloc[i]:%Y-%m-%d}  UTC {st['news_df']['UTC_Time'].iloc[i]}")
            pick = input("  Which one? number: ").strip()
            if not pick.isdigit() or not (1 <= int(pick) <= len(hits)):
                print("  Invalid choice.")
                continue
            idx = hits[int(pick) - 1]
        if idx < window:
            print(f"  Only {idx} news rows exist before that event, the strategies need {window}. "
                  "Paste older news or pick a later date.")
            continue
        return idx


def window_key(group, entry_time, win_news):
    return (group, str(entry_time), str(win_news["UTC_Time"].iloc[0]),
            str(win_news["UTC_Time"].iloc[-1]), len(win_news))


def signals_for_window(st, win_news, settings, keep_pct, cache):
    """The 4 solo-mode strategies computed ONLY from the rows before the target event."""
    key = window_key(st["group"], st["entry_time"], win_news)
    if key in cache:
        return cache[key]
    print(f"\nComputing the 4 strategies on the {len(win_news)} news rows before that event...")
    out = {}
    for solo in SOLO_MODES:
        t0 = time.time()
        try:
            info, why = get_next_trade_signal(win_news, st["group"], st["has_surprise"],
                                              st["pair_dfs"], st["pair_arrays_map"],
                                              settings, keep_pct, solo)
        except Exception as exc:
            info, why = None, f"{type(exc).__name__}: {exc}"
        out[solo["label"]] = (info, why)
        print(f"  {solo['label']} done ({time.time() - t0:.0f}s)")
    cache[key] = out
    return out


def signal_line(i, label, info, why):
    if info is None:
        return f"  {i}) {label}: NO STRATEGY - {why}"
    tot = get_num(info["final"], "Total_PnL_Percent")
    if np.isfinite(tot):
        action = "-> REVERSE the trade" if tot < 0 else "-> trade as is"
        tot_txt = f"{tot:+.2f}%"
    else:
        action, tot_txt = "-> trade as is", "n/a"
    return f"  {i}) {format_next_signal(info, label).strip()} | total PnL {tot_txt} {action}"


def print_signal_menu(signals):
    print("\n" + "=" * 72)
    print("STRATEGIES FOR THIS EVENT (built only from news before it)")
    print("=" * 72)
    for i, solo in enumerate(SOLO_MODES, 1):
        info, why = signals[solo["label"]]
        print(signal_line(i, solo["label"], info, why))


def parse_solo_choice(raw):
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if s.isdigit() and 1 <= int(s) <= len(SOLO_MODES):
        return SOLO_MODES[int(s) - 1]
    for solo in SOLO_MODES:
        if s == solo["label"].lower():
            return solo
    for key, label in (("better", "Surprise_Better"), ("worse", "Surprise_Worse"),
                       ("fade", "Candle_Fade"), ("trend", "Candle_Trend")):
        if key in s:
            return next(m for m in SOLO_MODES if m["label"] == label)
    return None


def ask_solo_choice(signals):
    """Returns a solo dict, 'skip' or 'quit'."""
    while True:
        raw = input("\nWhich strategy? number or name (e.g. 3, 'candle trend')  "
                    "[s = skip this event, q = finish]: ").strip()
        low = raw.lower()
        if low in ("q", "quit", "exit"):
            return "quit"
        if low in ("s", "skip"):
            return "skip"
        solo = parse_solo_choice(raw)
        if solo is None:
            print("  Not recognised.")
            continue
        if signals[solo["label"]][0] is None:
            print(f"  {solo['label']} has no strategy for this event - pick another or skip.")
            continue
        return solo


def prev_candle_info(pa, t):
    """Colour/open/close of the 1h candle before the entry candle."""
    pos = bw.index_position(pa["index"], t - pd.Timedelta(hours=1))
    if pos < 0:
        return "n/a", np.nan, np.nan
    o, c = float(pa["open"][pos]), float(pa["close"][pos])
    return ("green" if c > o else ("red" if c < o else "doji")), o, c


def describe_trade(rec):
    if rec["Status"] != "TRADED":
        return f"NOT TRADED - {rec['Status']}: {rec['Note']}"
    tag = " [REVERSED]" if rec["Reversed"] == "Yes" else ""
    return (f"{rec['Pair']} {str(rec['Traded_Direction']).upper()}{tag} "
            f"TP{rec['Used_TP_Percent']:.2f}% SL{rec['Used_SL_Percent']:.2f}% "
            f"-> {rec['Exit_Reason']} {rec['PnL_Percent']:+.3f}%")


def paper_trade_session(st, window, settings, keep_pct, stamp, cache):
    """Loop: date -> 4 strategies -> pick one (or skip) -> trade + log -> next round.
    Grid settings / cut % / window are reused. Ctrl+C or 'q' finishes. Returns the log rows."""
    bw.OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = bw.OUTPUT_DIR / f"PAPER_{stamp}_paper_trades_log.csv"
    rows = []
    cum_all = 0.0
    cum_mode = {m["label"]: 0.0 for m in SOLO_MODES}
    round_no = 0

    print("\n" + "=" * 72)
    print("PAPER-TRADING MODE")
    print("Pick the event date, see the 4 strategies, choose one (or skip).")
    print("A strategy whose total PnL is NEGATIVE is traded in REVERSE"
          + (" (direction flipped, TP and SL swapped)." if REVERSE_SWAPS_TP_SL
             else " (direction flipped, same TP/SL)."))
    print("Every decision is logged to disk. Ctrl+C or 'q' finishes and shows the final log.")
    print("=" * 72)

    try:
        while True:
            round_no += 1
            if round_no > 1 and not new_round_setup(st):
                break
            idx = ask_target(st, window)
            if idx == "quit":
                break
            target = st["news_df"].iloc[idx]
            win_news = st["news_df"].iloc[idx - window:idx].reset_index(drop=True)
            signals = signals_for_window(st, win_news, settings, keep_pct, cache)
            print_signal_menu(signals)
            choice = ask_solo_choice(signals)
            if choice == "quit":
                break

            rec = {c: np.nan for c in PAPER_COLUMNS}
            rec.update({
                "Trade_No": len(rows) + 1,
                "Logged_At": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Round": round_no,
                "Currency_Group": st["group"],
                "Entry_Time_Setting": str(st["entry_time"]),
                "Target_Date": target["History"].strftime("%Y-%m-%d"),
                "Target_UTC": target["UTC_Time"],
                "Raw_Diff": get_num(target, "Raw_Diff"),
                "Actual": target.get("Actual", np.nan),
                "Forecast": target.get("Forecast", np.nan),
                "Window_From": win_news["History"].iloc[0].strftime("%Y-%m-%d"),
                "Window_To": win_news["History"].iloc[-1].strftime("%Y-%m-%d"),
                "All_Strategies": " || ".join(
                    signal_line(i, m["label"], *signals[m["label"]]).strip()
                    for i, m in enumerate(SOLO_MODES, 1)),
            })

            if choice == "skip":
                rec.update({"Status": "SKIPPED_BY_USER", "Note": "no trade taken on this event"})
                rec["Cumulative_PnL_Percent"] = cum_all
            else:
                label = choice["label"]
                info = signals[label][0]
                f, w = info["final"], info["winner"]
                strat_total = get_num(f, "Total_PnL_Percent")
                reverse = bool(np.isfinite(strat_total) and strat_total < 0)
                try:
                    res = trade_target(info, target, st["group"], st["pair_arrays_map"],
                                       reverse=reverse, swap_tp_sl=REVERSE_SWAPS_TP_SL)
                except Exception as exc:
                    res = {"Status": "ENGINE_ERROR", "Note": f"trade: {type(exc).__name__}: {exc}"}
                pair = str(res.get("Pair", f["pair"]))
                colour, c_open, c_close = "n/a", np.nan, np.nan
                pa = st["pair_arrays_map"].get(pair)
                if pa is not None:
                    try:
                        colour, c_open, c_close = prev_candle_info(pa, target["UTC_Time"])
                    except Exception:
                        pass
                rec.update({
                    "Chosen_Solo_Mode": label,
                    "Strategy_Mode": info["label"],
                    "Pair": pair,
                    "Sub_Mode": res.get("Sub_Mode", "N/A"),
                    "Candle_Colour": colour, "Candle_Open": c_open, "Candle_Close": c_close,
                    "Candle_Body_Pct": res.get("Candle_Body_Pct", np.nan),
                    "Reversed": "Yes" if reverse else "No",
                    "Strategy_Direction": res.get("Strategy_Direction", np.nan),
                    "Traded_Direction": res.get("Direction", np.nan),
                    "Strategy_TP_Percent": res.get("Strategy_TP_Percent", np.nan),
                    "Strategy_SL_Percent": res.get("Strategy_SL_Percent", np.nan),
                    "Used_TP_Percent": res.get("TP_Percent", np.nan),
                    "Used_SL_Percent": res.get("SL_Percent", np.nan),
                    "Risk_Reward": res.get("Risk_Reward", np.nan),
                    "Status": res.get("Status", "ENGINE_ERROR"), "Note": res.get("Note", ""),
                    "PnL_Percent": res.get("PnL_Percent", np.nan),
                    "Is_Win": res.get("Is_Win", np.nan),
                    "Exit_Reason": res.get("Exit_Reason", np.nan),
                    "Train_Size": info["train_size"], "Forward_Size": info["fwd_size"],
                    "S1_Window_WinRate": get_num(w, "Actual_Win_Rate_Percent"),
                    "S1_Window_EV": get_num(w, "Actual_EV_Percent"),
                    "S1_Window_TotalPnL": get_num(w, "Actual_Total_PnL_Percent"),
                    "Strat_Train_WinRate": get_num(f, "win_rate_percent_TRAIN"),
                    "Strat_Fwd_WinRate": get_num(f, "win_rate_percent_FWD"),
                    "Strat_Train_PnL": get_num(f, "total_pnl_percent_TRAIN"),
                    "Strat_Fwd_PnL": get_num(f, "total_pnl_percent_FWD"),
                    "Strat_Total_PnL": strat_total,
                    "Strat_Avg_Delta": get_num(f, "avg_pnl_delta"),
                })
                if rec["Status"] == "TRADED":
                    cum_all += rec["PnL_Percent"]
                    cum_mode[label] += rec["PnL_Percent"]
                rec["Cumulative_PnL_Percent"] = cum_all
                rec["Cumulative_PnL_Mode"] = cum_mode[label]

            rows.append(rec)
            pd.DataFrame(rows, columns=PAPER_COLUMNS).to_csv(csv_path, index=False)

            traded = [r for r in rows if r["Status"] == "TRADED"]
            w_n = sum(int(r["Is_Win"]) for r in traded)
            who = rec["Chosen_Solo_Mode"] if choice != "skip" else "SKIP"
            print(f"\n>>> #{rec['Trade_No']} {rec['Target_Date']} [{who}] "
                  f"{describe_trade(rec) if choice != 'skip' else rec['Note']}")
            print(f"    paper total {cum_all:+.3f}% | trades {len(traded)} "
                  f"(W{w_n} L{len(traded) - w_n}) | logged -> {csv_path.name}")
    except (KeyboardInterrupt, EOFError):
        print("\n\nStopped - saving the paper-trade log...")
    return rows, csv_path


def build_paper_summary(df):
    s = build_summary(df, 0, 0, 0, False)
    for k in ("News rows", "Window size", "Steps planned", "Solo-mode rows"):
        s.pop(k, None)
    s = {"Events decided": len(df), **s}
    s.pop("Not traded - SKIPPED_BY_USER", None)
    s["Skipped by user"] = int((df["Status"] == "SKIPPED_BY_USER").sum())
    return s


def print_paper_report(df):
    print("\n" + "=" * 72)
    print("PAPER-TRADING RESULT")
    print("=" * 72)
    summ = build_paper_summary(df)
    for k, v in summ.items():
        if isinstance(v, float):
            v = "inf" if np.isinf(v) else f"{v:.4f}"
        print(f"  {k:<34}{v}")
    for col, title in (("Chosen_Solo_Mode", "By chosen solo mode"), ("Reversed", "Reversed vs as-is"),
                       ("Pair", "By pair")):
        tbl = group_table(df, col)
        if not tbl.empty:
            print(f"\n--- {title} ---")
            print(tbl.round(3).to_string(index=False))
    show_cols = ["Trade_No", "Target_Date", "Chosen_Solo_Mode", "Pair", "Reversed", "Traded_Direction",
                 "Used_TP_Percent", "Used_SL_Percent", "Exit_Reason", "PnL_Percent",
                 "Cumulative_PnL_Percent", "Status"]
    print("\n--- Final trade log ---")
    print(df[show_cols].round(3).to_string(index=False))


def save_paper_outputs(df, stamp, csv_path):
    df.to_csv(csv_path, index=False)
    xlsx_path = bw.OUTPUT_DIR / f"PAPER_{stamp}_paper_trades.xlsx"
    summ = build_paper_summary(df)
    summary_df = pd.DataFrame(
        [(k, ("inf" if isinstance(v, float) and np.isinf(v) else v)) for k, v in summ.items()],
        columns=["Metric", "Value"])
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Trades", index=False)
        for col, name in (("Chosen_Solo_Mode", "By_Solo_Mode"), ("Reversed", "By_Reversed"),
                          ("Pair", "By_Pair")):
            tbl = group_table(df, col)
            if not tbl.empty:
                tbl.to_excel(writer, sheet_name=name, index=False)
        for name in writer.sheets:
            writer.sheets[name].freeze_panes = "A2"
    return xlsx_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    group = bw.choose_currency_group()
    if not group:
        return
    entry_time = bw.choose_entry_time()
    settings = bw.choose_grid_settings()
    keep_pct = cp.choose_cut_percent()

    parsed = cp.t3.paste_news(group, entry_time)
    if parsed is None:
        return
    news_df, has_surprise = parsed
    news_df = news_df.sort_values("UTC_Time").reset_index(drop=True)
    n_rows = len(news_df)

    window = ask_window(n_rows)
    if n_rows <= window:
        print(f"\nOnly {n_rows} news rows but the window is {window}: need at least {window + 1} rows "
              f"(window + 1 target). Nothing to do.")
        input("\nPress Enter to exit...")
        return
    steps_total = n_rows - window

    pair_dfs, pair_arrays_map = cp.load_pairs(group)
    if not pair_arrays_map:
        print("No valid pair data found.")
        input("\nPress Enter to exit...")
        return

    min_train, max_train, min_fwd, max_fwd, _, min_wr = settings
    print("\n" + "=" * 72)
    print(f"WALK-FORWARD SOLO | group={group} | {n_rows} news rows | window={window} | "
          f"{steps_total} steps")
    print("solo modes tracked independently: "
          + ", ".join(m["label"] for m in SOLO_MODES))
    print(f"engine grid: train {min_train}-{max_train}, fwd {min_fwd}-{max_fwd} | "
          f"cut {keep_pct:g}% | min win rate {min_wr:g}% | pairs: {', '.join(pair_arrays_map)}")
    print("Ctrl+C stops early and still saves/reports what is done.")
    print("=" * 72)

    bw.OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = bw.OUTPUT_DIR / f"{group}_{stamp}_walk_forward_solo_feeless_log.csv"

    rows = []
    cum_by_mode = {m["label"]: 0.0 for m in SOLO_MODES}
    wins_by_mode = {m["label"]: 0 for m in SOLO_MODES}
    losses_by_mode = {m["label"]: 0 for m in SOLO_MODES}
    consecutive_errors = 0
    interrupted = False
    started = time.time()
    shown_traceback = False
    label_w = max(len(m["label"]) for m in SOLO_MODES)

    try:
        for step in range(steps_total):
            win_news = news_df.iloc[step:step + window].reset_index(drop=True)
            target = news_df.iloc[step + window]

            for solo in SOLO_MODES:
                t0 = time.time()
                label = solo["label"]
                rec = {c: np.nan for c in LOG_COLUMNS}
                rec.update({
                    "Step": step + 1,
                    "Solo_Mode": label,
                    "Target_Date": target["History"].strftime("%Y-%m-%d"),
                    "Target_UTC": target["UTC_Time"],
                    "Window_From": win_news["History"].iloc[0].strftime("%Y-%m-%d"),
                    "Window_To": win_news["History"].iloc[-1].strftime("%Y-%m-%d"),
                })

                try:
                    info, why = run_engine(win_news, group, has_surprise, pair_dfs,
                                           pair_arrays_map, settings, keep_pct, solo)
                    consecutive_errors = 0
                except Exception as exc:  # engine crashed on this window
                    consecutive_errors += 1
                    info, why = None, None
                    rec.update({"Status": "ENGINE_ERROR", "Note": f"{type(exc).__name__}: {exc}"})
                    if not shown_traceback:
                        shown_traceback = True
                        print("\n[first engine error - full traceback]")
                        traceback.print_exc()

                if info is None and rec["Status"] != "ENGINE_ERROR":
                    rec.update({"Status": "NO_STRATEGY", "Note": why})
                elif info is not None:
                    w, f = info["winner"], info["final"]
                    rec.update({
                        "Mode": info["label"], "Inverted": info["inverted"],
                        "Train_Size": info["train_size"], "Forward_Size": info["fwd_size"],
                        "S1_Window_WinRate": get_num(w, "Actual_Win_Rate_Percent"),
                        "S1_Window_EV": get_num(w, "Actual_EV_Percent"),
                        "S1_Window_TotalPnL": get_num(w, "Actual_Total_PnL_Percent"),
                        "Strat_Train_WinRate": get_num(f, "win_rate_percent_TRAIN"),
                        "Strat_Fwd_WinRate": get_num(f, "win_rate_percent_FWD"),
                        "Strat_Train_PnL": get_num(f, "total_pnl_percent_TRAIN"),
                        "Strat_Fwd_PnL": get_num(f, "total_pnl_percent_FWD"),
                        "Strat_Total_PnL": get_num(f, "Total_PnL_Percent"),
                        "Strat_Avg_Delta": get_num(f, "avg_pnl_delta"),
                    })
                    try:
                        rec.update(trade_target(info, target, group, pair_arrays_map))
                    except Exception as exc:
                        rec.update({"Status": "ENGINE_ERROR", "Note": f"trade: {type(exc).__name__}: {exc}"})

                if rec["Status"] == "TRADED":
                    cum_by_mode[label] += rec["PnL_Percent"]
                    wins_by_mode[label] += rec["Is_Win"]
                    losses_by_mode[label] += 1 - rec["Is_Win"]
                rec["Cumulative_PnL_Percent"] = cum_by_mode[label]
                rec["Engine_Seconds"] = round(time.time() - t0, 1)
                rows.append(rec)

                # ---- progress line ----
                elapsed = time.time() - started
                head = f"[{step + 1:>{len(str(steps_total))}}/{steps_total}] {rec['Target_Date']} {label:<{label_w}}"
                if rec["Status"] == "TRADED":
                    body = (f"{rec['Pair']} {rec['Direction']} TP{rec['TP_Percent']:.2f}% SL{rec['SL_Percent']:.2f}% "
                            f"-> {rec['Exit_Reason']} {rec['PnL_Percent']:+.3f}%")
                else:
                    body = f"{rec['Status']}: {rec['Note']}"
                print(f"{head} | {body} | total {cum_by_mode[label]:+.2f}% | "
                      f"W{wins_by_mode[label]} L{losses_by_mode[label]} | {rec['Engine_Seconds']:.0f}s")

            pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(csv_path, index=False)

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n{MAX_CONSECUTIVE_ERRORS} engine errors in a row - stopping (likely a setup problem; "
                      "see the traceback above).")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nStopped by Ctrl+C - saving what is done...")

    if not rows:
        print("No steps completed.")
        input("\nPress Enter to exit...")
        return

    log_df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    
    # ---- Generate next trade signals from the last window ----
    last_window_news = news_df.iloc[-window:].reset_index(drop=True)
    next_signals = {}
    print("\n" + "=" * 72)
    print("COMPUTING NEXT TRADE SIGNALS (last window)...")
    print("=" * 72)
    for solo in SOLO_MODES:
        label = solo["label"]
        info, reason = get_next_trade_signal(last_window_news, group, has_surprise, 
                                              pair_dfs, pair_arrays_map, settings, 
                                              keep_pct, solo)
        next_signals[label] = (info, reason)
        if info is None:
            print(f"  {label}: NO STRATEGY - {reason}")
        else:
            print(format_next_signal(info, label))

    summary = build_summary(log_df, window, n_rows, steps_total, interrupted)
    by_mode_summ = build_summary_by_mode(log_df, window, n_rows, steps_total, interrupted)
    print_report(summary, by_mode_summ, log_df, next_signals)
    xlsx_path = save_outputs(log_df, summary, by_mode_summ, group, stamp, csv_path, next_signals)
    print(f"\nTotal run time : {fmt_secs(time.time() - started)}")
    print(f"Excel saved    : {xlsx_path}")
    print(f"CSV log saved  : {csv_path}")

    # ---- paper-trading mode: pick strategy + date, trade it, log it, repeat ----
    try:
        ans = input("\nStart PAPER-TRADING mode (pick a strategy + event date, trade it, log it)? "
                    "Enter = yes, n = exit: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if not ans.startswith("n"):
        st = {"group": group, "entry_time": entry_time, "news_df": news_df,
              "has_surprise": has_surprise, "pair_dfs": pair_dfs,
              "pair_arrays_map": pair_arrays_map}
        cache = {window_key(group, entry_time, last_window_news): next_signals}
        paper_rows, paper_csv = paper_trade_session(st, window, settings, keep_pct, stamp, cache)
        if paper_rows:
            paper_df = pd.DataFrame(paper_rows, columns=PAPER_COLUMNS)
            print_paper_report(paper_df)
            paper_xlsx = save_paper_outputs(paper_df, stamp, paper_csv)
            print(f"\nPaper trades Excel : {paper_xlsx}")
            print(f"Paper trades CSV   : {paper_csv}")
        else:
            print("No paper trades were logged.")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
