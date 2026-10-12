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

Needs in the SAME folder:
    combined_pipeline_feeless.py   (the engine)
    backtesting_window_grid.py
    backtesting_three_modes_feeless.py
    Analyzing_v20_feeless.py
    <PAIR>60.csv price files
"""
import contextlib
import io
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
def trade_target(info, target, group, pair_arrays_map):
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

    out["Candle_Body_Pct"] = candle_body_pct(pa, t, direction)

    # ---- simulate with the same simulator the engine used ----
    window = bw.build_trade_window(pa, t)
    if window is None:
        return {**out, "Status": "SKIP_NO_PRICE", "Note": "no price data at target entry time"}

    tp_frac = to_fraction(final["tp_percent"], bw.TP_VALUES)
    sl_frac = to_fraction(final["sl_percent"], bw.SL_VALUES)
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
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
