"""
WALK-FORWARD RUNNER  (engine = combined_pipeline_feeless.py, feeless)
=====================================================================
You paste N news rows and choose a window size (default 55).

    rows  1..55  -> run the engine -> strategy -> trade it on row 56 -> save result
    rows  2..56  -> run the engine -> strategy -> trade it on row 57 -> save result
    rows  3..57  -> run the engine -> strategy -> trade it on row 58 -> ...
    ...
    until the last row has been traded.

"Engine" = the exact pipeline in combined_pipeline_feeless.py:
    Stage 1: three modes (Candle / Surprise_Better / Surprise_Worse) window grid search
             -> top 5 windows per mode by total PnL -> highest EV wins (mode + train/fwd sizes)
    Stage 2: Analyzing on the newest train+fwd events of the window
             -> cut by lowest |Avg PnL Delta| (default 50%) -> highest total PnL -> final strategy

The strategy (pair, TP, SL, and for Candle mode trend/fade) is then applied to the NEXT
event, which the engine never saw:
    Surprise modes : direction from the target's Actual vs Forecast (inverted for Surprise_Worse)
    Candle mode    : direction from the colour of the 1h candle just before the target's entry time

Row order: rows are sorted oldest -> newest by event time, so "row 1" is the OLDEST news
no matter in which order you pasted them.

When a step has no trade the reason is logged and the walk-forward just rolls on:
    NO_STRATEGY   engine could not produce a strategy for that window
    SKIP_NEUTRAL  surprise mode won but the target has Actual == Forecast (or no values)
    SKIP_DOJI     candle mode won but the previous 1h candle has Open == Close
    SKIP_NO_PRICE no price data for the strategy pair at the target time
    ENGINE_ERROR  the engine crashed on that window (message is logged)

The terminal stays open: progress is printed after every step, the final report is printed,
and you press Enter to exit. Ctrl+C stops early and still saves + reports what is done.

Output (./results): one Excel file (Summary, Log, By_Mode, By_Pair) + a CSV of the log
(the CSV is rewritten after every step, so nothing is lost if the terminal is closed).

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

LOG_COLUMNS = [
    "Step", "Target_Date", "Target_UTC", "Window_From", "Window_To",
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


def choose_mode_set():
    """Pick which Stage 1 modes to run in the walk-forward engine.
    Returns (name, [mode tuples])."""
    print("\n--- Walk-forward mode set ---")
    print("  1) Three modes   : Candle + Surprise_Better + Surprise_Worse (all together)")
    print("  2) Candle only   : candle colour mode only")
    print("  3) Surprise only : Surprise_Better + Surprise_Worse (highest EV wins)")
    raw = input("Choose mode set (default 1): ").strip()
    if raw == "2":
        return "candle", [m for m in cp.t3.MODES if m[0] == "candle_colour"]
    if raw == "3":
        return "surprise", [m for m in cp.t3.MODES if m[0] == "surprise"]
    return "three", list(cp.t3.MODES)


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
# Engine call: rows of ONE window -> strategy (or None + reason)
# ----------------------------------------------------------------------------
def run_engine(window_news, group, has_surprise, pair_dfs, pair_arrays_map, settings, keep_pct, modes):
    with quiet():
        results = cp.run_stage1(window_news, has_surprise, group, pair_arrays_map, settings, modes=modes)
        if not results:
            return None, "Stage 1 produced no window results"

        top = cp.pick_top_candidates(results)
        winner = top.iloc[0]
        mode = winner["Mode"]
        inverted = bool(winner["Inverted"])
        train_size = int(winner["Train_Size"])
        fwd_size = int(winner["Forward_Size"])

        # Extract sub_mode from winner for candle_colour mode
        sub_modes = None
        if mode == "candle_colour" and "Sub_Mode" in winner:
            sub_modes = [winner["Sub_Mode"]]

        news2 = cp.build_stage2_events(window_news, mode, inverted)
        s2 = cp.run_stage2(news2, mode, group, train_size, fwd_size, keep_pct,
                           pair_dfs, pair_arrays_map, sub_modes=sub_modes)
    if s2 is None:
        return None, "Stage 2 produced no results"

    return {
        "label": winner["Mode_Label"],
        "mode": mode,
        "inverted": inverted,
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
        "Steps completed": len(log_df),
        "Stopped early (Ctrl+C)": "YES" if interrupted else "no",
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


def print_report(summary, log_df):
    print("\n" + "=" * 72)
    print("WALK-FORWARD RESULT")
    print("=" * 72)
    for k, v in summary.items():
        if isinstance(v, float):
            v = "inf" if np.isinf(v) else f"{v:.4f}"
        print(f"  {k:<34}{v}")
    by_mode = group_table(log_df, "Mode")
    if not by_mode.empty:
        print("\n--- By mode ---")
        print(by_mode.round(3).to_string(index=False))
    by_pair = group_table(log_df, "Pair")
    if not by_pair.empty:
        print("\n--- By pair ---")
        print(by_pair.round(3).to_string(index=False))


def save_outputs(log_df, summary, group, stamp, csv_path, mode_name):
    log_df.to_csv(csv_path, index=False)
    xlsx_path = bw.OUTPUT_DIR / f"{group}_{stamp}_walk_forward_{mode_name}_feeless.xlsx"
    summary_df = pd.DataFrame(
        [(k, ("inf" if isinstance(v, float) and np.isinf(v) else v)) for k, v in summary.items()],
        columns=["Metric", "Value"])
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        log_df.to_excel(writer, sheet_name="Log", index=False)
        group_table(log_df, "Mode").to_excel(writer, sheet_name="By_Mode", index=False)
        group_table(log_df, "Pair").to_excel(writer, sheet_name="By_Pair", index=False)
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
    mode_name, mode_list = choose_mode_set()

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
    mode_labels = ", ".join(m[2] for m in mode_list)
    print("\n" + "=" * 72)
    print(f"WALK-FORWARD | group={group} | {n_rows} news rows | window={window} | "
          f"{steps_total} steps")
    print(f"engine mode set: {mode_name} ({mode_labels})")
    print(f"engine grid: train {min_train}-{max_train}, fwd {min_fwd}-{max_fwd} | "
          f"cut {keep_pct:g}% | min win rate {min_wr:g}% | pairs: {', '.join(pair_arrays_map)}")
    print("Ctrl+C stops early and still saves/reports what is done.")
    print("=" * 72)

    bw.OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = bw.OUTPUT_DIR / f"{group}_{stamp}_walk_forward_feeless_log.csv"

    rows = []
    cum = 0.0
    wins = losses = 0
    consecutive_errors = 0
    interrupted = False
    started = time.time()
    shown_traceback = False

    try:
        for step in range(steps_total):
            t0 = time.time()
            win_news = news_df.iloc[step:step + window].reset_index(drop=True)
            target = news_df.iloc[step + window]

            rec = {c: np.nan for c in LOG_COLUMNS}
            rec.update({
                "Step": step + 1,
                "Target_Date": target["History"].strftime("%Y-%m-%d"),
                "Target_UTC": target["UTC_Time"],
                "Window_From": win_news["History"].iloc[0].strftime("%Y-%m-%d"),
                "Window_To": win_news["History"].iloc[-1].strftime("%Y-%m-%d"),
            })

            try:
                info, why = run_engine(win_news, group, has_surprise, pair_dfs,
                                       pair_arrays_map, settings, keep_pct, mode_list)
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
                cum += rec["PnL_Percent"]
                wins += rec["Is_Win"]
                losses += 1 - rec["Is_Win"]
            rec["Cumulative_PnL_Percent"] = cum
            rec["Engine_Seconds"] = round(time.time() - t0, 1)
            rows.append(rec)

            # ---- progress line ----
            elapsed = time.time() - started
            eta = elapsed / (step + 1) * (steps_total - step - 1)
            head = f"[{step + 1:>{len(str(steps_total))}}/{steps_total}] {rec['Target_Date']}"
            if rec["Status"] == "TRADED":
                body = (f"{rec['Mode']} {int(rec['Train_Size'])}/{int(rec['Forward_Size'])} | "
                        f"{rec['Pair']} {rec['Direction']} TP{rec['TP_Percent']:.2f}% SL{rec['SL_Percent']:.2f}% "
                        f"-> {rec['Exit_Reason']} {rec['PnL_Percent']:+.3f}%")
            else:
                body = f"{rec['Status']}: {rec['Note']}"
            print(f"{head} | {body} | total {cum:+.2f}% | W{wins} L{losses} | "
                  f"{rec['Engine_Seconds']:.0f}s ETA {fmt_secs(eta)}")

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
    summary = build_summary(log_df, window, n_rows, steps_total, interrupted)
    print_report(summary, log_df)
    xlsx_path = save_outputs(log_df, summary, group, stamp, csv_path, mode_name)
    print(f"\nTotal run time : {fmt_secs(time.time() - started)}")
    print(f"Excel saved    : {xlsx_path}")
    print(f"CSV log saved  : {csv_path}")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
