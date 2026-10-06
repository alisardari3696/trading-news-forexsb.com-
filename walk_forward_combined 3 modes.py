"""
WALK-FORWARD RUNNER  (uses the exact logic of combined_backtest.py)
===================================================================
Put this file in the SAME folder as combined_backtest.py and the <PAIR>60.csv files
(or tell it the CSV folder when asked).

Run modes (you choose at start):  1 Combined | 2 Surprise only | 3 Candle colour only
(In mode 2 a neutral target, actual == forecast, is simply not traded.)

What it does
------------
You give it N news rows (e.g. 100) and a window size (default 55).

    target = row 56  ->  input = rows 1..55   -> run combined_backtest logic -> strategy
    target = row 57  ->  input = rows 2..56   -> run again                    -> strategy
    ...
    target = row N   ->  input = rows N-55..N-1

For every target event:
    1. SURPRISE mode  : Phase 1 (best window) + Phase 2 (final strategy) on the 55 input rows.
                        If a strategy is found -> trade it on the target (pair, TP/SL in ATR,
                        direction from the target's actual vs forecast).
    2. If there is NO strategy -> CANDLE COLOUR mode (trend / fade) on the same 55 rows.
                        If a strategy is found -> trade it on the target (direction from the colour
                        of the 1h candle just before the entry time, trend or fade).
    3. If there is no strategy there either -> NO TRADE.
    * If the target is neutral (actual == forecast) surprise mode is skipped and the event
      goes straight to CANDLE COLOUR mode (the event is never skipped just for being neutral).

Works for every group in combined_backtest.py: USD, EUR, GBP, JPY, AUD, NZD, CHF, CAD.
Pairs whose CSV is missing are skipped automatically.

News file format: tab separated, header row, columns  History  Actual  Forecast  (Previous optional)
e.g. copied from Investing/TradingView. Newest-first or oldest-first both work.
If the file has no Actual/Forecast columns only candle colour mode is used.

Output (folder ./results): one Excel file (Summary, Log, By_Pair, By_Mode, By_Year) + a CSV of the log.
"""
import sys
import io
import re
import time
import contextlib
import traceback
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined_backtest as cb  # noqa: E402

# ----------------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------------
def ask(prompt, default):
    val = input(f"{prompt} (default {default}): ").strip()
    return val if val else str(default)


def read_news_text():
    path = input("\nPath to news .txt/.tsv file (press Enter to paste instead): ").strip().strip('"')
    if path:
        return Path(path).read_text(encoding="utf-8-sig")
    print("Paste tab-separated news data (with header). Finish with an empty line:\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def parse_news(raw, group, entry_time_str, inverted):
    """Same parsing / timezone / surprise logic as cb.paste_news_data, but on a string.
    Returns (df, has_surprise_cols)."""
    df = pd.read_csv(StringIO(raw), sep="\t")
    if "History" not in df.columns:
        raise ValueError("Missing 'History' column.")
    hour, minute = map(int, entry_time_str.split(":"))

    def parse_date(s):
        s = str(s).strip()
        s = re.sub(r"(\w+\s+\d+)-(\d+),\s*(\d{4})", r"\1, \3", s)
        return pd.to_datetime(s, format="%b %d, %Y", errors="coerce")

    df["History"] = df["History"].astype(str).apply(parse_date)

    has_surprise = "Actual" in df.columns and "Forecast" in df.columns
    if has_surprise:
        def parse_value(v):
            v = str(v).replace("%", "").replace(",", "").strip().lower()
            return v[:-1] if v.endswith(("k", "m")) else v
        for c in ["Actual", "Forecast"]:
            df[c] = pd.to_numeric(df[c].astype(str).apply(parse_value), errors="coerce")
        df = df.dropna(subset=["History", "Actual", "Forecast"]).copy()
    else:
        df = df.dropna(subset=["History"]).copy()

    local_tz = ZoneInfo(cb.ASSET_TIMEZONES[group])
    df["UTC_Time"] = [
        pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hour, minute=minute, tz=local_tz)
        .tz_convert(ZoneInfo("UTC")).tz_localize(None)
        for d in df["History"]
    ]
    if has_surprise:
        adj = (df["Actual"] - df["Forecast"]) * (-1 if inverted else 1)
        df["Surprise_Type"] = adj.apply(lambda v: "positive" if v > 0 else "negative" if v < 0 else "neutral")
    else:
        df["Surprise_Type"] = "neutral"
    return df.sort_values("UTC_Time").reset_index(drop=True), has_surprise


def pip_size(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


# ----------------------------------------------------------------------------
# One mode: Phase 1 + Phase 2 on a block of news rows -> strategy row or None
# ----------------------------------------------------------------------------
def find_strategy(sub, mode, ctx):
    """Returns (strategy_row_or_None, info_dict). Mirrors main() of combined_backtest.py."""
    info = {}
    pam, pdfs, group = ctx["pair_arrays_map"], ctx["pair_dfs"], ctx["group"]

    # ---- same filters as main() ----
    sub = sub[[any(cb.has_candle(pam[p], t) for p in pam) for t in sub["UTC_Time"]]].reset_index(drop=True)
    if mode == "surprise":
        sub = sub[sub["Surprise_Type"] != "neutral"].reset_index(drop=True)
    else:
        ok = [any(cb.has_candle(pam[p], t) and cb.has_candle(pam[p], t - pd.Timedelta(hours=1)) for p in pam)
              for t in sub["UTC_Time"]]
        sub = sub[ok].reset_index(drop=True)
    info["usable_events"] = len(sub)
    if len(sub) < 3:
        info["status"] = "NOT ENOUGH EVENTS"
        return None, info

    # ---- PHASE 1 ----
    with contextlib.redirect_stdout(io.StringIO()):
        res = cb.run_window_grid_search(
            sub, pam, pdfs, mode, group,
            ctx["min_train"], ctx["max_train"], ctx["min_fwd"], ctx["max_fwd"],
            ctx["top_pct"], ctx["min_wr"])
    grid_results = res[0]
    if not grid_results:
        info["status"] = "NO GRID RESULTS"
        return None, info
    grid_df = pd.DataFrame(grid_results)
    q = grid_df[(grid_df["Actual_Win_Rate_Percent"] >= ctx["min_wr"])
                & (grid_df["Actual_EV_Percent"] > cb.EV_MIN_PERCENT)]
    info["qualified_windows"] = len(q)
    if q.empty:
        info["status"] = "NO WINDOW"
        return None, info
    best = q.sort_values("Actual_Total_PnL_Percent", ascending=False).iloc[0]
    tr, fw = int(best["Train_Size"]), int(best["Forward_Size"])
    info.update(train_size=tr, forward_size=fw,
                window_wr=best["Actual_Win_Rate_Percent"], window_ev=best["Actual_EV_Percent"],
                window_pnl=best["Actual_Total_PnL_Percent"])

    # ---- PHASE 2 ----
    with contextlib.redirect_stdout(io.StringIO()):
        analysis = cb.run_analysis(sub, pdfs, pam, mode, group, tr, fw, ctx["keep_pct"], ctx["min_wr"])
    if analysis is None:
        info["status"] = "ANALYSIS FAILED"
        return None, info
    _, row = cb.select_final_strategy(analysis["merged_cut"], ctx["min_wr"])
    if row is None:
        info["status"] = "NO STRATEGY"
        return None, info
    info["status"] = "STRATEGY FOUND"
    return row, info


# ----------------------------------------------------------------------------
# Trade the strategy on the target event
# ----------------------------------------------------------------------------
def trade_target(row, mode, target, ctx):
    pair, tp, sl = row["pair"], float(row["tp_atr"]), float(row["sl_atr"])
    pa = ctx["pair_arrays_map"][pair]
    t = target["UTC_Time"]
    if mode == "surprise":
        direction = cb.get_direction("surprise", pair, target["Surprise_Type"], ctx["group"])
        sub_mode = None
    else:
        sub_mode = row["sub_mode"]
        pos = cb.index_position(pa["index"], t - pd.Timedelta(hours=1))
        if pos < 0:
            return None, "no candle before entry"
        candle = {"Open": float(pa["open"][pos]), "Close": float(pa["close"][pos])}
        direction = cb.get_direction("candle_colour", pair, None, ctx["group"], candle=candle, sub_mode=sub_mode)
    if direction is None:
        return None, "no direction (neutral/doji)"
    window = cb.build_trade_window(pa, t)
    if window is None:
        return None, "no price data at target"
    r = cb.simulate_trade(window, direction, tp, sl)
    sign = 1 if direction == "long" else -1
    return {
        "Pair": pair, "Sub_Mode": sub_mode or "", "TP_ATR": tp, "SL_ATR": sl, "Direction": direction,
        "WR_Train": row["win_rate_percent_TRAIN"], "WR_Fwd": row["win_rate_percent_FWD"],
        "AvgPnL_Train": row["average_pnl_percent_TRAIN"], "AvgPnL_Fwd": row["average_pnl_percent_FWD"],
        "Entry_Price": r["entry_price"], "Exit_Price": r["exit_price"], "Exit_Reason": r["exit_reason"],
        "Hold_Hours": r["hold_hours"], "PnL_Percent": r["pnl_pct"] * 100,
        "PnL_Pips": sign * (r["exit_price"] - r["entry_price"]) / pip_size(pair),
        "Result": "WIN" if r["pnl_pct"] > 0 else "LOSS",
    }, ""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("WALK-FORWARD (surprise -> candle colour fallback -> no trade)")
    print("=" * 70)
    print("Groups:", ", ".join(cb.pair_sets))
    group = input("Currency group (example: USD): ").upper().strip()
    if group not in cb.pair_sets:
        print("Invalid currency group.")
        return
    entry_time = input("Trade entry time in local event timezone HH:MM: ").strip()
    inverted = input("Is higher Actual worse for the currency? (y/n): ").lower().strip() == "y"
    print("\nSelect run mode:")
    print("1. Combined (surprise first, candle colour if no surprise strategy)")
    print("2. Surprise only")
    print("3. Candle colour only")
    run_mode = input("Choice (1/2/3) (default 1): ").strip() or "1"
    if run_mode not in ("1", "2", "3"):
        print("Invalid run mode.")
        return
    win_rows = int(ask("News rows per window", 55))
    data_dir = ask("Folder with the <PAIR>60.csv files", ".")

    use_defaults = input("Use default grid settings (train 1-35, fwd 1-35, cutoff 50, min WR 70, keep 50)? (Y/n): ")
    if use_defaults.strip().lower() in ("n", "no"):
        min_train, max_train, min_fwd, max_fwd, top_pct, min_wr = cb.choose_grid_settings()
        keep_pct = cb.choose_data_cutoff()
    else:
        min_train, max_train, min_fwd, max_fwd, top_pct, min_wr, keep_pct = 1, 35, 1, 35, 0.50, 70.0, 50.0

    raw = read_news_text()
    news, has_surprise = parse_news(raw, group, entry_time, inverted)
    print(f"\nNews rows parsed: {len(news)}  ({news['History'].min().date()} -> {news['History'].max().date()})")
    if len(news) <= win_rows:
        print(f"Need more than {win_rows} rows to have at least one target event.")
        return
    if not has_surprise:
        if run_mode in ("1", "2"):
            if run_mode == "2":
                print("Surprise mode needs Actual and Forecast columns.")
                return
            print("No Actual/Forecast columns -> candle colour mode only.")
        run_mode = "3"

    cb.DATA_DIR = Path(data_dir)
    pair_dfs, pair_arrays_map = {}, {}
    for p in cb.pair_sets[group]:
        d = cb.load_csv(p)
        if d is not None:
            pair_dfs[p] = d
            pair_arrays_map[p] = cb.build_pair_arrays(p, d)
    if not pair_dfs:
        print("No pair CSV data found.")
        return
    print("Pairs loaded:", ", ".join(pair_dfs))

    ctx = dict(group=group, pair_dfs=pair_dfs, pair_arrays_map=pair_arrays_map,
               min_train=min_train, max_train=max_train, min_fwd=min_fwd, max_fwd=max_fwd,
               top_pct=top_pct, min_wr=min_wr, keep_pct=keep_pct)

    out_dir = cb.OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{group}_{entry_time.replace(':', '')}_{stamp}_walk_forward_log.csv"
    xlsx_path = out_dir / f"{group}_{entry_time.replace(':', '')}_{stamp}_walk_forward.xlsx"

    targets = list(range(win_rows, len(news)))
    print(f"\n{len(targets)} target events. Running ...\n")
    log, t0 = [], time.time()

    for k, t in enumerate(targets, 1):
        target = news.iloc[t]
        sub = news.iloc[t - win_rows:t].reset_index(drop=True)
        rec = {"Target_Date": target["History"].strftime("%Y-%m-%d"), "Entry_UTC": target["UTC_Time"]}
        if has_surprise:
            rec.update(Actual=target["Actual"], Forecast=target["Forecast"], Surprise=target["Surprise_Type"])
        trade, mode_used, note = None, "none", ""

        try:
            modes = {"1": ["surprise", "candle_colour"], "2": ["surprise"], "3": ["candle_colour"]}[run_mode]
            for mode in modes:
                if mode == "surprise" and target["Surprise_Type"] == "neutral":
                    # actual == forecast: no surprise direction -> candle colour only
                    rec["S_Status"] = "SKIPPED (neutral target)"
                    continue
                row, info = find_strategy(sub, mode, ctx)
                tag = "S" if mode == "surprise" else "C"
                rec[f"{tag}_Status"] = info.get("status")
                rec[f"{tag}_Train"] = info.get("train_size")
                rec[f"{tag}_Fwd"] = info.get("forward_size")
                rec[f"{tag}_Window_WR"] = info.get("window_wr")
                rec[f"{tag}_Window_EV"] = info.get("window_ev")
                if row is None:
                    continue
                trade, note = trade_target(row, mode, target, ctx)
                if trade is not None:
                    mode_used = mode
                    break
                break  # strategy found but target not tradable with it (e.g. doji candle)
        except Exception as e:  # keep the run alive, log the problem
            note = f"ERROR: {e}"
            traceback.print_exc()

        rec["Mode_Used"] = mode_used
        rec["Status"] = "TRADED" if trade else ("NO TRADE" if not note else f"SKIPPED ({note})")
        if trade:
            rec.update(trade)
        log.append(rec)
        pd.DataFrame(log).to_csv(csv_path, index=False)  # saved after every event
        print(f"[{k}/{len(targets)}] {rec['Target_Date']}  {rec['Status']:<10} {mode_used:<13} "
              f"{rec.get('Pair', ''):<7} {rec.get('PnL_Percent', float('nan')):+.3f}%  "
              f"({time.time() - t0:.0f}s)" if trade else
              f"[{k}/{len(targets)}] {rec['Target_Date']}  {rec['Status']}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- report ----------------
    df = pd.DataFrame(log)
    tr = df[df["Status"] == "TRADED"].copy()
    summary = {"Run mode": {"1": "Combined", "2": "Surprise only", "3": "Candle colour only"}[run_mode], "Group": group, "Entry time (local)": entry_time, "Window rows": win_rows,
               "Target events": len(df), "Traded": len(tr),
               "  via surprise": int((tr["Mode_Used"] == "surprise").sum()) if len(tr) else 0,
               "  via candle colour": int((tr["Mode_Used"] == "candle_colour").sum()) if len(tr) else 0,
               "No trade": int((df["Status"] != "TRADED").sum())}
    if len(tr):
        p = tr["PnL_Percent"].to_numpy()
        eq = np.concatenate([[0.0], np.cumsum(p)])
        gl = -p[p < 0].sum()
        summary.update({
            "Win rate %": (p > 0).mean() * 100, "Total PnL %": p.sum(), "Avg PnL % / trade": p.mean(),
            "Profit factor": p[p > 0].sum() / gl if gl else np.nan,
            "Max drawdown %": (np.maximum.accumulate(eq) - eq).max(),
            "Avg win %": p[p > 0].mean() if (p > 0).any() else np.nan,
            "Avg loss %": p[p < 0].mean() if (p < 0).any() else np.nan,
            "Total pips": tr["PnL_Pips"].sum()})
    summ_df = pd.DataFrame({"Metric": list(summary), "Value": list(summary.values())})

    def grp(col):
        if tr.empty:
            return pd.DataFrame()
        return tr.groupby(col).agg(Trades=("PnL_Percent", "size"),
                                   Win_Rate=("Result", lambda x: (x == "WIN").mean() * 100),
                                   Total_PnL_Percent=("PnL_Percent", "sum"),
                                   Total_Pips=("PnL_Pips", "sum")).reset_index()

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summ_df.to_excel(w, sheet_name="Summary", index=False)
        df.to_excel(w, sheet_name="Log", index=False)
        grp("Pair").to_excel(w, sheet_name="By_Pair", index=False)
        grp("Mode_Used").to_excel(w, sheet_name="By_Mode", index=False)
        if not tr.empty:
            tr.assign(Year=tr["Target_Date"].str[:4]).pipe(
                lambda d: d.groupby("Year").agg(Trades=("PnL_Percent", "size"),
                                                Win_Rate=("Result", lambda x: (x == "WIN").mean() * 100),
                                                Total_PnL_Percent=("PnL_Percent", "sum"))
            ).reset_index().to_excel(w, sheet_name="By_Year", index=False)
        for ws in w.sheets.values():
            ws.freeze_panes = "A2"

    print("\n" + "=" * 70)
    print(summ_df.to_string(index=False))
    print("=" * 70)
    print(f"Saved: {xlsx_path}")
    print(f"Saved: {csv_path}")
    try:
        input("\nPress Enter to close ...")
    except EOFError:
        pass


if __name__ == "__main__":
    main()
