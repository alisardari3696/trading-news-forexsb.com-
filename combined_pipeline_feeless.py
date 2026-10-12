"""
COMBINED PIPELINE (feeless) - joins backtesting_three_modes_feeless.py and
Analyzing_v20_feeless.py into ONE run. Neither original file is modified.

STAGE 1  (logic from backtesting_three_modes_feeless.py)
  Runs the three modes from one news paste on one currency group:
      Candle / Surprise_Better / Surprise_Worse
  with the window grid search (train/fwd bounds from the usual prompts).
  From EACH mode keeps the top 5 windows by Actual_Total_PnL_Percent,
  then out of those candidates picks the one with the highest
  Actual_EV_Percent  -> that gives the winning MODE (+ inverted flag)
  and the winning (train, forward) window sizes.

STAGE 2  (logic from Analyzing_v20_feeless.py)
  The winning mode + windows are fed straight into the Analyzing code
  (no re-prompts for mode/window). Results are cut (default 50%) keeping
  the lowest |Avg PnL Delta|, then sorted by highest TOTAL PnL
  (train PnL + forward PnL) and the #1 row becomes the final strategy.

Output: one Excel file in results/ with both stages + the final strategy.

Needs in the same folder:
  backtesting_window_grid.py
  backtesting_three_modes_feeless.py
  Analyzing_v20_feeless.py
"""
import time
from io import StringIO

import numpy as np
import pandas as pd

import backtesting_window_grid as bw
import backtesting_three_modes_feeless as t3
import Analyzing_v20_feeless as av


TOP_N_PER_MODE = 5
DEFAULT_CUT_PCT = 50.0


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
def choose_cut_percent():
    print("\n--- Stage 2 Results Display Filter ---")
    print(f"Keep top % of strategies by lowest |Avg PnL Delta| ({DEFAULT_CUT_PCT:g} cuts the output in half).")
    pct_str = input(f"Percentage (default {DEFAULT_CUT_PCT:g}): ").strip()
    if not pct_str:
        return DEFAULT_CUT_PCT
    try:
        pct = float(pct_str)
        if not 0 < pct <= 100:
            raise ValueError
    except ValueError:
        print(f"Invalid percentage. Using {DEFAULT_CUT_PCT:g}%.")
        return DEFAULT_CUT_PCT
    return pct


def load_pairs(group):
    pair_dfs, pair_arrays_map = {}, {}
    for pair in bw.pair_sets[group]:
        df = bw.load_csv(pair)
        if df is not None:
            pair_dfs[pair] = df
            pair_arrays_map[pair] = bw.build_pair_arrays(pair, df)
    return pair_dfs, pair_arrays_map


# ----------------------------------------------------------------------------
# STAGE 1 - three modes, window grid search, top 5 per mode, best EV
# ----------------------------------------------------------------------------
def run_stage1(news_df, has_surprise, group, pair_arrays_map, settings, modes=None, sub_modes=None):
    if modes is None:
        modes = t3.MODES
    results = {}
    for mode, inverted, label in modes:
        if mode == "surprise" and not has_surprise:
            print(f"\n=== {label}: skipped (no Actual/Forecast columns) ===")
            continue
        ev = t3.prepare_events(news_df, mode, inverted, pair_arrays_map)
        print(f"\n=== {label}: {len(ev)} usable events ===")
        if len(ev) == 0:
            continue
        grid, log, win = t3.run_grid(mode, ev, group, pair_arrays_map, settings, sub_modes=sub_modes)
        if not grid:
            print(f"{label}: no valid grid results.")
            continue
        results[label] = {
            "mode": mode,
            "inverted": inverted,
            "grid": grid,
            "log": log,
            "best_window": win,
        }
        print(f"{label}: best window (train/fwd) = {win}")
    return results


def pick_top_candidates(results, top_n=TOP_N_PER_MODE):
    """Top `top_n` windows per mode by Actual_Total_PnL_Percent, stacked."""
    frames = []
    for label, r in results.items():
        g = pd.DataFrame(r["grid"]).copy()
        g.insert(0, "Mode_Label", label)
        g.insert(1, "Mode", r["mode"])
        g.insert(2, "Inverted", r["inverted"])
        g = (
            g.sort_values("Actual_Total_PnL_Percent", ascending=False, kind="stable")
            .head(top_n)
            .reset_index(drop=True)
        )
        frames.append(g)
    top = pd.concat(frames, ignore_index=True)
    top = (
        top.sort_values(["Actual_EV_Percent", "Actual_Total_PnL_Percent"],
                        ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )
    top.insert(0, "Rank", range(1, len(top) + 1))
    top.insert(len(top.columns), "Selected", np.where(top.index == 0, "YES", ""))
    return top


# ----------------------------------------------------------------------------
# STAGE 2 - Analyzing code fed with the stage-1 winner
# ----------------------------------------------------------------------------
def build_stage2_events(raw_news, mode, inverted):
    """Rebuild the Analyzing-format news frame (with Surprise_Type) from the paste."""
    df = raw_news.copy()
    if mode == "surprise":
        df = df.dropna(subset=["Actual", "Forecast"]).copy()
        adj = df["Raw_Diff"] * (-1 if inverted else 1)
        df["Surprise_Type"] = np.where(adj > 0, "positive",
                                       np.where(adj < 0, "negative", "neutral"))
    else:
        df["Surprise_Type"] = "neutral"
    return df.sort_values("UTC_Time").reset_index(drop=True)


def run_stage2(news_df, mode, group, train_size, fwd_size, keep_pct,
               pair_dfs, pair_arrays_map, sub_modes=None):
    total_pasted = len(news_df)
    price_fit_mask = [
        any(av.has_candle(pair_arrays_map[p], t) for p in pair_arrays_map)
        for t in news_df["UTC_Time"]
    ]
    news_df = news_df[price_fit_mask].reset_index(drop=True)
    print(f"\nPasted data: {total_pasted}")
    print(f"Out of price data range: {total_pasted - len(news_df)}")
    print(f"Fitting price data: {len(news_df)}")

    if mode == "surprise":
        news_df = news_df[news_df["Surprise_Type"] != "neutral"].reset_index(drop=True)
        print(f"Filtered out neutral events. Usable surprise events: {len(news_df)}")
    else:
        valid_mask = []
        for t in news_df["UTC_Time"]:
            prev = t - pd.Timedelta(hours=1)
            valid_mask.append(any(
                av.has_candle(pair_arrays_map[p], t) and av.has_candle(pair_arrays_map[p], prev)
                for p in pair_arrays_map
            ))
        news_df = news_df[valid_mask].reset_index(drop=True)
        print(f"Filtered out events with no valid candle. Usable events: {len(news_df)}")

    total_usable = len(news_df)
    total_window = train_size + fwd_size
    if total_usable < total_window:
        print(f"\nError: window needs {train_size} train + {fwd_size} forward = {total_window} "
              f"events, but only {total_usable} usable events are available.")
        return None

    recent_news_df = news_df.iloc[-total_window:].reset_index(drop=True)
    train_news = recent_news_df.iloc[:train_size].reset_index(drop=True)
    fwd_news = recent_news_df.iloc[train_size:].reset_index(drop=True)

    print(f"\nTotal usable events available: {total_usable}")
    print(f"Putting oldest {total_usable - total_window} events aside.")
    print(f"Using newest {total_window} events:")
    print(f"  - Train Window: {len(train_news)} events "
          f"({train_news['UTC_Time'].iloc[0].date()} to {train_news['UTC_Time'].iloc[-1].date()})")
    print(f"  - Forward Test Window: {len(fwd_news)} events "
          f"({fwd_news['UTC_Time'].iloc[0].date()} to {fwd_news['UTC_Time'].iloc[-1].date()})")

    print(f"\nRunning {mode} mode on TRAIN and FORWARD TEST splits...")
    all_train_results, all_fwd_results = [], []
    for pair, df in pair_dfs.items():
        print(f"Processing {pair}...")
        pair_arrays = pair_arrays_map[pair]
        train_results = av.grid_search(pair, pair_arrays, mode, group, train_news)
        fwd_results = av.grid_search(pair, pair_arrays, mode, group, fwd_news)
        if not train_results.empty:
            all_train_results.append(train_results)
        if not fwd_results.empty:
            all_fwd_results.append(fwd_results)

    if not all_train_results and not all_fwd_results:
        print("No results generated.")
        return None

    train_all = pd.concat(all_train_results, ignore_index=True) if all_train_results else pd.DataFrame()
    fwd_all = pd.concat(all_fwd_results, ignore_index=True) if all_fwd_results else pd.DataFrame()

    # Filter by sub_modes if specified (for candle_colour mode)
    if sub_modes is not None and mode == "candle_colour":
        train_all = train_all[train_all["sub_mode"].isin(sub_modes)] if not train_all.empty else train_all
        fwd_all = fwd_all[fwd_all["sub_mode"].isin(sub_modes)] if not fwd_all.empty else fwd_all
        if train_all.empty and fwd_all.empty:
            print(f"No results for sub_modes: {sub_modes}")
            return None

    group_cols = ["mode", "sub_mode", "pair", "tp_percent", "sl_percent", "risk_to_reward_ratio"]
    train_sorted = (train_all.sort_values("total_pnl_percent", ascending=False)
                    if not train_all.empty else pd.DataFrame(columns=group_cols + av.METRIC_COLS))
    fwd_sorted = (fwd_all.sort_values("total_pnl_percent", ascending=False)
                  if not fwd_all.empty else pd.DataFrame(columns=group_cols + av.METRIC_COLS))

    merged_all = av.merge_train_fwd(train_sorted, fwd_sorted, group_cols)
    merged_all["avg_pnl_delta"] = (
        merged_all["average_pnl_percent_TRAIN"] - merged_all["average_pnl_percent_FWD"]
    ).abs()
    merged_all["Total_PnL_Percent"] = (
        merged_all["total_pnl_percent_TRAIN"].fillna(0)
        + merged_all["total_pnl_percent_FWD"].fillna(0)
    )

    total_rows = len(merged_all)
    keep_count = total_rows
    cut_df = merged_all
    if keep_pct < 100.0:
        keep_count = max(1, int(np.ceil(total_rows * keep_pct / 100.0)))
        cut_df = (
            merged_all.sort_values("avg_pnl_delta", ascending=True, kind="stable")
            .head(keep_count)
            .reset_index(drop=True)
        )
        print(f"\n|Avg PnL Delta| cutoff ({keep_pct:g}%): kept {keep_count} of {total_rows} strategies.")

    # ---- final choice: highest total PnL (train + forward) among the kept rows ----
    final = (
        cut_df.sort_values("Total_PnL_Percent", ascending=False, kind="stable")
        .reset_index(drop=True)
        .iloc[0]
    )

    # display columns (same layout as Analyzing_v20)
    display_cols = list(group_cols)
    for col in av.METRIC_COLS:
        if col in ["average_hold_hours", "total_rollover_fee_percent", "average_rollover_fee_percent"]:
            continue
        t_col, f_col = f"{col}_TRAIN", f"{col}_FWD"
        if t_col in merged_all.columns:
            display_cols.append(t_col)
        if f_col in merged_all.columns:
            display_cols.append(f_col)
        if col == "average_pnl_percent":
            display_cols.extend(["avg_pnl_delta", "Total_PnL_Percent"])

    ranked = merged_all.sort_values(
        ["avg_pnl_delta", "pair"], ascending=[True, True]
    ).reset_index(drop=True)
    ranked["number"] = range(1, len(ranked) + 1)
    ranked = ranked[["number"] + [c for c in display_cols if c in ranked.columns]]

    kept_ranked = cut_df.sort_values(
        "Total_PnL_Percent", ascending=False, kind="stable"
    ).reset_index(drop=True)
    kept_ranked["rank_by_total_pnl"] = range(1, len(kept_ranked) + 1)
    kept_ranked = kept_ranked[
        ["rank_by_total_pnl"] + [c for c in display_cols if c in kept_ranked.columns]
    ]

    best_group_cols = ["pair", "sub_mode"] if mode == "candle_colour" else ["pair"]
    best_merged = (
        cut_df.sort_values(best_group_cols + ["avg_pnl_delta"],
                           ascending=[True] * len(best_group_cols) + [True])
        .groupby(best_group_cols, as_index=False)
        .first()
    )
    best_display = best_group_cols + ["tp_percent", "sl_percent", "risk_to_reward_ratio"]
    for col in av.METRIC_COLS:
        if col in ["average_hold_hours", "total_rollover_fee_percent", "average_rollover_fee_percent"]:
            continue
        t_col, f_col = f"{col}_TRAIN", f"{col}_FWD"
        if t_col in best_merged.columns:
            best_display.append(t_col)
        if f_col in best_merged.columns:
            best_display.append(f_col)
        if col == "average_pnl_percent":
            best_display.extend(["avg_pnl_delta", "Total_PnL_Percent"])
    best_merged = best_merged[[c for c in best_display if c in best_merged.columns]]

    return {
        "final": final,
        "train_news": train_news,
        "fwd_news": fwd_news,
        "ranked": ranked,
        "kept_ranked": kept_ranked,
        "best_merged": best_merged,
        "total_rows": total_rows,
        "keep_count": keep_count,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    group = bw.choose_currency_group()
    if not group:
        return
    entry_time = bw.choose_entry_time()
    settings = bw.choose_grid_settings()
    min_train, max_train, min_fwd, max_fwd, _, _ = settings
    keep_pct = choose_cut_percent()

    # Mode set choice for standalone combined run
    print("\n--- Mode set ---")
    print("  1) Three modes")
    print("  2) Candle only")
    print("  3) Surprise only")
    raw = input("Choose mode set (default 1): ").strip()
    if raw == "2":
        mode_name, mode_list = "candle", [m for m in t3.MODES if m[0] == "candle_colour"]
    elif raw == "3":
        mode_name, mode_list = "surprise", [m for m in t3.MODES if m[0] == "surprise"]
    else:
        mode_name, mode_list = "three", list(t3.MODES)

    parsed = t3.paste_news(group, entry_time)
    if parsed is None:
        return
    news_df, has_surprise = parsed

    pair_dfs, pair_arrays_map = load_pairs(group)
    if not pair_arrays_map:
        print("No valid pair data found.")
        return

    # ======================= STAGE 1 =======================
    print("\n" + "=" * 72)
    print("STAGE 1 - Window grid search "
          f"(train {min_train}-{max_train}, fwd {min_fwd}-{max_fwd}) | modes: {', '.join(m[2] for m in mode_list)}")
    print("=" * 72)
    results = run_stage1(news_df, has_surprise, group, pair_arrays_map, settings, modes=mode_list)
    if not results:
        print("Nothing to save.")
        return

    top = pick_top_candidates(results)
    print(f"\n--- Stage 1: top {TOP_N_PER_MODE} windows per mode by "
          f"Actual_Total_PnL_Percent, ranked by Actual_EV_Percent ---")
    print(top[["Rank", "Mode_Label", "Train_Size", "Forward_Size", "Total_Trades",
               "Actual_Win_Rate_Percent", "Actual_Total_PnL_Percent",
               "Actual_EV_Percent", "Selected"]].to_string(index=False))

    winner = top.iloc[0]
    win_label = winner["Mode_Label"]
    win_mode = winner["Mode"]
    win_inverted = bool(winner["Inverted"])
    train_size = int(winner["Train_Size"])
    fwd_size = int(winner["Forward_Size"])

    # Extract sub_mode from winner for candle_colour mode
    sub_modes = None
    if win_mode == "candle_colour" and "Sub_Mode" in winner:
        sub_modes = [winner["Sub_Mode"]]

    print(f"\n*** STAGE 1 WINNER ***")
    print(f"    Mode      : {win_label} (inverted={win_inverted})")
    print(f"    Window    : train={train_size}, forward={fwd_size}")
    print(f"    EV        : {winner['Actual_EV_Percent']:.4f}%")
    print(f"    Total PnL : {winner['Actual_Total_PnL_Percent']:.4f}%")

    # ======================= STAGE 2 =======================
    print("\n" + "=" * 72)
    print(f"STAGE 2 - Analyzing | mode={win_label} | train={train_size} "
          f"fwd={fwd_size} | cut={keep_pct:g}%")
    print("=" * 72)
    news2 = build_stage2_events(news_df, win_mode, win_inverted)
    s2 = run_stage2(news2, win_mode, group, train_size, fwd_size, keep_pct,
                    pair_dfs, pair_arrays_map, sub_modes=sub_modes)
    if s2 is None:
        print("Stage 2 produced no results.")
        return

    final = s2["final"]
    final_row = pd.DataFrame([{
        "Mode_Label": win_label,
        "Mode": win_mode,
        "Inverted": win_inverted,
        "Train_Size": train_size,
        "Forward_Size": fwd_size,
        "Cut_Percent": keep_pct,
        "pair": final["pair"],
        "sub_mode": final["sub_mode"],
        "tp_percent": final["tp_percent"],
        "sl_percent": final["sl_percent"],
        "risk_to_reward_ratio": final["risk_to_reward_ratio"],
        "win_rate_percent_TRAIN": final["win_rate_percent_TRAIN"],
        "win_rate_percent_FWD": final["win_rate_percent_FWD"],
        "total_pnl_percent_TRAIN": final["total_pnl_percent_TRAIN"],
        "total_pnl_percent_FWD": final["total_pnl_percent_FWD"],
        "Total_PnL_Percent": final["Total_PnL_Percent"],
        "average_pnl_percent_TRAIN": final["average_pnl_percent_TRAIN"],
        "average_pnl_percent_FWD": final["average_pnl_percent_FWD"],
        "avg_pnl_delta": final["avg_pnl_delta"],
    }])

    print("\n*** FINAL STRATEGY (highest Total PnL among the cut results) ***")
    print(final_row.to_string(index=False))

    # ======================= SAVE =======================
    bw.OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = bw.OUTPUT_DIR / f"{group}_{timestamp}_combined_pipeline_{mode_name}_feeless.xlsx"

    selected = top.iloc[[0]].copy()
    window_row = pd.DataFrame([{
        "Mode_Label": win_label,
        "Mode": win_mode,
        "Inverted": win_inverted,
        "Train_Size": train_size,
        "Forward_Size": fwd_size,
        "Actual_Total_PnL_Percent": winner["Actual_Total_PnL_Percent"],
        "Actual_EV_Percent": winner["Actual_EV_Percent"],
        "Actual_Win_Rate_Percent": winner["Actual_Win_Rate_Percent"],
        "Total_Trades": winner["Total_Trades"],
        "Max_Drawdown_Percent": winner["Max_Drawdown_Percent"],
    }])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        final_row.to_excel(writer, sheet_name="Final_Strategy", index=False)
        window_row.to_excel(writer, sheet_name="Selected_Window", index=False)
        top.to_excel(writer, sheet_name="Stage1_Top_Candidates", index=False)
        for label, r in results.items():
            (pd.DataFrame(r["grid"])
             .sort_values("Actual_Total_PnL_Percent", ascending=False)
             .to_excel(writer, sheet_name=f"Stage1_Summary_{label}"[:31], index=False))
        s2["kept_ranked"].to_excel(writer, sheet_name="Stage2_Cut_Results", index=False)
        s2["ranked"].to_excel(writer, sheet_name="Stage2_All_Results", index=False)
        s2["best_merged"].to_excel(writer, sheet_name="Stage2_Best_Per_Pair", index=False)
        s2["train_news"].to_excel(writer, sheet_name="Train_News", index=False)
        s2["fwd_news"].to_excel(writer, sheet_name="Fwd_News", index=False)
        for sheet_name in writer.sheets:
            writer.sheets[sheet_name].freeze_panes = "A2"

    print("\n--- FINISHED ---")
    print(f"Strategies before cut : {s2['total_rows']}")
    print(f"Strategies after cut  : {s2['keep_count']} ({keep_pct:g}%)")
    print(f"File saved            : {out_path}")


if __name__ == "__main__":
    main()
