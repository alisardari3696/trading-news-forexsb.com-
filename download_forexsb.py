"""
download_forexsb.py — Download historical forex data from forexsb.com.

forexsb only provides M1/M5/M15/M30 binary data via Dukascopy.
H1/H4/D1 are generated client-side via LTF (Last-Timeframe aggregation).
This script downloads M30 and resamples to H1, matching your existing *60.csv format.

Usage:
    python download_forexsb.py USD        # 7 pairs for USD
    python download_forexsb.py EUR GBP    # multiple currencies
    python download_forexsb.py ALL        # all 28 pairs

Output: ./<PAIR>60.csv  (tab-separated, no header, same format as your existing files)
"""

import gzip
import ssl
import struct
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Bypass SSL in VM environments where certs are expired
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(".")
BASE_URL = "https://data.forexsb.com/datafeed/data/dukascopy"

# Currency group -> pairs (same as your backtesting scripts)
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

# Source period to download (M30 → resample to H1)
SRC_PERIOD = 30
# Target period code (H1 = 60)
TARGET_PERIOD = 60

MILLENNIUM = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def download_raw(symbol: str, period: int) -> bytes | None:
    """Download the .lb.gz binary file for a symbol+period."""
    url = f"{BASE_URL}/{symbol}{period}.lb.gz"
    print(f"  Fetching {url} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status}")
                return None
            raw = gzip.decompress(resp.read())
            print(f"OK ({len(raw)} bytes)")
            return raw
    except Exception as e:
        print(f"FAIL: {e}")
        return None


def parse_binary(data: bytes, symbol: str) -> list[dict]:
    """Parse the dukascopy M30 binary into OHLCV bars."""
    if len(data) == 0:
        return []

    bar_size = 28 if len(data) % 28 == 0 else 24
    n_bars = len(data) // bar_size
    price_scale = 1000 if symbol.endswith("JPY") else 100000

    bars = []
    for i in range(n_bars):
        offset = i * bar_size
        ts_ms = MILLENNIUM + struct.unpack_from("<i", data, offset)[0] * 60_000
        open_ = struct.unpack_from("<i", data, offset + 4)[0]
        high = struct.unpack_from("<i", data, offset + 8)[0]
        low = struct.unpack_from("<i", data, offset + 12)[0]
        close = struct.unpack_from("<i", data, offset + 16)[0]
        volume = struct.unpack_from("<i", data, offset + 20)[0]
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        bars.append({
            "datetime": dt,
            "open": open_ / price_scale,
            "high": high / price_scale,
            "low": low / price_scale,
            "close": close / price_scale,
            "volume": volume,
        })

    return bars


def resample_to_h1(bars: list[dict]) -> list[dict]:
    """Resample M30 bars to H1 using the same LTF logic as forexsb JS app."""
    h1 = defaultdict(lambda: {"open": None, "high": 0, "low": 999, "close": 0, "volume": 0})

    for b in bars:
        hour = b["datetime"].replace(minute=0, second=0, microsecond=0)
        d = h1[hour]
        if d["open"] is None:
            d["open"] = b["open"]
        if b["high"] > d["high"]:
            d["high"] = b["high"]
        if b["low"] < d["low"]:
            d["low"] = b["low"]
        d["close"] = b["close"]
        d["volume"] += b["volume"]

    result = []
    for dt in sorted(h1.keys()):
        d = h1[dt]
        if d["open"] is not None:
            result.append({
                "datetime": dt,
                "open": d["open"],
                "high": d["high"],
                "low": d["low"],
                "close": d["close"],
                "volume": d["volume"],
            })
    return result


def save_csv(pair: str, bars: list[dict]) -> Path:
    """Save as tab-separated CSV with no header row (matches your *60.csv format)."""
    out_path = DATA_DIR / f"{pair}60.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        for b in bars:
            dt = b["datetime"].strftime("%Y-%m-%d %H:%M")
            f.write(
                f"{dt}\t{b['open']}\t{b['high']}\t{b['low']}\t"
                f"{b['close']}\t{b['volume']}\n"
            )
    print(f"  Saved {out_path}  ({len(bars)} bars)")
    return out_path


def choose_currency_groups():
    print("\nAvailable currency groups:")
    groups = list(pair_sets.keys())
    for i, g in enumerate(groups, 1):
        pairs = pair_sets[g]
        print(f"  {i}. {g}  ({', '.join(pairs)})")
    print()
    choice = input("Enter number(s) or currency code(s) (e.g. '1' or 'USD' or '1,3,5'): ").strip()

    selected = []
    for part in choice.split(','):
        part = part.strip()
        if part in pair_sets:
            selected.append(part)
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(groups):
                selected.append(groups[idx])
            else:
                print(f"  WARNING: Invalid number '{part}', skipping.")
        else:
            print(f"  WARNING: Unknown '{part}', skipping.")

    if not selected:
        print("Nothing selected.")
        return None
    return selected


def main():
    groups = choose_currency_groups()
    if not groups:
        return

    # Resolve currency groups -> pairs
    pairs = []
    for g in groups:
        pairs.extend(pair_sets[g.upper().strip()])

    # Deduplicate preserving order
    seen = set()
    unique_pairs = [p for p in pairs if not (p in seen or seen.add(p))]

    print(f"Will download {len(unique_pairs)} pairs (M30 → H1):")
    for p in unique_pairs:
        print(f"  {p}")
    print()

    ok = 0
    fail = 0
    skipped = 0
    for pair in unique_pairs:
        out_path = DATA_DIR / f"{pair}60.csv"
        if out_path.exists():
            print(f"[{pair}] Already exists, skipping.")
            skipped += 1
            continue
        print(f"[{pair}]")
        raw = download_raw(pair, SRC_PERIOD)
        if raw is None or len(raw) == 0:
            print(f"  SKIPPED (no data)")
            fail += 1
            continue

        bars = parse_binary(raw, pair)
        if not bars:
            print(f"  SKIPPED (parse error)")
            fail += 1
            continue

        h1_bars = resample_to_h1(bars)
        save_csv(pair, h1_bars)
        ok += 1

    print(f"\nDone: {ok} downloaded, {fail} failed, {skipped} skipped.")


if __name__ == "__main__":
    main()