"""
TARDIS BULK DOWNLOADER
======================
Downloads all free first-of-month data from Tardis.dev
for BTCUSDT on Binance Futures, from 2020 to present.

Free tier = first day of every month, no API key needed.
2020-2026 = ~72 months = ~72 days of tick-level data per type.

Data types downloaded:
  - book_snapshot_25  → order book (top 25 levels) → imbalance, walls
  - trades            → tick trades → CVD reconstruction
  - liquidations      → forced liquidations → cascade detection
  - derivative_ticker → funding rate + OI at tick level

Usage:
    python 0a_tardis_download.py                    # Download everything
    python 0a_tardis_download.py --type trades      # Single type only
    python 0a_tardis_download.py --from 2023-01     # Start from specific month
    python 0a_tardis_download.py --workers 4        # Parallel downloads
"""

import os
import time
import argparse
import requests
import gzip
import shutil
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
TARDIS_BASE   = "https://datasets.tardis.dev/v1"
EXCHANGE      = "binance-futures"
SYMBOL        = "BTCUSDT"
PERP_SYMBOL   = "BTCUSDT"          # For liquidations file
DATA_DIR      = "./data/tardis"
START_MONTH   = "2020-01"          # Binance Futures BTCUSDT launched ~Sep 2019
END_MONTH     = None               # None = up to current month

# Data types and their URL patterns
# Format: (type_name, url_symbol, local_prefix)
DATA_TYPES = {
    "book_snapshot_25": {
        "url":    f"{TARDIS_BASE}/{EXCHANGE}/book_snapshot_25/{{year}}/{{month:02d}}/01/{SYMBOL}.csv.gz",
        "desc":   "Order book snapshots (top 25 levels) → imbalance + wall detection",
    },
    "trades": {
        "url":    f"{TARDIS_BASE}/{EXCHANGE}/trades/{{year}}/{{month:02d}}/01/{SYMBOL}.csv.gz",
        "desc":   "Tick-by-tick trades → CVD reconstruction",
    },
    "liquidations": {
        "url":    f"{TARDIS_BASE}/{EXCHANGE}/liquidations/{{year}}/{{month:02d}}/01/{PERP_SYMBOL}.csv.gz",
        "desc":   "Liquidation events → cascade detection",
    },
    "derivative_ticker": {
        "url":    f"{TARDIS_BASE}/{EXCHANGE}/derivative_ticker/{{year}}/{{month:02d}}/01/{SYMBOL}.csv.gz",
        "desc":   "Funding rate + open interest at tick level",
    },
}

os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────
# GENERATE ALL MONTH DATES
# ─────────────────────────────────────────
def generate_months(start_str, end_str=None):
    start = datetime.strptime(start_str, "%Y-%m").replace(tzinfo=timezone.utc)

    if end_str:
        end = datetime.strptime(end_str, "%Y-%m").replace(tzinfo=timezone.utc)
    else:
        # Up to last completed month
        now = datetime.now(timezone.utc)
        end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end -= relativedelta(months=1)

    months = []
    current = start
    while current <= end:
        months.append(current)
        current += relativedelta(months=1)

    return months


# ─────────────────────────────────────────
# DOWNLOAD SINGLE FILE
# ─────────────────────────────────────────
def download_file(data_type, year, month, force=False):
    """
    Download and decompress a single Tardis file.
    Returns (success, filepath, message)
    """
    config = DATA_TYPES[data_type]
    url = config["url"].format(year=year, month=month)

    # Local path
    type_dir = os.path.join(DATA_DIR, data_type)
    os.makedirs(type_dir, exist_ok=True)

    gz_path  = os.path.join(type_dir, f"{year}-{month:02d}-01.csv.gz")
    csv_path = os.path.join(type_dir, f"{year}-{month:02d}-01.csv")

    # Skip if already downloaded and decompressed
    if os.path.exists(csv_path) and not force:
        size_mb = os.path.getsize(csv_path) / 1024 / 1024
        return True, csv_path, f"already exists ({size_mb:.1f} MB)"

    try:
        # Stream download
        r = requests.get(url, stream=True, timeout=60)

        if r.status_code == 404:
            return False, None, "not available (404)"
        if r.status_code == 401:
            return False, None, "requires API key (401)"
        if r.status_code != 200:
            return False, None, f"HTTP {r.status_code}"

        # Save gz file
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Decompress
        with gzip.open(gz_path, "rb") as f_in:
            with open(csv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove gz after decompressing
        os.remove(gz_path)

        size_mb = os.path.getsize(csv_path) / 1024 / 1024
        return True, csv_path, f"downloaded ({size_mb:.1f} MB)"

    except requests.exceptions.Timeout:
        return False, None, "timeout"
    except Exception as e:
        # Clean up partial files
        for p in [gz_path, csv_path]:
            if os.path.exists(p):
                os.remove(p)
        return False, None, f"error: {e}"


# ─────────────────────────────────────────
# DOWNLOAD ALL
# ─────────────────────────────────────────
def download_all(data_types, months, workers=3):
    tasks = []
    for month_dt in months:
        for dtype in data_types:
            tasks.append((dtype, month_dt.year, month_dt.month))

    total       = len(tasks)
    succeeded   = 0
    failed      = 0
    skipped     = 0
    total_mb    = 0.0

    print(f"\nDownloading {total} files ({len(months)} months x {len(data_types)} types)")
    print(f"Workers: {workers} | Output: {DATA_DIR}\n")

    def do_download(task):
        dtype, year, month = task
        return dtype, year, month, download_file(dtype, year, month)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(do_download, t): t for t in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            dtype, year, month, (ok, filepath, msg) = future.result()
            label = f"{year}-{month:02d} {dtype}"

            if ok:
                if "already exists" in msg:
                    skipped += 1
                    status = "SKIP"
                else:
                    succeeded += 1
                    status = "OK  "
                    # Extract MB from message
                    try:
                        mb = float(msg.split("(")[1].split(" ")[0])
                        total_mb += mb
                    except:
                        pass
            else:
                failed += 1
                status = "FAIL"

            print(f"  [{i:>4}/{total}] {status} {label:<55} {msg}")

    print(f"\n{'='*60}")
    print(f"Complete: {succeeded} downloaded, {skipped} skipped, {failed} failed")
    print(f"New data: {total_mb:.1f} MB")

    return succeeded, failed


# ─────────────────────────────────────────
# VERIFY DOWNLOADS
# ─────────────────────────────────────────
def verify_downloads(data_types, months):
    print("\n=== DOWNLOAD SUMMARY ===\n")

    for dtype in data_types:
        type_dir = os.path.join(DATA_DIR, dtype)
        files    = sorted(os.listdir(type_dir)) if os.path.exists(type_dir) else []
        csvs     = [f for f in files if f.endswith(".csv")]
        total_mb = sum(
            os.path.getsize(os.path.join(type_dir, f)) for f in csvs
        ) / 1024 / 1024

        print(f"  {dtype}")
        print(f"    Files:    {len(csvs)} months")
        print(f"    Size:     {total_mb:.1f} MB")
        if csvs:
            print(f"    Range:    {csvs[0][:7]} → {csvs[-1][:7]}")
        print()

    print("Next step: python 0b_process_tardis.py")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",    type=str, default=None,
                        help="Single data type: book_snapshot_25, trades, liquidations, derivative_ticker")
    parser.add_argument("--from",    dest="from_month", type=str, default=START_MONTH,
                        help="Start month YYYY-MM (default: 2020-01)")
    parser.add_argument("--to",      dest="to_month",   type=str, default=None,
                        help="End month YYYY-MM (default: last completed month)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel download workers (default: 3, be gentle)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-download even if file exists")
    parser.add_argument("--verify",  action="store_true",
                        help="Just show what has been downloaded")
    args = parser.parse_args()

    # Install dateutil if needed
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        print("Installing python-dateutil...")
        os.system("pip install python-dateutil -q")
        from dateutil.relativedelta import relativedelta

    # Which types to download
    if args.type:
        if args.type not in DATA_TYPES:
            print(f"Unknown type: {args.type}")
            print(f"Valid types: {list(DATA_TYPES.keys())}")
            exit(1)
        selected_types = [args.type]
    else:
        selected_types = list(DATA_TYPES.keys())

    # Generate month list
    months = generate_months(args.from_month, args.to_month)

    print("=== TARDIS BULK DOWNLOADER ===")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol:   {SYMBOL}")
    print(f"Period:   {args.from_month} → {months[-1].strftime('%Y-%m')}")
    print(f"Months:   {len(months)}")
    print(f"Types:    {selected_types}")
    print()
    for dtype in selected_types:
        print(f"  {dtype}: {DATA_TYPES[dtype]['desc']}")

    if args.verify:
        verify_downloads(selected_types, months)
        exit(0)

    # Warn about size
    estimated_gb = len(months) * len(selected_types) * 0.15  # rough ~150MB per file
    print(f"\nEstimated size: ~{estimated_gb:.0f} GB (varies heavily by month)")
    print("Tip: trades files are largest (~500MB/day), book snapshots ~200MB, liquidations tiny.")
    print("Start with derivative_ticker + liquidations if storage is limited.\n")

    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        exit(0)

    download_all(selected_types, months, workers=args.workers)
    verify_downloads(selected_types, months)