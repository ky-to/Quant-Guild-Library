import databento as db
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

# 1. Setup Configuration
API_KEY = "db-659Q6GUhP4J4WA8gsNQCBxqK9V8eu"
# Note: Ensure the dataset (XNAS.ITCH) supports these specific symbols
SYMBOLS = ["AAPL"]  # , "COST", "AMZN", "BRK.B", "GOOGL", "NVDA", "TSM", "SIXE"]
START_TIME = "2018-05-01T09:30:00"
END_TIME = "2026-03-14T16:00:00"
DATASET = "XNAS.ITCH"

client = db.Historical(API_KEY)


def safe_path_part(value: str) -> str:
    return value.replace(":", "-").replace(".", "_")


def iter_daily_ranges(start_str: str, end_str: str):
    start_dt = datetime.fromisoformat(start_str)
    end_dt = datetime.fromisoformat(end_str)

    session_start = start_dt.time()
    session_end = end_dt.time()

    current_date = start_dt.date()
    final_date = end_dt.date()

    while current_date <= final_date:
        day_start = datetime.combine(current_date, session_start)
        day_end = datetime.combine(current_date, session_end)

        chunk_start = max(day_start, start_dt)
        chunk_end = min(day_end, end_dt)

        if chunk_start < chunk_end:
            yield chunk_start, chunk_end

        current_date += timedelta(days=1)


def download_and_convert_to_csv(symbol):
    """Downloads 1-minute OHLCV data and saves one CSV per day."""
    safe_symbol = symbol.replace(".", "_")
    window_label = f"{safe_path_part(START_TIME)}_to_{safe_path_part(END_TIME)}"
    output_dir = Path(f"{safe_symbol}_{window_label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start, chunk_end in iter_daily_ranges(START_TIME, END_TIME):
        day_label = chunk_start.strftime("%Y-%m-%d")
        filename = output_dir / f"{safe_symbol}_{day_label}.csv"

        print(f"Requesting data for {symbol} on {day_label}...")

        try:
            count = client.metadata.get_record_count(
                dataset=DATASET,
                symbols=[symbol],
                schema="ohlcv-1m",
                start=chunk_start.isoformat(timespec="seconds"),
                end=chunk_end.isoformat(timespec="seconds"),
            )

            if count == 0:
                print(f"No data for {symbol} on {day_label}, skipping.")
                continue

            data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=[symbol],
                schema="ohlcv-1m",
                start=chunk_start.isoformat(timespec="seconds"),
                end=chunk_end.isoformat(timespec="seconds"),
            )

            data.to_csv(str(filename), pretty_ts=True, pretty_px=True)
            print(f"Successfully saved: {filename}")

        except Exception as e:
            print(f"Error processing {symbol} on {day_label}: {e}")


# 2. Run with 5 parallel workers
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(download_and_convert_to_csv, SYMBOLS)

print("Batch processing complete.")