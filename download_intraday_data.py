import databento as db

client = db.Historical("db-659Q6GUhP4J4WA8gsNQCBxqK9V8eu")

# 1. Check if records actually exist for this range
# Changed to Jan 2nd (Tuesday) because Jan 1st was a holiday
dataset = "XNAS.ITCH"
symbol = "AAPL"
start = "2024-01-02T14:30:00" # Market Open (UTC)
end = "2024-01-02T21:00:00"   # Market Close (UTC)

count = client.metadata.get_record_count(
    dataset=dataset,
    symbols=[symbol],
    schema="ohlcv-1m",
    start=start,
    end=end
)

if count == 0:
    print(f"No records found for {symbol} on this date. It might be a weekend or holiday.")
else:
    print(f"Found {count} records. Downloading...")
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema="ohlcv-1m",
        start=start,
        end=end,
        path=f"{symbol}_test_data.dbn.zst"
    )
    
    # Store as CSV
    data.to_csv(f"{symbol}_test_data.csv", pretty_ts=True, pretty_px=True)
    print("Success: CSV created with data.")