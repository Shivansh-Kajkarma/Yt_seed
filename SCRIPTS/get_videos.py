import pandas as pd
from pathlib import Path
from utils.youtube_utils import fetch_for_seed_channels

BASE = Path(__file__).resolve().parent

def main():
    seed_csv = BASE / "seed_channels.csv"
    out_csv = BASE / "sample_videos_moon.csv"

    if not seed_csv.exists():
        print(f"ERROR: {seed_csv} not found. Place seed_channels.csv in project root.")
        return

    seed_df = pd.read_csv(seed_csv)
    # validate columns
    if 'Channel_Name' not in seed_df.columns or 'Channel_URL' not in seed_df.columns:
        print("ERROR: seed_channels.csv must have Channel_Name and Channel_URL columns.")
        return

    # optionally limit to first N channels for quota safety (configurable)
    # For live run you can remove .head(2)
    seed_to_process = seed_df  # e.g., process first 9 seeds; change as needed

    print("Loading seed channels and fetching videos (API mode)...")
    df_videos = fetch_for_seed_channels(seed_to_process, limit_per_channel=30, filter_shorts=True)

    if df_videos is not None and not df_videos.empty:
        df_videos.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Success! Saved {len(df_videos)} videos to '{out_csv.name}'")
    else:
        print("No videos fetched; check logs.")

if __name__ == "__main__":
    main()
