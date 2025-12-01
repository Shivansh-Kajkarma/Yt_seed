import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Setup Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.mongo_utils import load_collection_as_df

# --- Config ---
RUN_TAG = "colinandsammer"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print(f"🔍 Loading Phase 3 Step 3A data for run_tag: {RUN_TAG}")

    # Load from MongoDB
    collection_name = f"{RUN_TAG.upper()}_phase3_step3a"

    try:
        df = load_collection_as_df(collection_name)

        if df.empty:
            print(f"❌ No data found in collection: {collection_name}")
            return

        print(f"✅ Loaded {len(df)} records from MongoDB")

        # Display basic info
        print(f"\nColumns: {list(df.columns)}")
        # print(f"\nFirst record:")
        # print(df.iloc[0].to_dict())

        # Filter to specific columns
        desired_columns = [
            "Discovered_Channel_Name",
            "score_format_match",
            
        ]

        # Keep only columns that exist in the dataframe
        columns_to_save = [col for col in desired_columns if col in df.columns]
        df_filtered = df[columns_to_save]

        # Sort: All True first, then all False
        if "score_format_match" in df_filtered.columns:
            df_filtered = df_filtered.sort_values("score_format_match", ascending=False)

        print(f"\n📋 Saving {len(columns_to_save)} columns: {columns_to_save}")
        print(f"📊 Sorted by score_format_match (True first, then False)")

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"{RUN_TAG}_phase3_step3a_{timestamp}.csv"
        df_filtered.to_csv(csv_path, index=False)
        print(f"\n💾 Saved to: {csv_path}")

        # Save to JSON (for detailed analysis)
        json_path = OUTPUT_DIR / f"{RUN_TAG}_phase3_step3a_{timestamp}.json"
        df.to_json(json_path, orient="records", indent=2)
        print(f"💾 Saved to: {json_path}")

        # Basic stats
        print(f"\n📊 Statistics:")
        print(f"   Total Channels: {len(df)}")

        if "score_format_match" in df.columns:
            print(f"\n   Format Match Distribution:")
            print(df["score_format_match"].value_counts())
            print(
                f"\n   Format Match Rate: {df['score_format_match'].sum()} / {len(df)} ({df['score_format_match'].mean() * 100:.1f}%)"
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
