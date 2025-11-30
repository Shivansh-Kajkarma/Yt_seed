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
RUN_TAG = "lennypodcat"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print(f"🔍 Loading Phase 2 data for run_tag: {RUN_TAG}")

    # Load from MongoDB
    collection_name = f"{RUN_TAG.upper()}_phase2"

    try:
        df = load_collection_as_df(collection_name)

        if df.empty:
            print(f"❌ No data found in collection: {collection_name}")
            return

        print(f"✅ Loaded {len(df)} records from MongoDB")

        # Display basic info
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst record:")
        print(df.iloc[0].to_dict())

        # Filter to only desired columns
        desired_columns = [
            "Discovered_Channel_ID",
            "Discovered_Channel_Name",
            "Discovered_Channel_URL",
            "Discovered_Country",
            "Discovered_Subs",
            "Discovered_Video_Count",
            "Discovered_Videos_JSON",
            "Seed_Channel_Name",
            "Timestamp",
            "run_tag",
        ]

        # Keep only columns that exist in the dataframe
        columns_to_save = [col for col in desired_columns if col in df.columns]
        df_filtered = df[columns_to_save]

        print(f"\n📋 Saving {len(columns_to_save)} columns: {columns_to_save}")

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"{RUN_TAG}_phase2_{timestamp}.csv"
        df_filtered.to_csv(csv_path, index=False)
        print(f"\n💾 Saved to: {csv_path}")

        # Save to JSON (for detailed analysis)
        json_path = OUTPUT_DIR / f"{RUN_TAG}_phase2_{timestamp}.json"
        df.to_json(json_path, orient="records", indent=2)
        print(f"💾 Saved to: {json_path}")

        # Basic stats
        print(f"\n📊 Statistics:")
        print(f"   Total Discovered Channels: {df['Discovered_Channel_ID'].nunique()}")
        print(f"   Discovery Levels: {df['Discovery_Level'].value_counts().to_dict()}")
        print(f"   Top Seed Channels:")
        print(df["Seed_Channel_Name"].value_counts().head())

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
