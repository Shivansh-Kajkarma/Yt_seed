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
    print(f"🔍 Loading Final Ranked data for run_tag: {RUN_TAG}")

    # Load from MongoDB
    collection_name = f"{RUN_TAG.upper()}_final_ranked"

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

        # Filter to specific columns
        desired_columns = [
            "Discovered_Channel_Name",
            "Discovered_Subs",
            "Final_Status",
            "Final_Tier",
            "score_format_confidence",
            "score_format_match",
            "score_similarity",
            "sub_score_content",
            "sub_score_intent",
            "sub_score_keywords",
            "sub_score_niche",
            "Discovered_Channel_URL",
        ]

        # Keep only columns that exist in the dataframe
        columns_to_save = [col for col in desired_columns if col in df.columns]
        df_filtered = df[columns_to_save]

        print(f"\n📋 Saving {len(columns_to_save)} columns: {columns_to_save}")

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"{RUN_TAG}_final_ranked_{timestamp}.csv"
        df_filtered.to_csv(csv_path, index=False)
        print(f"\n💾 Saved to: {csv_path}")

        # Save to JSON (for detailed analysis)
        json_path = OUTPUT_DIR / f"{RUN_TAG}_final_ranked_{timestamp}.json"
        df.to_json(json_path, orient="records", indent=2)
        print(f"💾 Saved to: {json_path}")

        # Basic stats
        print(f"\n📊 Statistics:")
        print(f"   Total Channels: {len(df)}")

        if "Final_Tier" in df.columns:
            print(f"\n   Tier Distribution:")
            print(df["Final_Tier"].value_counts().sort_index())

        if "Format_Match" in df.columns:
            print(f"\n   Format Match: {df['Format_Match'].sum()} / {len(df)}")

        if "Weighted_Score" in df.columns:
            print(f"\n   Score Stats:")
            print(f"      Mean: {df['Weighted_Score'].mean():.3f}")
            print(f"      Max: {df['Weighted_Score'].max():.3f}")
            print(f"      Min: {df['Weighted_Score'].min():.3f}")

        print(f"\n   Top Seed Channels:")
        print(df["Seed_Channel_Name"].value_counts().head())

        # Show top channels by score
        if "Weighted_Score" in df.columns:
            print(f"\n   🏆 Top 10 Channels by Weighted Score:")
            top_10 = df.nlargest(10, "Weighted_Score")[
                [
                    "Discovered_Channel_Name",
                    "Weighted_Score",
                    "Final_Tier",
                    "Format_Match",
                ]
            ]
            print(top_10.to_string(index=False))

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
