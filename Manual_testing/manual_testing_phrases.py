import pandas as pd
from pathlib import Path
from utils.youtube_utils import (
    search_videos_multi_focused,
    get_channel_metadata_batch
)
import pprint

# --- Configuration ---
BASE = Path(__file__).resolve().parent

# The 8 keywords you want to test
TEST_KEYWORDS = [
    "build lifestyle business",
    "boring business ideas",
    "entrepreneurship strategies",
    "business case study",
    "passive income strategies",
    "financial freedom journey",
    "wealth building habits",
    "money mindset shifts",
    "millionaire habits",
    "financial independence tips",
    "productivity desk setup",
    "time management techniques",
    "self improvement journey",
    "discipline and motivation",
    "habit formation strategies",
     "creator monetization strategies",
    "start youtube channel",
    "personal brand building",
    "content creator tips",
    "youtube growth strategies"
]

# Ali Abdaal's Channel ID to exclude him from the results
# (Found using the API search, his handle is @aliabdaal)
SEED_CHANNEL_ID_TO_EXCLUDE = "UCoOae5nYA7VqaXzerajD0lg"

# Search settings
RESULTS_PER_KEYWORD = 20  # How many channels to fetch per keyword
MAX_KEYWORDS_TO_SEARCH = len(TEST_KEYWORDS) # We want to use all 8 of your keywords

def main():
    print("=" * 70)
    print("MANUAL KEYWORD SEARCH TEST")
    print(f"Testing {len(TEST_KEYWORDS)} keywords related to Ali Abdaal...")
    print(f"Fetching {RESULTS_PER_KEYWORD} channels per keyword.")
    print("=" * 70)

    # --- 1. Search using your existing function ---
    try:
        candidate_ids_set = search_videos_multi_focused(
            keywords=TEST_KEYWORDS,
            max_results_per_search=RESULTS_PER_KEYWORD,
            max_keywords=MAX_KEYWORDS_TO_SEARCH
        )
    except Exception as e:
        print(f"❌ Error during search: {e}")
        print("   Make sure your youtube_utils.py file is correct and API_KEY is set.")
        return

    if not candidate_ids_set:
        print("No candidate channels were found.")
        return

    # --- 2. Filter out the seed channel ---
    print(f"Found {len(candidate_ids_set)} total unique channels.")
    candidate_ids_set.discard(SEED_CHANNEL_ID_TO_EXCLUDE)
    
    if not candidate_ids_set:
        print("Search only found the seed channel itself. No new channels found.")
        return

    print(f"Found {len(candidate_ids_set)} new unique candidate channels (after removing seed).")

    # --- 3. Fetch metadata to see who they are ---
    print("\nFetching metadata for found channels...")
    candidate_ids_list = list(candidate_ids_set)
    
    try:
        metadata = get_channel_metadata_batch(candidate_ids_list)
    except Exception as e:
        print(f"❌ Error fetching metadata: {e}")
        print("Found IDs (unverified):")
        pprint.pprint(candidate_ids_list)
        return

    if not metadata:
        print("❌ Fetched metadata, but the list was empty.")
        return

    # --- 4. Display the results clearly ---
    print("\n" + "=" * 70)
    print("TEST RESULTS: CHANNELS FOUND")
    print("=" * 70)
    
    df = pd.DataFrame(metadata)
    
    # Ensure 'subscribers' is numeric, handling potential errors
    df['subscribers'] = pd.to_numeric(df['subscribers'], errors='coerce').fillna(0).astype(int)
    
    # Add a 'Subscribers (M)' column for easier reading
    df['subscribers_M'] = df['subscribers'].apply(lambda x: x / 1_000_000)
    df = df.sort_values(by='subscribers', ascending=False)

    # Print a clean table to the console
    print(f"{'Channel Name':<40} {'Subscribers (M)':<15} {'Video Count':<10} {'Channel ID':<25}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        print(f"{row['name'][:38]:<40} {row['subscribers_M']:<15.2f} {row['video_count']:<10} {row['id']:<25}")

    # Save the full results to a CSV file for you
    output_csv = BASE / "manual_keyword_test_results.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 70)
    print(f"✅ Full results saved to {output_csv.name}")

if __name__ == "__main__":
    main()




