# /utils/youtube_utils.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import re
import html  # <-- ADD THIS IMPORT
from datetime import datetime  # <-- ADD THIS IMPORT

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# -----------------------------
# Utility: Extract Channel ID
# -----------------------------
def extract_channel_id(channel_url: str) -> str:
    """
    Convert YouTube channel URL or handle to channel ID using the YouTube API.
    Supports '@handle', '/channel/ID', and '/c/customName' formats.
    """
    # ... (This function is already perfect, no changes)
    match = re.search(r"channel/([A-Za-z0-9_-]+)", channel_url)
    if match:
        return match.group(1)

    # 2️⃣ Handle or custom name → use search API to resolve
    handle = channel_url.split("/")[-1]
    if handle.startswith("@"):
        handle = handle[1:]

    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=channel&q={handle}&key={API_KEY}"
    resp = requests.get(url)
    data = resp.json()

    if "items" in data and data["items"]:
        return data["items"][0]["snippet"]["channelId"]
    else:
        print(f"⚠️ Could not resolve channel ID for {channel_url}")
        return ""


# -----------------------------
# Fetch recent videos per channel
# -----------------------------
def fetch_recent_videos(channel_id: str, max_results: int = 5):
    """
    Fetch recent videos for a channel by channel_id.
    Returns a list of dicts with {video_id, title, description, published_at}.
    """
    url = (
        f"https://www.googleapis.com/youtube/v3/search?"
        f"key={API_KEY}&channelId={channel_id}&part=snippet,id&order=date&maxResults={max_results}"
    )

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"⚠️ Failed to fetch for {channel_id}: {resp.text}")
        return []

    data = resp.json()
    results = []

    for item in data.get("items", []):
        if item["id"]["kind"] == "youtube#video":
            snippet = item["snippet"]
            
            # --- START: FIXES ---
            
            # 1. Clean title: "What&#39;s" -> "What's"
            clean_title = html.unescape(snippet["title"])
            
            # 2. Clean description: Unescape and remove newlines that break CSVs
            raw_desc = snippet.get("description", "")
            clean_desc = html.unescape(raw_desc).replace("\n", " ").replace("\r", " ")
            
            # 3. Format timestamp: "2025-10-21T17:02:11Z" -> "2025-10-21 17:02:11"
            raw_time = snippet["publishedAt"]
            # Parse the ISO format string (replace 'Z' for compatibility)
            dt_obj = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
            # Format it into a simple, readable string
            friendly_time = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

            # --- END: FIXES ---

            results.append({
                "video_id": item["id"]["videoId"],
                "title": clean_title,       # <-- Use cleaned title
                "description": clean_desc,    # <-- Use cleaned description
                "published_at": friendly_time # <-- Use formatted time
            })
    return results


# -----------------------------
# Fetch for all seed channels
# -----------------------------
def fetch_for_seed_channels(seed_df: pd.DataFrame, limit_per_channel: int = 5) -> pd.DataFrame:
    """
    Given a seed channel DataFrame (Channel_Name + Channel_URL), fetch recent videos for each.
    """
    # ... (This function is also perfect, no changes)
    all_records = []

    for _, row in seed_df.iterrows(): 
        channel_url = row["Channel_URL"]
        channel_name = row["Channel_Name"]

        print(f"🎯 Fetching recent videos for: {channel_name}")
        channel_id = extract_channel_id(channel_url)
        if not channel_id:
            continue

        videos = fetch_recent_videos(channel_id, limit_per_channel)
        for v in videos:
            all_records.append({
                "Channel_Name": channel_name,
                "Channel_ID": channel_id,
                **v
            })

    df_videos = pd.DataFrame(all_records)
    print(f"✅ Fetched {len(df_videos)} total videos")
    return df_videos