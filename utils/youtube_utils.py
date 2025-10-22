import os
import requests
import pandas as pd
from dotenv import load_dotenv
import re
import html
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# -----------------------------
# Utility: Extract Channel ID
# -----------------------------
def extract_channel_id(channel_url: str) -> str:
    
    match = re.search(r"channel/([A-Za-z0-9_-]+)", channel_url)
    if match:
        return match.group(1)
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
    
    This now uses a 2-step process:
    1. Search API to find recent video_ids.
    2. Videos API to get full descriptions for those IDs.
    """
    
    # --- STEP 1: Use Search API to find recent video IDs ---
    search_url = (
        f"https://www.googleapis.com/youtube/v3/search?"
        f"key={API_KEY}&channelId={channel_id}&part=snippet,id&order=date&maxResults={max_results}"
    )

    resp_search = requests.get(search_url)
    if resp_search.status_code != 200:
        print(f"⚠️ Failed (Search) for {channel_id}: {resp_search.text}")
        return []

    data_search = resp_search.json()
    video_details = {}  # Use a dict for easy lookup
    video_ids = []

    for item in data_search.get("items", []):
        if item["id"]["kind"] == "youtube#video":
            video_id = item["id"]["videoId"]
            video_ids.append(video_id)
            
            # store the basic info for now
            video_details[video_id] = {
                "video_id": video_id,
                "title": html.unescape(item["snippet"]["title"]),
                "published_at": item["snippet"]["publishedAt"], # Will format this later
                "description": "" # Will be filled by Step 2
            }
            
    if not video_ids:
        print(f"No recent videos found for {channel_id}")
        return []

    # --- STEP 2: Use Videos API to get full descriptions ---
    video_ids_str = ",".join(video_ids)
    videos_url = (
        f"https://www.googleapis.com/youtube/v3/videos?"
        f"key={API_KEY}&part=snippet&id={video_ids_str}"
    )
    
    resp_videos = requests.get(videos_url)
    if resp_videos.status_code != 200:
        print(f"⚠️ Failed (Videos) for {channel_id}: {resp_videos.text}")
        # We can still return the basic info from Step 1
        return list(video_details.values()) 

    data_videos = resp_videos.json()

    # video_details with full descriptions
    for item in data_videos.get("items", []):
        video_id = item["id"]
        if video_id in video_details:
            # 1. Get full description
            raw_desc = item["snippet"].get("description", "")
            # 2. Clean it (unescape HTML and remove CSV-breaking newlines)
            clean_desc = html.unescape(raw_desc).replace("\n", " ").replace("\r", " ")
            
            video_details[video_id]["description"] = clean_desc
            
            # 3. Format timestamp 
            raw_time = item["snippet"]["publishedAt"]
            dt_obj = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
            video_details[video_id]["published_at"] = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

    return list(video_details.values())


# -----------------------------
# Fetch for all seed channels
# -----------------------------
def fetch_for_seed_channels(seed_df: pd.DataFrame, limit_per_channel: int = 5) -> pd.DataFrame:
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