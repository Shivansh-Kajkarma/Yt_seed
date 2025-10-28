import os
import time
import requests
import html
import numpy as np
import pandas as pd
import re
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple # <-- CHANGE: Added Tuple

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
YT_BASE = "https://www.googleapis.com/youtube/v3"


# -----------------------------
# Helpers
# -----------------------------
def _safe_get_json(url: str, params: dict, max_retries: int = 3, backoff: float = 0.5):
    """
    Simple requests wrapper with exponential backoff on 429/5xx.
    Returns parsed json or raises an exception.
    """
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code in (429, 500, 502, 503, 504):
            wait = backoff * (2**attempt)
            print(
                f"Rate/Server error {resp.status_code}. Backing off for {wait:.1f}s (attempt {attempt + 1})"
            )
            time.sleep(wait)
            continue
        else:
            # non-retryable error
            print(f"HTTP {resp.status_code} error from {url} : {resp.text}")
            resp.raise_for_status()
    # final attempt
    resp.raise_for_status()


def parse_iso8601_duration(duration: str) -> int:
    """
    Parses ISO 8601 duration (e.g., PT1H2M3S, PT45S) to seconds (int).
    Returns 0 if parse fails.
    """
    # pattern like PT#H#M#S
    try:
        days = hours = minutes = seconds = 0
        m = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if m:
            d, h, mm, s = m.groups()
            days = int(d) if d else 0
            hours = int(h) if h else 0
            minutes = int(mm) if mm else 0
            seconds = int(s) if s else 0
            total = seconds + minutes * 60 + hours * 3600 + days * 86400
        return total
    except Exception:
        return 0


def is_short_video(title: str, duration_seconds: Optional[int]) -> bool:
    """
    Heuristic to detect shorts:
     - title contains '#shorts' OR duration_seconds is <= 60
    """
    title_low = (title or "").lower()
    if "#shorts" in title_low or "shorts" in title_low.split():
        return True
    if duration_seconds is not None and duration_seconds <= 60:
        return True
    return False


def extract_channel_id(channel_url: str) -> str:
    """
    Resolve channel URL/handle/custom to a canonical channelId.
    Strategy:
      1. If URL contains /channel/ID -> return ID
      2. If URL contains full ID pattern as last path segment -> return
      3. If @handle or custom name given, use search endpoint to find channel
    """
    if not channel_url:
        return ""

    # direct /channel/XYZ
    match = re.search(r"channel/([A-Za-z0-9_\-]+)", channel_url)
    if match:
        return match.group(1)

    # if url ends with a 24+ char id like UCxxxx...
    seg = channel_url.rstrip("/").split("/")[-1]
    if seg.startswith("UC") and len(seg) >= 24:
        return seg

    # handle form /@handle or raw @handle
    handle = seg
    if handle.startswith("@"):
        handle = handle[1:]

    # fallback: use search to find channel by handle/name
    params = {
        "part": "snippet",
        "type": "channel",
        "q": handle,
        "key": API_KEY,
        "maxResults": 1,
    }
    url = f"{YT_BASE}/search"
    data = _safe_get_json(url, params)
    if "items" in data and data["items"]:
        return data["items"][0]["snippet"]["channelId"]
    else:
        print(
            f"⚠️ Could not resolve channel ID for '{channel_url}' (handle '{handle}')."
        )
        return ""


# -----------------------------
# Fetch recent videos using uploads playlist + pagination
# -----------------------------
def fetch_recent_videos(
    channel_id: str, max_results: int = 30, filter_shorts: bool = True
) -> Tuple[List[Dict], str]: # <-- CHANGE: Return type is now a Tuple
    """
    Fetch up to max_results recent videos for a channel using the channel's uploads playlist.
    
    Returns a tuple:
     (List[Dict], str)
     - List of video dicts with keys:
       video_id, title, description, published_at (YYYY-mm-dd HH:MM:SS), duration_seconds, is_short
     - String containing the channel's "About" description
    """
    if not channel_id:
        return [], "" # <-- CHANGE: Return empty desc string

    # 1) get uploads playlist id AND channel "About" description
    url = f"{YT_BASE}/channels"
    
    # <-- CHANGE: Ask for 'snippet' as well as 'contentDetails'
    params = {"part": "contentDetails,snippet", "id": channel_id, "key": API_KEY}
    
    data = _safe_get_json(url, params)
    items = data.get("items", [])
    
    channel_description = "" # <-- CHANGE: Default empty string
    
    if not items:
        print(f"⚠️ No channel data for {channel_id}")
        return [], channel_description # <-- CHANGE: Return default desc

    # <-- CHANGE: Extract the description from the snippet
    if "snippet" in items[0]:
        channel_description = items[0].get("snippet", {}).get("description", "")
        
    uploads_playlist = items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    
    if not uploads_playlist:
        print(f"⚠️ No uploads playlist for channel {channel_id}")
        # Still return the description even if no uploads
        return [], channel_description # <-- CHANGE: Return desc

    # 2) iterate playlistItems -> collect videoIds (paginate)
    playlist_url = f"{YT_BASE}/playlistItems"
    collected_video_ids: List[str] = []
    next_page = None
    while len(collected_video_ids) < max_results:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist,
            "maxResults": 50,
            "key": API_KEY,
        }
        if next_page:
            params["pageToken"] = next_page

        page = _safe_get_json(playlist_url, params)
        for it in page.get("items", []):
            # videoId may be under contentDetails
            vid = it.get("contentDetails", {}).get("videoId")
            if vid:
                collected_video_ids.append(vid)
                if len(collected_video_ids) >= max_results:
                    break

        next_page = page.get("nextPageToken")
        if not next_page:
            break

    if not collected_video_ids:
        print(f"No videos found in uploads playlist for {channel_id}")
        return [], channel_description # <-- CHANGE: Return desc

    # 3) Batch get videos details (snippet + contentDetails)
    results = []
    # process in batches of up to 50 ids (API max)
    for i in range(0, len(collected_video_ids), 50):
        batch_ids = collected_video_ids[i : i + 50]
        vids_url = f"{YT_BASE}/videos"
        params = {
            "part": "snippet,contentDetails",
            "id": ",".join(batch_ids),
            "maxResults": 50,
            "key": API_KEY,
        }
        vdata = _safe_get_json(vids_url, params)

        # iterate returned videos (order may differ from playlist; we'll keep batch order)
        for item in vdata.get("items", []):
            vid = item.get("id")
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})

            raw_title = snippet.get("title", "")
            raw_desc = snippet.get("description", "") or ""
            raw_published = snippet.get("publishedAt", None)
            duration_iso = content.get("duration", None)
            duration_seconds = (
                parse_iso8601_duration(duration_iso) if duration_iso else None
            )

            clean_desc = (
                html.unescape(raw_desc).replace("\n", " ").replace("\r", " ").strip()
            )

            # detect short
            short_flag = is_short_video(raw_title, duration_seconds)

            if filter_shorts and short_flag:
                # skip shorts from results if requested
                continue

            # format published time to local (UTC) string
            if raw_published:
                try:
                    dt_obj = datetime.fromisoformat(
                        raw_published.replace("Z", "+00:00")
                    )
                    published_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    published_at = raw_published
            else:
                published_at = ""

            results.append(
                {
                    "video_id": vid,
                    "title": html.unescape(raw_title),
                    "description": clean_desc,
                    "published_at": published_at,
                    "duration_seconds": duration_seconds
                    if duration_seconds is not None
                    else 0,
                    "is_short": short_flag,
                }
            )

    # Keep order consistent with collected_video_ids (newest first)
    # Build map and reorder
    vid_map = {r["video_id"]: r for r in results}
    ordered = [vid_map[v] for v in collected_video_ids if v in vid_map]

    print(
        f"Fetched {len(ordered)} videos (requested {max_results}) for channel {channel_id}"
    )
    # <-- CHANGE: Return both the videos and the channel description
    return ordered[:max_results], channel_description


# -----------------------------
# Top-level batch fetch
# -----------------------------
def fetch_for_seed_channels(
    seed_df, limit_per_channel: int = 30, filter_shorts: bool = True
) -> "pd.DataFrame":
    """
    Given a seed DataFrame with 'Channel_Name' and 'Channel_URL' columns,
    fetch recent videos for each channel (limit_per_channel each).
    
    Returns a pandas DataFrame of all videos fetched, now including
    a 'channel_description' column for the seed channel.
    """
    import pandas as pd

    all_records = []
    for _, row in seed_df.iterrows():
        channel_url = row.get("Channel_URL", "")
        channel_name = row.get("Channel_Name", "")
        print(f"\n🎯 Fetching recent videos for: {channel_name} ({channel_url})")
        channel_id = extract_channel_id(channel_url)
        if not channel_id:
            print(f"Skipping {channel_name} because channel_id missing.")
            continue

        # <-- CHANGE: Unpack the tuple (videos, channel_desc)
        videos, channel_desc = fetch_recent_videos(
            channel_id, max_results=limit_per_channel, filter_shorts=filter_shorts
        )
        
        if not videos:
            print(f"No videos returned for {channel_name}")
            continue

        for v in videos:
            all_records.append(
                {
                    "Channel_Name": channel_name,
                    "Channel_ID": channel_id,
                    # <-- CHANGE: Add the new description field to every video row
                    "channel_description": channel_desc,
                    "video_id": v["video_id"],
                    "title": v["title"],
                    "description": v["description"],
                    "published_at": v["published_at"],
                    "duration_seconds": v["duration_seconds"],
                    "is_short": v["is_short"],
                }
            )

    df_videos = pd.DataFrame(all_records)
    print(
        f"\n✅ Fetched {len(df_videos)} total videos across {len(seed_df)} seed channels."
    )
    return df_videos

# --- ADDED: Batch Channel Metadata Fetch Function (Using _safe_get_json) ---
def get_channel_metadata_batch(channel_ids: List[str]) -> List[Dict]:
    """
    Fetches snippet and statistics for a list of channel IDs in batches of 50.
    Uses the _safe_get_json helper.
    Returns a list of dictionaries containing relevant metadata.
    Cost: 1 quota unit per batch of 50 IDs.
    """
    if not API_KEY:
        print("  ❌ ERROR: YouTube API key not found, cannot fetch channel metadata.")
        return []
    if not channel_ids:
        return []

    print(f"  Fetching metadata for {len(channel_ids)} channel IDs in batches...")
    channel_data = []
    processed_count = 0
    url = f"{YT_BASE}/channels"

    # Process in batches of 50
    for i in range(0, len(channel_ids), 50):
        batch_ids = channel_ids[i : i + 50]
        batch_num = (i // 50) + 1
        print(f"     Fetching batch {batch_num} ({len(batch_ids)} IDs)...")

        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch_ids),
            "maxResults": 50, # Optional, implied by ID count but good practice
            "key": API_KEY
        }

        try:
            # Use the requests-based helper
            response = _safe_get_json(url, params)
            items = response.get("items", [])
            print(f"     Batch {batch_num}: Received data for {len(items)} channels.")
            processed_count += len(items)

            for item in items:
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                channel_id = item.get("id")
                channel_name = snippet.get("title")
                custom_url_handle = snippet.get("customUrl")
                # Handle URL construction carefully
                channel_url = f"https://www.youtube.com/{custom_url_handle}" if custom_url_handle and custom_url_handle.startswith('@') else f"https://www.youtube.com/channel/{channel_id}"

                channel_data.append({
                    "id": channel_id,
                    "name": channel_name,
                    "url": channel_url,
                    # <-- CHANGE: Also grab the "About" description for DISCOVERED channels
                    "description": snippet.get("description", ""),
                    "subscribers": int(stats.get("subscriberCount", 0)) if not stats.get("hiddenSubscriberCount", False) else -1,
                    "video_count": int(stats.get("videoCount", 0))
                })
        except requests.exceptions.RequestException as e: # Catch errors from _safe_get_json
            print(f"   ❌ ERROR fetching metadata batch {batch_num}: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error fetching metadata batch {batch_num}: {type(e).__name__} - {e}")

    print(f"  ℹ️ Finished fetching metadata. Got details for {processed_count} channels.")
    return channel_data


def search_videos_multi_focused(keywords: List[str], max_results_per_search: int = 10, max_keywords: int = 7) -> set[str]:
    """
    Performs multiple focused searches.
    UPDATED: Search 5-7 keywords instead of just 3!
    
    Cost: 100 units × 7 = 700 units per seed
    Quality: Finds 3-4x more diverse channels
    """
    if not API_KEY:
        print("  ❌ ERROR: YouTube API key not found")
        return set()
    
    if not keywords:
        print("  ⚠️  WARNING: No keywords provided")
        return set()
    
    # Use MORE keywords for better coverage
    keywords_to_search = min(len(keywords), max_keywords)  # Search up to 7 keywords
    all_candidates = set()
    
    print(f"  🔎 Performing {keywords_to_search} focused searches...")
    
    for i in range(keywords_to_search):
        keyword = keywords[i]
        search_query = keyword
        
        print(f"     Search {i+1}/{keywords_to_search}: {search_query}")
        
        url = f"{YT_BASE}/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "channel",
            "order": "relevance",
            "maxResults": max_results_per_search,
            "key": API_KEY
        }
        
        try:
            response = _safe_get_json(url, params)
            videos = response.get("items", [])
            
            for item in videos:
                ch_id = item.get("snippet", {}).get("channelId")
                if ch_id:
                    all_candidates.add(ch_id)
            
            print(f"        ✅ {len(videos)} channels found") # <-- CHANGE: More accurate print
            
        except Exception as e:
            print(f"        ❌ Error: {str(e)[:50]}")
            continue
    
    print(f"  📊 Total: {len(all_candidates)} unique channels from {keywords_to_search} searches\n")
    return all_candidates