import os
import time
import requests
import html
import re
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Dict, Optional

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
            wait = backoff * (2 ** attempt)
            print(f"Rate/Server error {resp.status_code}. Backing off for {wait:.1f}s (attempt {attempt+1})")
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
        m = re.match(r'P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
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
      1. If URL contains /channel/ID → return ID
      2. If URL contains full ID pattern as last path segment → return
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
        "maxResults": 1
    }
    url = f"{YT_BASE}/search"
    data = _safe_get_json(url, params)
    if "items" in data and data["items"]:
        return data["items"][0]["snippet"]["channelId"]
    else:
        print(f"⚠️ Could not resolve channel ID for '{channel_url}' (handle '{handle}').")
        return ""


# -----------------------------
# Fetch recent videos using uploads playlist + pagination
# -----------------------------
def fetch_recent_videos(channel_id: str, max_results: int = 30, filter_shorts: bool = True) -> List[Dict]:
    """
    Fetch up to max_results recent videos for a channel using the channel's uploads playlist.
    Returns list of dicts with keys:
      video_id, title, description, published_at (YYYY-mm-dd HH:MM:SS), duration_seconds, is_short
    """
    if not channel_id:
        return []

    # 1) get uploads playlist id
    url = f"{YT_BASE}/channels"
    params = {"part": "contentDetails", "id": channel_id, "key": API_KEY}
    data = _safe_get_json(url, params)
    items = data.get("items", [])
    if not items:
        print(f"⚠️ No channel data for {channel_id}")
        return []

    uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"].get("uploads")
    if not uploads_playlist:
        print(f"⚠️ No uploads playlist for channel {channel_id}")
        return []

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
        return []

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
            duration_seconds = parse_iso8601_duration(duration_iso) if duration_iso else None

            clean_desc = html.unescape(raw_desc).replace("\n", " ").replace("\r", " ").strip()

            # detect short
            short_flag = is_short_video(raw_title, duration_seconds)

            if filter_shorts and short_flag:
                # skip shorts from results if requested
                continue

            # format published time to local (UTC) string
            if raw_published:
                try:
                    dt_obj = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
                    published_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    published_at = raw_published
            else:
                published_at = ""

            results.append({
                "video_id": vid,
                "title": html.unescape(raw_title),
                "description": clean_desc,
                "published_at": published_at,
                "duration_seconds": duration_seconds if duration_seconds is not None else 0,
                "is_short": short_flag
            })

    # Keep order consistent with collected_video_ids (newest first)
    # Build map and reorder
    vid_map = {r["video_id"]: r for r in results}
    ordered = [vid_map[v] for v in collected_video_ids if v in vid_map]

    print(f"Fetched {len(ordered)} videos (requested {max_results}) for channel {channel_id}")
    return ordered[:max_results]


# -----------------------------
# Top-level batch fetch
# -----------------------------
def fetch_for_seed_channels(seed_df, limit_per_channel: int = 30, filter_shorts: bool = True) -> 'pd.DataFrame':
    """
    Given a seed DataFrame with 'Channel_Name' and 'Channel_URL' columns,
    fetch recent videos for each channel (limit_per_channel each).
    Returns a pandas DataFrame of all videos fetched.
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

        videos = fetch_recent_videos(channel_id, max_results=limit_per_channel, filter_shorts=filter_shorts)
        if not videos:
            print(f"No videos returned for {channel_name}")
            continue

        for v in videos:
            all_records.append({
                "Channel_Name": channel_name,
                "Channel_ID": channel_id,
                "video_id": v["video_id"],
                "title": v["title"],
                "description": v["description"],
                "published_at": v["published_at"],
                "duration_seconds": v["duration_seconds"],
                "is_short": v["is_short"]
            })

    df_videos = pd.DataFrame(all_records)
    print(f"\n✅ Fetched {len(df_videos)} total videos across {len(seed_df)} seed channels.")
    return df_videos
