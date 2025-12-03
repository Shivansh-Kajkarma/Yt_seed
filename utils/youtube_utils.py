import os
import time
import requests
import html
import re
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Counter
from utils.mongo_utils import check_quota_and_pause

load_dotenv()
# API_KEY = os.getenv("YOUTUBE_API_KEY")
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
        try:  # <-- Added try block for requests errors
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (429, 500, 502, 503, 504):
                wait = backoff * (2**attempt)
                print(
                    f"  Rate/Server error {resp.status_code}. Backing off for {wait:.1f}s (attempt {attempt + 1})"
                )
                time.sleep(wait)
                continue
            else:
                # non-retryable error
                print(f"  HTTP {resp.status_code} error from {url} : {resp.text}")
                resp.raise_for_status()  # Raise HTTPError for bad responses
        except requests.exceptions.RequestException as e:  # Catch requests errors
            print(f"  Network/Request error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise  # Re-raise after last attempt
            wait = backoff * (2**attempt)
            print(f"  Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue
        # If loop finishes without returning/raising (e.g., retries exhausted on HTTP errors)
        # raise Exception(f"Failed after {max_retries} attempts for URL: {url}") # Or re-raise last resp.raise_for_status() implicitly

    # If retries exhausted for retryable errors (429, 5xx)
    resp.raise_for_status()  # Raise the last error status encountered


def parse_iso8601_duration(duration: str) -> int:
    """
    Parses ISO 8601 duration (e.g., PT1H2M3S, PT45S) to seconds (int).
    Returns 0 if parse fails.
    """
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
     - title contains '#shorts' OR duration_seconds is <= 180
    """
    title_low = (title or "").lower()
    if "#shorts" in title_low or "shorts" in title_low.split():
        return True
    if duration_seconds is not None and duration_seconds <= 180:
        return True
    return False


def extract_channel_id(channel_url: str, api_key: str = None) -> str:
    """
    Resolve channel URL/handle/custom to a canonical channelId.

    Args:
        channel_url: YouTube channel URL, handle, or ID.
        api_key: YouTube API key (optional, falls back to global API_KEY).
    """
    if not channel_url:
        return ""

    # Use provided api_key or fall back to global
    key_to_use = api_key 
    if not key_to_use:
        print("❌ ERROR: No API key available for extract_channel_id.")
        return ""

    match = re.search(r"channel/(UC[a-zA-Z0-9_\-]+)", channel_url)
    if match:
        return match.group(1)

    seg = channel_url.rstrip("/").split("/")[-1]
    if seg.startswith("UC") and len(seg) >= 24:
        return seg

    handle = seg
    if handle.startswith("@"):
        handle = handle[1:]

    params = {
        "part": "snippet",
        "type": "channel",
        "q": handle,
        "key": key_to_use,
        "maxResults": 1,
    }
    url = f"{YT_BASE}/search"
    try:  # <-- Added try block for API call
        data = _safe_get_json(url, params)
        if "items" in data and data["items"]:
            return data["items"][0]["snippet"]["channelId"]
        else:
            print(
                f"⚠️ Could not resolve channel ID for '{channel_url}' (handle '{handle}'). No items found."
            )
            return ""
    except Exception as e:  # Catch errors from _safe_get_json
        print(
            f"⚠️ API Error resolving channel ID for '{channel_url}' (handle '{handle}'): {e}"
        )
        return ""


def fetch_recent_videos(
    channel_id: str,
    max_results: int = 30,
    filter_shorts: bool = True,
    max_items_to_scan: int = 500,
    min_videos_in_first_batch: int = 5,
    run_tag: str = "default",
    seed_name: str = "",
    api_key: str = None,
) -> Tuple[List[Dict], str]:
    """
    Fetch recent non-shorts videos and the channel description for a given channel.
    Includes quota-safe handling.

    Args:
        channel_id: YouTube channel ID.
        max_results: Max videos to fetch.
        filter_shorts: Whether to exclude shorts.
        max_items_to_scan: Total playlist items to check (safety break).
        min_videos_in_first_batch: If first page yields too few valid videos, skip channel.
        run_tag: Run ID for Mongo checkpoint logging.
        seed_name: Optional seed name.
        api_key: YouTube API key (optional, falls back to global API_KEY).

    Returns:
        Tuple: (List of video dictionaries, Channel description)
    """
    if not channel_id:
        return [], ""

    # Use provided api_key or fall back to global
    key_to_use = api_key 
    if not key_to_use:
        print("❌ ERROR: API key missing.")
        return [], ""

    results: List[Dict] = []
    total_items_scanned = 0
    is_first_batch = True
    channel_description = ""

    # 1️⃣ Fetch channel uploads playlist
    url = f"{YT_BASE}/channels"
    params = {"part": "contentDetails,snippet", "id": channel_id, "key": key_to_use}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            print(f"⚠️ No channel data for ID {channel_id}")
            return [], ""
        item = items[0]
        channel_description = item.get("snippet", {}).get("description", "")
        uploads_playlist = (
            item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        )
        if not uploads_playlist:
            print(f"⚠️ No uploads playlist for {channel_id}")
            return [], channel_description
    except Exception as e:
        # from utils.mongo_utils import check_quota_and_pause
        check_quota_and_pause(e, run_tag, seed_name)
        print(f"❌ Error fetching channel details for {channel_id}: {e}")
        return [], ""

    # 2️⃣ Loop through playlist pages
    next_page = None
    while len(results) < max_results:
        if total_items_scanned >= max_items_to_scan:
            print(f"⚠️ Hit scan limit ({max_items_to_scan} videos). Stopping.")
            break

        try:
            params_pl = {
                "part": "contentDetails",
                "playlistId": uploads_playlist,
                "maxResults": 50,
                "key": key_to_use,
            }
            if next_page:
                params_pl["pageToken"] = next_page

            resp = requests.get(
                f"{YT_BASE}/playlistItems", params=params_pl, timeout=30
            )
            resp.raise_for_status()
            page = resp.json()
            next_page = page.get("nextPageToken")

            video_ids = [
                it.get("contentDetails", {}).get("videoId")
                for it in page.get("items", [])
                if it.get("contentDetails", {}).get("videoId")
            ]
            total_items_scanned += len(video_ids)

            if not video_ids:
                if not next_page:
                    break
                continue

        except Exception as e:
            # from utils.mongo_utils import check_quota_and_pause
            check_quota_and_pause(e, run_tag, seed_name)
            print(f"❌ Playlist fetch error: {e}")
            break

        # 3️⃣ Fetch each video's metadata
        try:
            params_vid = {
                "part": "snippet,contentDetails",
                "id": ",".join(video_ids),
                "key": key_to_use,
            }
            resp_v = requests.get(f"{YT_BASE}/videos", params=params_vid, timeout=30)
            resp_v.raise_for_status()
            vids = resp_v.json().get("items", [])
        except Exception as e:
            # from utils.mongo_utils import check_quota_and_pause
            check_quota_and_pause(e, run_tag, seed_name)
            print(f"❌ Video details fetch error: {e}")
            break

        for item in vids:
            if len(results) >= max_results:
                break
            vid_id = item.get("id")
            snippet = item.get("snippet", {})
            desc = snippet.get("description", "") or ""
            title = html.unescape(snippet.get("title", ""))

            # --- THIS IS THE FIX ---
            published_at_iso = snippet.get("publishedAt", "")  # <-- FIX: GET THE DATE
            # --- END OF FIX ---

            duration = item.get("contentDetails", {}).get("duration", None)
            duration_seconds = parse_iso8601_duration(duration) if duration else 0
            if filter_shorts and is_short_video(title, duration_seconds):
                continue

            results.append(
                {
                    "video_id": vid_id,
                    "title": title,
                    "description": desc.replace("\n", " ").strip(),
                    "duration_seconds": duration_seconds,
                    "is_short": is_short_video(title, duration_seconds),
                    "published_at": published_at_iso,  # <-- FIX: ADD THE DATE
                }
            )

        if is_first_batch:
            is_first_batch = False
            if len(results) < min_videos_in_first_batch:
                print(f"⚠️ Skipping {channel_id}: too few long videos.")
                break

        if not next_page:
            break

    print(f"✅ Finished fetching for {channel_id}. Found {len(results)} valid videos.")
    return results[:max_results], channel_description


def fetch_for_seed_channels(
    seed_df,
    limit_per_channel: int = 30,
    filter_shorts: bool = True,
    run_tag: str = "default",
    api_key: str = None,
) -> "pd.DataFrame":
    """Fetches videos and channel descriptions for seed channels

    Args:
        seed_df: DataFrame with Channel_Name and Channel_URL columns.
        limit_per_channel: Max videos per channel.
        filter_shorts: Whether to exclude shorts.
        run_tag: Run ID for logging.
        api_key: YouTube API key (optional, falls back to global API_KEY).
    """
    import pandas as pd  # Import here as it's only used here

    all_records = []
    for _, row in seed_df.iterrows():
        channel_url = row.get("Channel_URL", "")
        channel_name = row.get("Channel_Name", "")
        print(
            f"\n🎯 Fetching for seed: {channel_name} ({channel_url or 'URL Missing'})"
        )
        if not channel_url:
            print("   Skipping due to missing URL.")
            continue

        channel_id = extract_channel_id(channel_url, api_key=api_key)
        if not channel_id:
            print(f"   Skipping {channel_name} (could not resolve ID).")
            continue

        videos, channel_desc = fetch_recent_videos(
            channel_id,
            max_results=limit_per_channel,
            filter_shorts=filter_shorts,
            run_tag=run_tag,
            api_key=api_key,
        )

        if videos:  # Only add if videos were found
            for v in videos:
                all_records.append(
                    {
                        "Channel_Name": channel_name,
                        "Channel_ID": channel_id,
                        "channel_description": channel_desc,
                        "video_id": v["video_id"],
                        "title": v["title"],
                        "description": v["description"],
                        "published_at": v["published_at"],
                        "duration_seconds": v["duration_seconds"],
                        "is_short": v["is_short"],
                    }
                )
        else:
            print(f"   No videos fetched for {channel_name}")

    if not all_records:
        print("\n⚠️ No videos fetched for any seed channel.")
        return pd.DataFrame()  # Return empty DataFrame

    df_videos = pd.DataFrame(all_records)
    print(
        f"\n✅ Fetched {len(df_videos)} total videos across {seed_df['Channel_Name'].nunique()} seed channels."
    )
    return df_videos


def get_channel_metadata_batch(
    channel_ids: List[str],
    run_tag: str = "default",
    seed_name: str = "",
    api_key: str = None,
) -> List[Dict]:
    """
    Fetches channel metadata (title, description, subs, country, etc.) for multiple YouTube channels.

    Args:
        channel_ids: List of YouTube channel IDs.
        run_tag: Current pipeline run ID (for Mongo checkpoint).
        seed_name: Optional seed name for clear run tracking.
        api_key: YouTube API key (optional, falls back to global API_KEY).

    Returns:
        A list of dictionaries containing channel metadata.
    """
    # Use provided api_key or fall back to global
    key_to_use = api_key 
    if not key_to_use:
        print("❌ ERROR: API key missing.")
        return []
    if not channel_ids:
        return []

    print(f"  Fetching metadata for {len(channel_ids)} channels...")
    channel_data = []
    processed_count = 0
    url = f"{YT_BASE}/channels"

    for i in range(0, len(channel_ids), 50):
        batch_ids = channel_ids[i : i + 50]
        batch_num = (i // 50) + 1
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch_ids),
            "maxResults": 50,
            "key": key_to_use,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            processed_count += len(items)

            for item in items:
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                channel_id = item.get("id")
                channel_name = snippet.get("title", "")
                custom_url_handle = snippet.get("customUrl")
                channel_url = (
                    f"https://www.youtube.com/{custom_url_handle}"
                    if custom_url_handle and custom_url_handle.startswith("@")
                    else f"https://www.youtube.com/channel/{channel_id}"
                )

                channel_data.append(
                    {
                        "id": channel_id,
                        "name": channel_name,
                        "url": channel_url,
                        "description": snippet.get("description", ""),
                        "subscribers": int(stats.get("subscriberCount", 0))
                        if not stats.get("hiddenSubscriberCount", False)
                        else -1,
                        "video_count": int(stats.get("videoCount", 0)),
                        "country": snippet.get("country", "Unknown"),
                    }
                )

        except Exception as e:
            # from utils.mongo_utils import check_quota_and_pause
            check_quota_and_pause(e, run_tag, seed_name)
            print(f"   ❌ ERROR fetching metadata batch {batch_num}: {e}")
            continue

    print(
        f"  ✅ Finished fetching metadata. Got details for {processed_count} channels."
    )
    return channel_data


def search_videos_multi_focused(
    keywords: List[str],
    max_results_per_search: int = 10,
    max_keywords: int = 7,
    run_tag: str = "default",
    seed_name: str = "",
    api_key: str = None,
) -> set[str]:
    """
    Performs multiple focused YouTube searches (biased to English) using VIDEO search type.
    Extracts **unique channel IDs** from the returned video results.

    Args:
        keywords: List of search phrases/keywords.
        max_results_per_search: Number of video results per keyword.
        max_keywords: Limit how many keywords to use per run.
        run_tag: Used for logging and Mongo checkpoint (if quota hits).
        seed_name: Optional seed channel name (for clearer logging in Mongo).
        api_key: YouTube API key (optional, falls back to global API_KEY).

    Returns:
        A set of **unique channel IDs** discovered across all keyword searches.
    """
    # Use provided api_key or fall back to global
    key_to_use = api_key 
    if not key_to_use:
        print("❌ ERROR: API key missing.")
        return set()
    if not keywords:
        print("⚠️ WARNING: No keywords provided.")
        return set()

    keywords_to_search = min(len(keywords), max_keywords)
    all_candidate_channel_ids = set()

    print(
        f"  🔎 Performing {keywords_to_search} focused VIDEO searches (biased to English)..."
    )

    for i, keyword in enumerate(keywords[:keywords_to_search]):
        search_query = keyword
        print(f"     Search {i + 1}/{keywords_to_search}: '{search_query}'")

        url = f"{YT_BASE}/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "order": "relevance",
            "maxResults": max_results_per_search,
            "key": key_to_use,
            "relevanceLanguage": "en",
            "regionCode": "US",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            found_channels_in_batch = set()

            for item in items:
                if (
                    item.get("id", {}).get("kind") == "youtube#video"
                    and "snippet" in item
                ):
                    ch_id = item.get("snippet", {}).get("channelId")
                    if ch_id:
                        all_candidate_channel_ids.add(ch_id)
                        found_channels_in_batch.add(ch_id)

            print(
                f"        ✅ Found videos from {len(found_channels_in_batch)} unique channels"
            )

        except Exception as e:
            # from utils.mongo_utils import check_quota_and_pause
            check_quota_and_pause(e, run_tag, seed_name)
            print(f"        ❌ Search Error: {str(e)[:100]}")
            continue

    print(
        f"  📊 Total: {len(all_candidate_channel_ids)} unique candidate channels found.\n"
    )
    return all_candidate_channel_ids


def frequency_search_by_titles(
    titles: List[str],
    seed_channel_id: str,
    max_results_per_title: int = 20,
    api_key: str = None,
) -> Tuple[Counter, Dict[str, Dict]]:
    """
    Client's approach: Search each title, count channel frequency

    Args:
        titles: List of video titles to search
        seed_channel_id: ID of seed channel (to exclude from results)
        max_results_per_title: How many results per title search
        api_key: YouTube API key (optional, falls back to global API_KEY).

    Returns:
        (channel_frequency_counter, channel_metadata_dict)
    """
    # Use provided api_key or fall back to global
    key_to_use = api_key 
    if not key_to_use:
        print("❌ ERROR: YouTube API key not found.")
        return Counter(), {}

    print(f"\n🔍 Starting frequency search for {len(titles)} titles...")

    channel_frequency = Counter()
    channel_metadata = {}  # Store metadata for each channel
    search_url = f"{YT_BASE}/search"

    for i, title in enumerate(titles, 1):
        print(f"\n[{i}/{len(titles)}] Searching: '{title[:60]}...'")

        try:
            params = {
                "part": "snippet",
                "q": title,
                "type": "video",
                "maxResults": max_results_per_title,
                # "regionCode": "US",          # English content
                "relevanceLanguage": "en",  # English preference
                "key": key_to_use,
            }

            data = _safe_get_json(search_url, params)
            items = data.get("items", [])

            found_channels = []
            for item in items:
                channel_id = item["snippet"]["channelId"]
                channel_name = item["snippet"]["channelTitle"]

                # Skip seed channel itself
                if channel_id == seed_channel_id:
                    continue

                # Increment frequency
                channel_frequency[channel_id] += 1

                # Store metadata (first occurrence)
                if channel_id not in channel_metadata:
                    channel_metadata[channel_id] = {
                        "id": channel_id,
                        "name": channel_name,
                        "first_seen_in": title[:60],
                    }

                found_channels.append(channel_name)

            print(f"   Found {len(found_channels)} channels")
            if found_channels[:3]:
                print(f"   Top 3: {', '.join(found_channels[:3])}")

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"   ❌ Error searching '{title[:50]}': {e}")
            continue

    print(f"\n✅ Frequency search complete!")
    print(f"   Total unique channels: {len(channel_frequency)}")

    return channel_frequency, channel_metadata


def enrich_channel_metadata_batch(
    channel_ids: List[str],
    existing_metadata: Dict[str, Dict] = None,
    api_key: str = None,
) -> Dict[str, Dict]:
    """
    Enrich channel metadata with full details (subs, videos, etc.)
    Uses batch API calls for efficiency

    Args:
        channel_ids: List of channel IDs to enrich
        existing_metadata: Existing partial metadata dict (optional)
        api_key: YouTube API key (optional, falls back to global API_KEY).

    Returns:
        Dict of channel_id -> full_metadata
    """
    # Use provided api_key or fall back to global
    key_to_use = api_key 
    if not key_to_use:
        print("❌ ERROR: API key missing")
        return {}

    if not channel_ids:
        return {}

    print(f"\n📦 Enriching metadata for {len(channel_ids)} channels...")

    enriched = existing_metadata.copy() if existing_metadata else {}
    channels_url = f"{YT_BASE}/channels"

    # Batch in groups of 50
    for i in range(0, len(channel_ids), 50):
        batch_ids = channel_ids[i : i + 50]
        batch_num = (i // 50) + 1

        try:
            params = {
                "part": "snippet,statistics",
                "id": ",".join(batch_ids),
                "key": key_to_use,
            }

            data = _safe_get_json(channels_url, params)
            items = data.get("items", [])

            print(f"   Batch {batch_num}: Enriched {len(items)} channels")

            for item in items:
                channel_id = item["id"]
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})

                # Merge with existing metadata
                if channel_id in enriched:
                    enriched[channel_id].update(
                        {
                            "name": snippet.get("title"),
                            "description": snippet.get("description", ""),
                            "country": snippet.get("country", "unknown"),
                            "subscribers": int(stats.get("subscriberCount", 0)),
                            "video_count": int(stats.get("videoCount", 0)),
                            "view_count": int(stats.get("viewCount", 0)),
                            "custom_url": snippet.get("customUrl", ""),
                        }
                    )
                else:
                    enriched[channel_id] = {
                        "id": channel_id,
                        "name": snippet.get("title"),
                        "description": snippet.get("description", ""),
                        "country": snippet.get("country", "unknown"),
                        "subscribers": int(stats.get("subscriberCount", 0)),
                        "video_count": int(stats.get("videoCount", 0)),
                        "view_count": int(stats.get("viewCount", 0)),
                        "custom_url": snippet.get("customUrl", ""),
                    }

            time.sleep(0.3)  # Rate limit

        except Exception as e:
            print(f"   ❌ Error enriching batch {batch_num}: {e}")
            continue

    print(f"✅ Enrichment complete!")
    return enriched


def filter_by_frequency_threshold(
    channel_frequency: Counter,
    channel_metadata: Dict[str, Dict],
    min_frequency: int,
    total_searches: int,
) -> List[Dict]:
    """
    Filter channels by frequency threshold and prepare results

    Args:
        channel_frequency: Counter of channel appearances
        channel_metadata: Dict of channel metadata
        min_frequency: Minimum appearances to be considered
        total_searches: Total number of searches performed

    Returns:
        List of candidate channels with scores
    """
    print(f"\n📊 Filtering by frequency threshold: {min_frequency}/{total_searches}")

    candidates = []

    for channel_id, frequency in channel_frequency.items():
        if frequency >= min_frequency:
            metadata = channel_metadata.get(channel_id, {})

            frequency_score = frequency / total_searches

            candidates.append(
                {
                    "channel_id": channel_id,
                    "channel_name": metadata.get("name", "Unknown"),
                    "frequency": frequency,
                    "frequency_score": frequency_score,
                    "appearances": f"{frequency}/{total_searches}",
                    "subscribers": metadata.get("subscribers", 0),
                    "video_count": metadata.get("video_count", 0),
                    "country": metadata.get("country", "unknown"),
                    "first_seen_in": metadata.get("first_seen_in", ""),
                }
            )

    # Sort by frequency (highest first)
    candidates.sort(key=lambda x: x["frequency"], reverse=True)

    print(f"✅ Found {len(candidates)} candidates above threshold")

    return candidates


def _load_from_google_sheet(sheet_url: str) -> Optional[pd.DataFrame]:
    """
    Internal function to load seed channels from a PUBLIC Google Sheet URL.

    Note: The Google Sheet must be "Published to the web" as a CSV.
    (File -> Share -> Publish to web -> Select sheet -> Select CSV)

    Args:
        sheet_url: The public URL of the Google Sheet (must be a CSV export link).

    Returns:
        A DataFrame with 'Channel_Name' and 'Channel_URL', or None if loading fails.
    """
    print(f"Loading Google Sheet from URL: {sheet_url}")
    try:
        # Check if it's a direct CSV export link (from "publish" or "export")
        if "output=csv" in sheet_url or "export?format=csv" in sheet_url:
            csv_export_url = sheet_url
            print(f"Using direct CSV URL: {csv_export_url}")
        else:
            # Try to construct the export link from a standard /edit URL
            match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", sheet_url)
            if not match:
                print(
                    "Error: Invalid Google Sheet URL. Must be a standard /edit link or a public 'output=csv' link."
                )
                return None

            sheet_id = match.group(1)
            gid_match = re.search(r"gid=([0-9]+)", sheet_url)
            gid = gid_match.group(1) if gid_match else "0"

            csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            print(f"Constructed CSV export URL: {csv_export_url}")

        df = pd.read_csv(csv_export_url, encoding="utf-8-sig")

        # print(f"[Debug] Columns found by pandas: {list(df.columns)}")

        # Sanitize column names
        df.columns = df.columns.str.strip()

        if "Channel_Name" not in df.columns or "Channel_URL" not in df.columns:
            print(
                "Error: Google Sheet must contain 'Channel_Name' and 'Channel_URL' columns."
            )
            print(f"[Debug] Sanitized columns: {list(df.columns)}")  # More debug
            return None

        print(f"Successfully loaded {len(df)} channels from Google Sheet.")
        return df[["Channel_Name", "Channel_URL"]]

    except Exception as e:
        print(f"Error loading Google Sheet from '{sheet_url}': {e}")
        print(
            "Please ensure the URL is correct and the sheet is 'Published to the web' as a CSV."
        )
        return None
