import os
import time
import requests
import html
import re
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Counter

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
        try: # <-- Added try block for requests errors
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
                resp.raise_for_status() # Raise HTTPError for bad responses
        except requests.exceptions.RequestException as e: # Catch requests errors
             print(f"  Network/Request error on attempt {attempt + 1}: {e}")
             if attempt == max_retries - 1:
                 raise # Re-raise after last attempt
             wait = backoff * (2**attempt)
             print(f"  Retrying in {wait:.1f}s...")
             time.sleep(wait)
             continue
        # If loop finishes without returning/raising (e.g., retries exhausted on HTTP errors)
        # raise Exception(f"Failed after {max_retries} attempts for URL: {url}") # Or re-raise last resp.raise_for_status() implicitly

    # If retries exhausted for retryable errors (429, 5xx)
    resp.raise_for_status() # Raise the last error status encountered


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


def extract_channel_id(channel_url: str) -> str:
    """
    Resolve channel URL/handle/custom to a canonical channelId.
    """
    if not channel_url:
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
        "key": API_KEY,
        "maxResults": 1,
    }
    url = f"{YT_BASE}/search"
    try: # <-- Added try block for API call
        data = _safe_get_json(url, params)
        if "items" in data and data["items"]:
            return data["items"][0]["snippet"]["channelId"]
        else:
            print(
                f"⚠️ Could not resolve channel ID for '{channel_url}' (handle '{handle}'). No items found."
            )
            return ""
    except Exception as e: # Catch errors from _safe_get_json
        print(
            f"⚠️ API Error resolving channel ID for '{channel_url}' (handle '{handle}'): {e}"
        )
        return ""


# -----------------------------
# Fetch recent videos using uploads playlist + pagination
# -----------------------------
# def fetch_recent_videos(
#     channel_id: str, max_results: int = 30, filter_shorts: bool = True
# ) -> Tuple[List[Dict], str]:
#     """
#     Fetch up to max_results recent videos and channel description.
#     """
#     if not channel_id: return [], ""
#     if not API_KEY:
#         print("❌ ERROR: YouTube API key not found. Cannot fetch videos.")
#         return [], ""

#     channel_description = ""
#     uploads_playlist = None

#     # 1) Get channel details (snippet + contentDetails)
#     url = f"{YT_BASE}/channels"
#     params = {"part": "contentDetails,snippet", "id": channel_id, "key": API_KEY}
#     try:
#         data = _safe_get_json(url, params)
#         items = data.get("items", [])
#         if not items:
#             print(f"⚠️ No channel data found for ID {channel_id}")
#             return [], ""

#         item = items[0]
#         channel_description = item.get("snippet", {}).get("description", "")
#         uploads_playlist = item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")

#         if not uploads_playlist:
#             print(f"⚠️ No uploads playlist found for channel {channel_id}")
#             # Still return description if found
#             return [], channel_description

#     except Exception as e:
#         print(f"❌ Error fetching channel details for {channel_id}: {e}")
#         return [], ""

#     # 2) Iterate playlistItems to collect videoIds
#     playlist_url = f"{YT_BASE}/playlistItems"
#     collected_video_ids: List[str] = []
#     next_page = None
#     try:
#         while len(collected_video_ids) < max_results:
#             params = {
#                 "part": "contentDetails", # Only need contentDetails here
#                 "playlistId": uploads_playlist,
#                 "maxResults": min(50, max_results - len(collected_video_ids)), # Fetch needed amount up to 50
#                 "key": API_KEY,
#             }
#             if next_page:
#                 params["pageToken"] = next_page

#             page = _safe_get_json(playlist_url, params)
#             page_ids = [it.get("contentDetails", {}).get("videoId") for it in page.get("items", []) if it.get("contentDetails", {}).get("videoId")]
#             collected_video_ids.extend(page_ids)

#             next_page = page.get("nextPageToken")
#             if not next_page: break

#     except Exception as e:
#         print(f"❌ Error fetching playlist items for {channel_id}: {e}")
#         # Return what we have collected so far + description
#         return [], channel_description

#     if not collected_video_ids:
#         print(f"  No videos found in uploads playlist for {channel_id}")
#         return [], channel_description

#     # Ensure we only fetch up to max_results
#     collected_video_ids = collected_video_ids[:max_results]

#     # 3) Batch get details for collected videoIds
#     results = []
#     vids_url = f"{YT_BASE}/videos"
#     try:
#         # Process in batches of up to 50
#         for i in range(0, len(collected_video_ids), 50):
#             batch_ids = collected_video_ids[i : i + 50]
#             params = {
#                 "part": "snippet,contentDetails",
#                 "id": ",".join(batch_ids),
#                 "maxResults": 50,
#                 "key": API_KEY,
#             }
#             vdata = _safe_get_json(vids_url, params)

#             for item in vdata.get("items", []):
#                 vid = item.get("id")
#                 snippet = item.get("snippet", {})
#                 content = item.get("contentDetails", {})

#                 raw_title = snippet.get("title", "")
#                 raw_desc = snippet.get("description", "") or ""
#                 raw_published = snippet.get("publishedAt", None)
#                 duration_iso = content.get("duration", None)
#                 duration_seconds = (
#                     parse_iso8601_duration(duration_iso) if duration_iso else None
#                 )
#                 short_flag = is_short_video(raw_title, duration_seconds)

#                 if filter_shorts and short_flag: continue

#                 clean_desc = html.unescape(raw_desc).replace("\n", " ").replace("\r", " ").strip()
#                 published_at = ""
#                 if raw_published:
#                     try:
#                         dt_obj = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
#                         published_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
#                     except Exception:
#                         published_at = raw_published # Fallback

#                 results.append({
#                     "video_id": vid,
#                     "title": html.unescape(raw_title),
#                     "description": clean_desc,
#                     "published_at": published_at,
#                     "duration_seconds": duration_seconds if duration_seconds is not None else 0,
#                     "is_short": short_flag,
#                 })

#         # Reorder results based on original collected_video_ids order
#         vid_map = {r["video_id"]: r for r in results}
#         ordered_results = [vid_map[v_id] for v_id in collected_video_ids if v_id in vid_map]

#         print(f"  Fetched details for {len(ordered_results)} videos for channel {channel_id}")
#         return ordered_results, channel_description

#     except Exception as e:
#         print(f"❌ Error fetching video details for {channel_id}: {e}")
#         # Return empty list and description
#         return [], channel_description
def fetch_recent_videos(
    channel_id: str, max_results: int = 30, filter_shorts: bool = True
) -> Tuple[List[Dict], str]:
    """
    Fetch up to max_results recent videos (filtering shorts *during* fetch until limit is met or playlist ends)
    and channel description.
    """
    if not channel_id: return [], ""
    if not API_KEY: print("❌ ERROR: API key missing."); return [], ""

    channel_description = ""
    uploads_playlist = None
    results: List[Dict] = [] # Final list of videos passing filters

    # 1) Get channel details (snippet + contentDetails)
    url = f"{YT_BASE}/channels"
    params = {"part": "contentDetails,snippet", "id": channel_id, "key": API_KEY}
    try:
        data = _safe_get_json(url, params)
        items = data.get("items", [])
        if not items: print(f"⚠️ No channel data for ID {channel_id}"); return [], ""
        item = items[0]
        channel_description = item.get("snippet", {}).get("description", "")
        uploads_playlist = item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        if not uploads_playlist: print(f"⚠️ No uploads playlist for {channel_id}"); return [], channel_description
    except Exception as e: print(f"❌ Error fetching channel details for {channel_id}: {e}"); return [], ""

    # 2) Iterate through playlist pages, fetch details, filter, UNTIL max_results is reached OR playlist ends
    playlist_url = f"{YT_BASE}/playlistItems"
    vids_url = f"{YT_BASE}/videos"
    next_page = None
    fetched_ids_in_batch = []
    processed_playlist_items = 0

    print(f"  Fetching videos for {channel_id} (target: {max_results} non-shorts)...")

    # This loop continues as long as we haven't reached the target number of valid videos
    while len(results) < max_results:
        print(f"    Fetching playlist batch (found {len(results)}/{max_results} valid videos so far)...")
        # --- Fetch a batch of Video IDs from the playlist ---
        try:
            params_pl = {
                "part": "contentDetails",
                "playlistId": uploads_playlist,
                "maxResults": 50, # Always fetch max batch size for efficiency
                "key": API_KEY,
            }
            if next_page:
                params_pl["pageToken"] = next_page

            page = _safe_get_json(playlist_url, params_pl)
            fetched_ids_in_batch = [
                it.get("contentDetails", {}).get("videoId")
                for it in page.get("items", [])
                if it.get("contentDetails", {}).get("videoId")
            ]
            processed_playlist_items += len(page.get("items", []))
            next_page = page.get("nextPageToken") # Get token for the *next* iteration

            # Check if the playlist ended in this fetch
            if not fetched_ids_in_batch and not next_page:
                 print(f"    No more video IDs found in playlist (end reached).")
                 break # Exit the while loop if no more items AND no next page token

            # Handle case where a page might be empty but there's a next page (rare)
            if not fetched_ids_in_batch and next_page:
                 print(f"    Empty batch but next page exists, continuing...")
                 continue # Go to next iteration to fetch the next page

        except Exception as e:
            print(f"    ❌ Error fetching playlist batch: {e}")
            break # Stop if playlist fetching fails

        # --- Fetch details for the IDs in this batch ---
        print(f"    Fetching details for {len(fetched_ids_in_batch)} videos...")
        batch_results_unfiltered = []
        try:
            # Need to check fetched_ids_in_batch again as it might be empty if we continued above
            if not fetched_ids_in_batch: continue

            params_vid = {
                "part": "snippet,contentDetails",
                "id": ",".join(fetched_ids_in_batch),
                "maxResults": 50,
                "key": API_KEY,
            }
            vdata = _safe_get_json(vids_url, params_vid)
            batch_results_unfiltered = vdata.get("items", [])

        except Exception as e:
            print(f"    ❌ Error fetching video details batch: {e}")
            break # Stop if video details fetching fails

        # --- Filter this batch and add to results ---
        print(f"    Filtering batch...")
        filtered_count_in_batch = 0
        for item in batch_results_unfiltered:
            # Check if we've already hit the target *before* processing this item
            if len(results) >= max_results: break

            vid = item.get("id")
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            raw_title = snippet.get("title", "")
            duration_iso = content.get("duration", None)
            duration_seconds = parse_iso8601_duration(duration_iso) if duration_iso else None
            is_short = is_short_video(raw_title, duration_seconds) # Use your updated is_short_video logic here

            # Apply the filter
            if filter_shorts and is_short:
                 continue # Skip this video

            # Video passed filter, format and add it
            raw_desc = snippet.get("description", "") or ""
            raw_published = snippet.get("publishedAt", None)
            clean_desc = html.unescape(raw_desc).replace("\n", " ").replace("\r", " ").strip()
            published_at = ""
            if raw_published:
                try:
                    dt_obj = datetime.fromisoformat(raw_published.replace("Z", "+00:00"))
                    published_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except Exception: published_at = raw_published

            results.append({
                "video_id": vid,
                "title": html.unescape(raw_title),
                "description": clean_desc,
                "published_at": published_at,
                "duration_seconds": duration_seconds if duration_seconds is not None else 0,
                "is_short": is_short,
            })
            filtered_count_in_batch += 1

        print(f"    Added {filtered_count_in_batch} valid videos from this batch.")

        # --- Check if playlist ended (important!) ---
        # If there was no next page token returned from the playlist fetch, we're done.
        if not next_page:
            print("    Reached end of playlist.")
            break # Exit the while loop

    # End of while loop

    # --- Final Output ---
    # Ensure we don't exceed max_results just in case
    final_results = results[:max_results]
    print(f"  ✅ Finished fetching for {channel_id}. Found {len(final_results)} valid videos (target: {max_results}). Processed approx {processed_playlist_items} playlist items.")
    return final_results, channel_description


# -----------------------------
# Top-level batch fetch
# -----------------------------
def fetch_for_seed_channels(
    seed_df, limit_per_channel: int = 30, filter_shorts: bool = True
) -> "pd.DataFrame":
    """ Fetches videos and channel descriptions for seed channels """
    import pandas as pd # Import here as it's only used here
    all_records = []
    for _, row in seed_df.iterrows():
        channel_url = row.get("Channel_URL", "")
        channel_name = row.get("Channel_Name", "")
        print(f"\n🎯 Fetching for seed: {channel_name} ({channel_url or 'URL Missing'})")
        if not channel_url:
            print("   Skipping due to missing URL.")
            continue

        channel_id = extract_channel_id(channel_url)
        if not channel_id:
            print(f"   Skipping {channel_name} (could not resolve ID).")
            continue

        videos, channel_desc = fetch_recent_videos(
            channel_id, max_results=limit_per_channel, filter_shorts=filter_shorts
        )

        if videos: # Only add if videos were found
            for v in videos:
                all_records.append({
                    "Channel_Name": channel_name,
                    "Channel_ID": channel_id,
                    "channel_description": channel_desc,
                    "video_id": v["video_id"],
                    "title": v["title"],
                    "description": v["description"],
                    "published_at": v["published_at"],
                    "duration_seconds": v["duration_seconds"],
                    "is_short": v["is_short"],
                })
        else:
             print(f"   No videos fetched for {channel_name}")

    if not all_records:
         print("\n⚠️ No videos fetched for any seed channel.")
         return pd.DataFrame() # Return empty DataFrame

    df_videos = pd.DataFrame(all_records)
    print(f"\n✅ Fetched {len(df_videos)} total videos across {seed_df['Channel_Name'].nunique()} seed channels.")
    return df_videos


# --- Batch Channel Metadata Fetch ---
def get_channel_metadata_batch(channel_ids: List[str]) -> List[Dict]:
    """ Fetches metadata for multiple channel IDs """
    if not API_KEY: print("❌ ERROR: API key missing."); return []
    if not channel_ids: return []

    print(f"  Fetching metadata for {len(channel_ids)} channels...")
    channel_data = []
    processed_count = 0
    url = f"{YT_BASE}/channels"

    for i in range(0, len(channel_ids), 50):
        batch_ids = channel_ids[i : i + 50]
        batch_num = (i // 50) + 1
        # print(f"     Fetching batch {batch_num} ({len(batch_ids)} IDs)...") # Less verbose

        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch_ids),
            "maxResults": 50,
            "key": API_KEY
        }

        try:
            response = _safe_get_json(url, params)
            items = response.get("items", [])
            # print(f"     Batch {batch_num}: Got {len(items)} channels.") # Less verbose
            processed_count += len(items)

            for item in items:
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                channel_id = item.get("id")
                channel_name = snippet.get("title")
                custom_url_handle = snippet.get("customUrl")
                channel_url = f"https://www.youtube.com/{custom_url_handle}" if custom_url_handle and custom_url_handle.startswith('@') else f"https://www.youtube.com/channel/{channel_id}"

                channel_data.append({
                    "id": channel_id,
                    "name": channel_name,
                    "url": channel_url,
                    "description": snippet.get("description", ""),
                    "subscribers": int(stats.get("subscriberCount", 0)) if not stats.get("hiddenSubscriberCount", False) else -1,
                    "video_count": int(stats.get("videoCount", 0))
                })
        except Exception as e:
            print(f"   ❌ ERROR fetching metadata batch {batch_num}: {e}")

    print(f"  Finished fetching metadata. Got details for {processed_count} channels.")
    return channel_data


# --- Multi-Focused Channel Search ---
def search_videos_multi_focused(keywords: List[str], max_results_per_search: int = 10, max_keywords: int = 7) -> set[str]:
    """ Performs multiple searches biased towards English """
    if not API_KEY: print("❌ ERROR: API key missing."); return set()
    if not keywords: print("⚠️ WARNING: No keywords provided."); return set()

    keywords_to_search = min(len(keywords), max_keywords)
    all_candidates = set()

    print(f"  🔎 Performing {keywords_to_search} focused searches (biased to English)...")

    for i in range(keywords_to_search):
        keyword = keywords[i]
        search_query = keyword # No quotes, as discussed

        print(f"     Search {i+1}/{keywords_to_search}: '{search_query}'")

        url = f"{YT_BASE}/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "channel",
            "order": "relevance",
            "maxResults": max_results_per_search,
            "key": API_KEY,
            "relevanceLanguage": "en" # <-- ADDED LANGUAGE BIAS
        }

        try:
            response = _safe_get_json(url, params)
            items = response.get("items", []) # Changed var name from 'videos' to 'items'
            found_count = 0
            for item in items:
                # Ensure it's actually a channel result
                if item.get("id", {}).get("kind") == "youtube#channel":
                    ch_id = item.get("id", {}).get("channelId")
                    # Sometimes search returns videoId even with type=channel, filter those
                    if ch_id:
                         all_candidates.add(ch_id)
                         found_count += 1

            print(f"        ✅ {found_count} channels found")

        except Exception as e:
            print(f"        ❌ Search Error: {str(e)[:100]}") # Show more error context
            continue

    print(f"  📊 Total: {len(all_candidates)} unique candidate channels found from {keywords_to_search} searches.\n")
    return all_candidates



# Client approach----
# ==================================================
# CLIENT'S FREQUENCY-BASED APPROACH
# ==================================================

def frequency_search_by_titles(
    titles: List[str],
    seed_channel_id: str,
    max_results_per_title: int = 20
) -> Tuple[Counter, Dict[str, Dict]]:
    """
    Client's approach: Search each title, count channel frequency
    
    Args:
        titles: List of video titles to search
        seed_channel_id: ID of seed channel (to exclude from results)
        max_results_per_title: How many results per title search
    
    Returns:
        (channel_frequency_counter, channel_metadata_dict)
    """
    if not API_KEY:
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
                "relevanceLanguage": "en",   # English preference
                "key": API_KEY
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
                        "first_seen_in": title[:60]
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
    existing_metadata: Dict[str, Dict] = None
) -> Dict[str, Dict]:
    """
    Enrich channel metadata with full details (subs, videos, etc.)
    Uses batch API calls for efficiency
    
    Args:
        channel_ids: List of channel IDs to enrich
        existing_metadata: Existing partial metadata dict (optional)
    
    Returns:
        Dict of channel_id -> full_metadata
    """
    if not API_KEY:
        print("❌ ERROR: API key missing")
        return {}
    
    if not channel_ids:
        return {}
    
    print(f"\n📦 Enriching metadata for {len(channel_ids)} channels...")
    
    enriched = existing_metadata.copy() if existing_metadata else {}
    channels_url = f"{YT_BASE}/channels"
    
    # Batch in groups of 50
    for i in range(0, len(channel_ids), 50):
        batch_ids = channel_ids[i:i+50]
        batch_num = (i // 50) + 1
        
        try:
            params = {
                "part": "snippet,statistics",
                "id": ",".join(batch_ids),
                "key": API_KEY
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
                    enriched[channel_id].update({
                        "name": snippet.get("title"),
                        "description": snippet.get("description", ""),
                        "country": snippet.get("country", "unknown"),
                        "subscribers": int(stats.get("subscriberCount", 0)),
                        "video_count": int(stats.get("videoCount", 0)),
                        "view_count": int(stats.get("viewCount", 0)),
                        "custom_url": snippet.get("customUrl", ""),
                    })
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
    total_searches: int
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
            
            candidates.append({
                "channel_id": channel_id,
                "channel_name": metadata.get("name", "Unknown"),
                "frequency": frequency,
                "frequency_score": frequency_score,
                "appearances": f"{frequency}/{total_searches}",
                "subscribers": metadata.get("subscribers", 0),
                "video_count": metadata.get("video_count", 0),
                "country": metadata.get("country", "unknown"),
                "first_seen_in": metadata.get("first_seen_in", ""),
            })
    
    # Sort by frequency (highest first)
    candidates.sort(key=lambda x: x["frequency"], reverse=True)
    
    print(f"✅ Found {len(candidates)} candidates above threshold")
    
    return candidates
