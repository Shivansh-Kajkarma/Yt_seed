import os
import time
import requests
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from typing import List, Dict

# --- Load API Key ---
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# --- Initialize YouTube API Client ---
youtube = None
if API_KEY:
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        print("✅ YouTube API client initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize YouTube API client: {e}")
else:
    print("❌ YouTube API key not found in environment variables.")

# --- Helper from your youtube_utils (if needed for Strategy 3) ---
# Assuming you have _safe_get_json and YT_BASE defined as before
YT_BASE = "https://www.googleapis.com/youtube/v3"

def _safe_get_json(url: str, params: dict, max_retries: int = 3, backoff: float = 0.5):
    """ Simple requests wrapper with exponential backoff """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return resp.json()
        except requests.exceptions.RequestException as e:
            # Handle requests-specific errors (network, timeout, etc.)
            print(f"  Request error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise # Re-raise the last error if all retries fail
            wait = backoff * (2**attempt)
            time.sleep(wait)
        except Exception as e:
            # Handle other potential errors during request/parsing
            print(f"  Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
            wait = backoff * (2**attempt)
            time.sleep(wait)
    return None # Should technically not be reached if raise works

# --- Simplified fetch_recent_videos (just gets IDs for Strategy 3) ---
def fetch_recent_video_ids(channel_id: str, max_results: int = 5) -> List[str]:
    """ Gets IDs of recent videos using uploads playlist. """
    if not youtube: return []
    try:
        # 1. Get uploads playlist ID
        chan_resp = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()
        uploads_playlist = chan_resp['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # 2. Get videos from playlist
        pl_resp = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=uploads_playlist,
            maxResults=max_results
        ).execute()

        video_ids = [item['contentDetails']['videoId'] for item in pl_resp.get('items', [])]
        print(f"   Fetched {len(video_ids)} recent video IDs for channel {channel_id}")
        return video_ids
    except HttpError as e:
        print(f"   API Error fetching recent videos for {channel_id}: {e}")
        return []
    except Exception as e:
        print(f"   Unexpected Error fetching recent videos for {channel_id}: {e}")
        return []


# ============================================
# Strategy 2: "Featured Channels" from Branding
# ============================================
def get_featured_channels(seed_channel_id: str) -> List[str]:
    """
    Gets the list of channel IDs featured on the seed channel's page.
    Uses the official API's brandingSettings.
    Returns a list of channel IDs.
    """
    if not youtube:
        print("  ❌ YouTube API client not initialized.")
        return []

    print(f"\n🔍 Strategy 2: Fetching Featured Channels for {seed_channel_id}...")
    try:
        response = youtube.channels().list(
            part='brandingSettings',
            id=seed_channel_id
        ).execute()

        if not response.get('items'):
            print(f"  ⚠️ No channel found with ID: {seed_channel_id}")
            return []

        # Navigate safely through the response structure
        branding = response['items'][0].get('brandingSettings', {})
        channel_settings = branding.get('channel', {})
        featured_urls = channel_settings.get('featuredChannelsUrls', []) # This contains URLs, not IDs

        # --- IMPORTANT: Convert URLs to IDs ---
        # featuredChannelsUrls gives URLs like "/channel/UCxxxx". We need the ID part.
        featured_channel_ids = []
        if featured_urls:
             print(f"   Found {len(featured_urls)} featured channel URLs. Extracting IDs...")
             for url in featured_urls:
                 match = re.search(r"channel/(UC[a-zA-Z0-9_\-]+)", url)
                 if match:
                     featured_channel_ids.append(match.group(1))
                 else:
                      print(f"     ⚠️ Could not extract ID from featured URL: {url}")
        else:
            print("   No featured channels listed by the creator.")


        print(f"  ✅ Found {len(featured_channel_ids)} featured channel IDs.")
        return featured_channel_ids

    except HttpError as e:
        print(f"  ❌ API Error fetching featured channels: {e}")
        return []
    except Exception as e:
        print(f"  ❌ Unexpected Error in get_featured_channels: {e}")
        return []

# =======================================================
# Strategy 3: Approx. "Channels Also Watched" via Related Videos
# =======================================================
def find_related_channels_via_videos(seed_channel_id: str, num_seed_videos: int = 5, results_per_video: int = 10) -> List[str]:
    """
    Approximates "Viewers also watched" by:
    1. Getting recent videos from the seed channel.
    2. Finding videos related to *each* of those seed videos via search.
    3. Collecting the unique channel IDs from the related videos.
    Returns a list of potential competitor/related channel IDs.
    """
    if not youtube:
        print("  ❌ YouTube API client not initialized.")
        return []

    print(f"\n🔍 Strategy 3: Finding related channels via videos for {seed_channel_id}...")

    # 1. Get recent video IDs from the seed channel
    seed_video_ids = fetch_recent_video_ids(seed_channel_id, max_results=num_seed_videos)
    if not seed_video_ids:
        print("   Could not fetch seed videos. Cannot proceed.")
        return []

    candidate_channel_ids = set()
    print(f"   Searching for videos related to {len(seed_video_ids)} seed videos...")

    # 2. For each seed video, find related videos and their channels
    for i, video_id in enumerate(seed_video_ids, 1):
        print(f"     Processing seed video {i}/{len(seed_video_ids)} ({video_id})...")
        try:
            search_response = youtube.search().list(
                part='snippet',
                relatedToVideoId=video_id, # <-- KEY PARAMETER!
                type='video',
                maxResults=results_per_video
            ).execute()

            found_in_batch = 0
            for item in search_response.get('items', []):
                channel_id = item['snippet']['channelId']
                # Add if it's not the seed channel itself
                if channel_id != seed_channel_id:
                    if channel_id not in candidate_channel_ids:
                         found_in_batch += 1
                    candidate_channel_ids.add(channel_id)

            print(f"       Found {found_in_batch} new related channel IDs.")
            time.sleep(0.5) # Small delay between searches

        except HttpError as e:
            print(f"     ❌ API Error searching related videos for {video_id}: {e}")
            # Don't stop the whole process, just skip this video
            continue
        except Exception as e:
            print(f"     ❌ Unexpected Error searching related videos for {video_id}: {e}")
            continue

    print(f"\n  ✅ Strategy 3 found {len(candidate_channel_ids)} unique potential related channel IDs.")
    return list(candidate_channel_ids)


# === Example Usage (Replace with your actual Channel ID) ===
if __name__ == "__main__":
    if not youtube:
        print("\nCannot run examples because YouTube API client failed to initialize.")
    else:
        # --- Example for Ali Abdaal (replace with actual ID) ---
        # You need to find Ali Abdaal's actual Channel ID (starts with UC...)
        # You can often find it in the source code of his channel page or using online tools
        ali_abdaal_channel_id = "UCLXo7UDZvByw2ixzpQCufnA" # Replace with actual ID if different

        print("\n" + "="*50)
        print("Testing Strategy 2: Featured Channels")
        print("="*50)
        featured = get_featured_channels(ali_abdaal_channel_id)
        if featured:
            print("\nFeatured Channel IDs found:")
            for ch_id in featured:
                print(f"- {ch_id}")
        else:
             print("\nNo featured channels found or error occurred.")


        print("\n" + "="*50)
        print("Testing Strategy 3: Related via Videos")
        print("="*50)
        related_via_vids = find_related_channels_via_videos(ali_abdaal_channel_id, num_seed_videos=5, results_per_video=10)
        if related_via_vids:
            print("\nPotential Related Channel IDs found:")
            # Fetch names for the first few IDs for better context
            if len(related_via_vids) > 0:
                 try:
                      ids_to_fetch = related_via_vids[:min(len(related_via_vids), 10)] # Get names for up to 10
                      channel_details = youtube.channels().list(part='snippet', id=",".join(ids_to_fetch)).execute()
                      name_map = {item['id']: item['snippet']['title'] for item in channel_details.get('items', [])}
                      print("   (Showing names for first few results)")
                      for ch_id in related_via_vids:
                           print(f"- {ch_id} ({name_map.get(ch_id, 'Name not fetched')})")
                 except Exception as e:
                      print("   (Error fetching channel names, showing IDs only)")
                      for ch_id in related_via_vids:
                           print(f"- {ch_id}")
            else:
                 for ch_id in related_via_vids:
                     print(f"- {ch_id}")

        else:
             print("\nNo related channels found via videos or error occurred.")