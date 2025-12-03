import yt_dlp
import requests
import re
import random
import os
import time
from pathlib import Path

# ==========================================
# 1. CONFIGURATION & STEALTH CONSTANTS
# ==========================================

# Locate cookies.txt in the root directory (parent of 'utils')
BASE_DIR = Path(__file__).resolve().parent.parent
print(f"Using BASE_DIR: {BASE_DIR}")
COOKIES_PATH = os.path.join(BASE_DIR, "cookies.txt")

# Random User-Agents to rotate (Anti-Bot)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:118.0) Gecko/20100101 Firefox/118.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# ==========================================
# 2. YOUR SRT PARSING LOGIC (Unchanged)
# ==========================================

def parse_srt_to_readable(caption_text):
    """
    Parses a raw SRT string and returns a clean, human-readable transcript.
    """
    if not caption_text:
        return ""

    lines = caption_text.splitlines()
    transcript_parts = []
    
    for line in lines:
        line = line.strip()
        # 1. Skip empty lines
        if not line: continue
        # 2. Skip numeric indices
        if line.isdigit(): continue
        # 3. Skip timestamps
        if '-->' in line: continue
            
        # 4. Clean up text
        line = line.replace('>>', '')
        line = re.sub(r'\[.*?\]', '', line)
        
        if line:
            transcript_parts.append(line)

    full_text = " ".join(transcript_parts)
    clean_text = re.sub(r'\s+', ' ', full_text).strip()
    return clean_text

# ==========================================
# 3. MAIN FETCH FUNCTION (With Stealth)
# ==========================================

def fetch_video_data_ytdlp(video_id: str):
    """
    Fetches metadata + Captions using yt-dlp with Cookies & Random Headers.
    Returns the EXACT dictionary structure from your original code.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # 1. Configure Stealth Options
    selected_user_agent = random.choice(USER_AGENTS)
    time.sleep(random.uniform(1.0, 5.0))
    ydl_opts = {
        # --- STEALTH ---
        'cookiefile': COOKIES_PATH if os.path.exists(COOKIES_PATH) else None,
        'http_headers': {
            'User-Agent': selected_user_agent,
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.youtube.com/',
        },
        # --- PERFORMANCE ---
        'skip_download': True,      # No video download
        'quiet': True,              # No console output
        'no_warnings': True,
        'ignoreerrors': True,       # Don't crash script on error
        
        # --- DATA EXTRACTION ---
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        
        # Simulating the 'youtube:player_client=default,web' logic
        'extractor_args': {'youtube': {'player_client': ['default', 'web']}},
    }

    # 2. Run Extraction
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            data = ydl.extract_info(url, download=False)
            
        if not data:
            print(f"❌ Error fetching {video_id}: No data returned")
            return None

        # ==========================================
        # 4. YOUR PARSING LOGIC (Mapped Exactly)
        # ==========================================
        
        # category
        category = "Unknown"
        categories = data.get('categories')
        if categories and len(categories) > 0:
            category = categories[0]
        
        # thumbnail
        thumbnail = None
        thumbnails = data.get('thumbnails')
        if thumbnails and len(thumbnails) > 0:
            thumbnail = thumbnails[-1].get('url')
        if not thumbnail:
            thumbnail = data.get('thumbnail')
        
        # has_chapters
        chapters = data.get('chapters') or []
        has_chapters = len(chapters) > 0
        
        # has_captions logic
        auto_caps_dict = data.get('automatic_captions') or {}
        # Handle edge case where it might be a list or dict
        auto_caps = []
        if isinstance(auto_caps_dict, dict):
            auto_caps = auto_caps_dict.get('en') or []
            
        subs_dict = data.get('subtitles') or {}
        subs = []
        if isinstance(subs_dict, dict):
            subs = subs_dict.get('en') or []
            
        has_captions = len(auto_caps) > 0 or len(subs) > 0
        
        # Fetch actual caption content (Your requests logic)
        caption_urls = []
        
        # Helper to find SRT url
        def extract_srt(track_list):
            if not track_list: return
            for track in track_list:
                if track and isinstance(track, dict) and track.get('url'):
                    u = track['url']
                    if 'fmt=srt' in u or 'format=srt' in u: # Enhanced check
                        caption_urls.append(u)
                        return # Break inner loop
                        
        extract_srt(auto_caps)
        if not caption_urls: # If no auto, try manual
            extract_srt(subs)

        # Download the text
        caption_text = None
        if caption_urls:
            try:
                # Use the same headers for the request to avoid blocking
                headers = {'User-Agent': selected_user_agent}
                response = requests.get(caption_urls[0], headers=headers, timeout=10)
                if response.status_code == 200:
                    caption_text = parse_srt_to_readable(response.text)
            except Exception as e:
                print(f"⚠️ Error fetching raw captions for {video_id}: {e}")

        # ==========================================
        # 5. FINAL RETURN (Your Exact JSON Structure)
        # ==========================================
        return {
            'video_id': video_id,
            'title': data.get('title'),
            'description': data.get('description'),
            'category': category,
            'tags': data.get('tags') or [],
            'thumbnail': thumbnail,
            "duration": data.get('duration'),
            'has_chapters': has_chapters,
            'chapters': chapters,
            'has_captions': has_captions,
            'caption_tracks': caption_text, # This is the full readable string
            'length_of_captions': len(caption_text) if caption_text else 0,
        }

    except Exception as e:
        print(f"❌ Critical Error in fetch_video_data_ytdlp for {video_id}: {e}")
        return None

# Optional: Test block to ensure it works immediately
if __name__ == "__main__":
    print("Testing utils/yt_dlp_utils.py...")
    vid = "dDd9vJwz2-I" # Example ID
    result = fetch_video_data_ytdlp(vid)
    if result:
        print(f"✅ Success! Title: {result['title']}")
        print(f"   Captions Found: {len(result['caption_tracks']) if result['caption_tracks'] else 0} chars")
    else:
        print("❌ Failed.")