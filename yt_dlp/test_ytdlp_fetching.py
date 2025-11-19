import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import re
# video_ids = ["dDd9vJwz2-I",
# "-2AIelIFV5w",
# "xAA3F3FoX24",
# "-VI6o-f2ILY",
# "TpiuZdilY78",
# "damR4-DCPJs",
# "uq7bbQiMpJI",
# "3ePPiPRTE_A",
# "fxBCJ_bpONE",
# "tTMn0bquiOI"
# ]  # your 30 video IDs

video_ids = ["dDd9vJwz2-I",
"-2AIelIFV5w",
"xAA3F3FoX24"
] 

def parse_srt_to_readable(caption_text):
    """
    Parses a raw SRT string and returns a clean, human-readable transcript.
    
    Args:
        caption_text (str): The raw content of the .srt file.
        
    Returns:
        str: A clean, continuous string of text.
    """
    if not caption_text:
        return ""

    # Split the text into lines
    lines = caption_text.splitlines()
    
    transcript_parts = []
    
    for line in lines:
        line = line.strip()
        
        # 1. Skip empty lines
        if not line:
            continue
            
        # 2. Skip numeric indices (SRT blocks start with a number like 1, 100, etc.)
        if line.isdigit():
            continue
            
        # 3. Skip timestamps (Lines containing '-->')
        if '-->' in line:
            continue
            
        # 4. Clean up the actual text line
        # Remove speaker change indicators often found in captions (>>)
        line = line.replace('>>', '')
        
        # Remove standard sound effects brackets like [music] or [laughter]
        # (Optional: remove this line if you want to keep sound tags)
        line = re.sub(r'\[.*?\]', '', line)
        
        # Add to list
        if line:
            transcript_parts.append(line)

    # Join all parts with a space to create a continuous flow
    full_text = " ".join(transcript_parts)
    
    # Clean up any accidental double spaces created during the join
    clean_text = re.sub(r'\s+', ' ', full_text).strip()
    
    return clean_text

def fetch_video_data(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        'yt-dlp', url,
        '--dump-single-json',
        '--no-download',
        '--quiet',
        '--extractor-args', 'youtube:player_client=default,web',
        '--no-warnings',
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Error fetching {video_id}: {result.stderr}")
            return None
            
        data = json.loads(result.stdout)
        
        # category: (.categories[0] // "Unknown")
        category = "Unknown"
        categories = data.get('categories')
        if categories and len(categories) > 0:
            category = categories[0]
        
        # thumbnail: (.thumbnails[-1].url // .thumbnail)
        thumbnail = None
        thumbnails = data.get('thumbnails')
        if thumbnails and len(thumbnails) > 0:
            thumbnail = thumbnails[-1].get('url')
        if not thumbnail:
            thumbnail = data.get('thumbnail')
        
        # has_chapters: (.chapters | length > 0)
        chapters = data.get('chapters') or []
        has_chapters = len(chapters) > 0
        
        # has_captions: (((.automatic_captions.en | length) > 0) or ((.subtitles.en | length) > 0))
        auto_caps_dict = data.get('automatic_captions') or {}
        auto_caps = auto_caps_dict.get('en') or [] if isinstance(auto_caps_dict, dict) else []
        
        subs_dict = data.get('subtitles') or {}
        subs = subs_dict.get('en') or [] if isinstance(subs_dict, dict) else []
        
        has_captions = len(auto_caps) > 0 or len(subs) > 0
        
        # caption_tracks: fetch actual SRT content instead of URLs
        caption_urls = []
        for track in auto_caps:
            if track and isinstance(track, dict) and track.get('url'):
                url = track['url']
                if 'fmt=srt' in url:
                    caption_urls.append(url)
                    break  # ← Take only first SRT URL, no need for multiple formats

        for track in subs:
            if track and isinstance(track, dict) and track.get('url'):
                url = track['url']
                if 'fmt=srt' in url:
                    caption_urls.append(url)
                    break

        # Fetch the actual caption content
        caption_text = None
        if caption_urls:
            try:
                response = requests.get(caption_urls[0], timeout=10)
                if response.status_code == 200:
                    caption_text = parse_srt_to_readable(response.text)  # ← Raw SRT text as one big string
            except Exception as e:
                print(f"Error fetching captions for {video_id}: {e}")

        
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
            'caption_tracks': caption_text,
            'length_of_captions': len(caption_text) if caption_text else 0,
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error for {video_id}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching {video_id}: {e}")
        return None
    
# Run in parallel with 8 workers
results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    # Submit all tasks
    futures = {executor.submit(fetch_video_data, vid): vid for vid in video_ids}
    
    # Collect results as they complete
    for future in as_completed(futures):
        result = future.result()
        if result:
            results.append(result)
            print(f"✓ Fetched: {result['title']}")

# Save all results
with open('video_data.json', 'w') as f:
    json.dump(results, f, indent=2)
