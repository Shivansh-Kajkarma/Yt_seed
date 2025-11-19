import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
        # caption_tracks: [ (.automatic_captions.en // []), (.subtitles.en // []) | .[].url ] | unique | select(. != null)
        # caption_urls = []
        # for track in auto_caps:
        #     if track and isinstance(track, dict) and track.get('url'):
        #         caption_urls.append(track['url'])
        # for track in subs:
        #     if track and isinstance(track, dict) and track.get('url'):
        #         caption_urls.append(track['url'])
        # caption_tracks = list(set(filter(None, caption_urls)))  # unique and filter nulls

        caption_urls = []
        for track in auto_caps:
            if track and isinstance(track, dict) and track.get('url'):
                url = track['url']
                if 'fmt=srt' in url:  # ← Add this filter
                    caption_urls.append(url)
        for track in subs:
            if track and isinstance(track, dict) and track.get('url'):
                url = track['url']
                if 'fmt=srt' in url:  # ← Add this filter
                    caption_urls.append(url)
        caption_tracks = list(set(filter(None, caption_urls)))
        
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
            'caption_tracks': caption_tracks
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
