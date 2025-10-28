import re
import time
import random
import requests
import yt_dlp
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# --------------------------
# Core Transcript Fetch Logic
# --------------------------
def fetch_transcript(video_id, max_retries=3):
    """Fetch transcript using hybrid method (API → yt_dlp)."""
    for attempt in range(1, max_retries + 1):
        try:
            # Try official transcript API first
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi', 'bn'])
                text = ' '.join([t['text'] for t in transcript])
                return video_id, text, 'api'
            except (TranscriptsDisabled, NoTranscriptFound):
                pass
            except Exception as e:
                if "no element found" not in str(e):
                    print(f"⚠️ API error on attempt {attempt} for {video_id}: {e}")

            # Fallback to yt-dlp
            ydl_opts = {
                'quiet': True,
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'en-orig', 'a.en', 'hi', 'hi-orig', 'bn'],
                'simulate': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                subs = info.get('subtitles') or info.get('automatic_captions')
                if not subs:
                    raise ValueError("No subtitles")

                for lang_key in subs.keys():
                    if any(x in lang_key for x in ['en', 'hi', 'bn']):
                        r = requests.get(subs[lang_key][0]['url'], timeout=10)
                        if r.ok:
                            vtt = r.text
                            text = re.sub(r'\d+\n|-->.*\n', '', vtt)
                            text = re.sub(r'\s+', ' ', text).strip()
                            return video_id, text, lang_key

            raise ValueError("No subtitles found in preferred languages")

        except Exception as e:
            wait_time = 2 ** attempt + random.random()
            print(f"⚠️ Attempt {attempt}/{max_retries} failed for {video_id}: {e} (retrying in {wait_time:.1f}s)")
            time.sleep(wait_time)

    return video_id, "", "failed"


# --------------------------
# Parallel Runner
# --------------------------
def process_videos(video_ids, max_workers=5):
    """Fetch transcripts concurrently for multiple videos."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_transcript, vid): vid for vid in video_ids}

        for future in as_completed(futures):
            vid = futures[future]
            try:
                video_id, text, source = future.result()
                print(f"✅ [{source}] {video_id} → {len(text)} chars")
                results.append((video_id, text, source))
            except Exception as e:
                print(f"❌ Failed {vid}: {e}")
                results.append((vid, "", "error"))
    return results


# --------------------------
# Save to CSV
# --------------------------
def save_to_csv(results, output_path="transcripts.csv"):
    """Save transcript results to CSV file."""
    # Create DataFrame
    df = pd.DataFrame(results, columns=['video_id', 'transcription', 'source'])
    
    # Add additional columns
    df['word_count'] = df['transcription'].apply(lambda x: len(x.split()) if x else 0)
    df['char_count'] = df['transcription'].apply(lambda x: len(x) if x else 0)
    df['status'] = df['transcription'].apply(lambda x: 'success' if x else 'failed')
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Print summary
    print(f"\n💾 Saved to: {output_path}")
    print(f"📈 Success rate: {df['status'].value_counts().get('success', 0)}/{len(df)}")
    
    return df


# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    test_videos = [
        "v5lc7UAAats",  # MKBHD
        "N2P6W2YiVxE",  # Technical Guruji
        "nj8RO74JdSw"   # Some video
    ]

    print(f"🚀 Fetching transcripts for {len(test_videos)} videos (parallel mode)...")
    results = process_videos(test_videos, max_workers=5)

    print("\n📊 Summary:")
    for vid, text, source in results:
        status = "✅" if text else "❌"
        print(f"{status} {vid:15} | {source:10} | {len(text):6} chars")
    
    # Save to CSV
    df = save_to_csv(results, output_path="transcripts.csv")
    
    # Show sample
    print("\n📄 Sample output:")
    print(df[['video_id', 'status', 'word_count', 'source']].head())
    
    # Show first transcript preview
    if len(df[df['status'] == 'success']) > 0:
        print("\n🔍 First transcript preview:")
        first_success = df[df['status'] == 'success'].iloc[0]
        print(f"Video: {first_success['video_id']}")
        print(f"Text: {first_success['transcription'][:200]}...")
        
