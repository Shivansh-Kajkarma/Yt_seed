import os
import re
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
# Use a valid and available model, like gemini-1.5-flash
# Check Google AI Studio for the latest available models
try:
    # generation_config added to limit output tokens for safety/cost
    generation_config = genai.GenerationConfig(
        max_output_tokens=200,
        temperature=0.1 # Lower temperature for more deterministic keywords
    )
    # safety_settings added to avoid potential blocks on slightly edgy content
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=generation_config,
        safety_settings=safety_settings
        )
except Exception as e:
    print(f"Error configuring Gemini model: {e}")
    # Fallback or raise error depending on desired behavior
    # For now, let's allow the script to continue but warn
    model = None
    print("Warning: Gemini model could not be initialized. Keyword extraction will fail.")

# --- Helper Functions ---

def get_video_id_from_url(url):
    """Extracts YouTube video ID from various URL formats."""
    if not isinstance(url, str):
        return None
    # Standard watch URL
    match = re.search(r"watch\?v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    # Shortened youtu.be URL
    match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    # Shorts URL
    match = re.search(r"shorts/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    # Embed URL
    match = re.search(r"embed/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    print(f"Warning: Could not extract video ID from URL: {url}")
    return None # Return None if no ID found


def get_transcripts_text(video_ids: list[str], max_to_try: int = 15, min_needed: int = 3, target_num: int = 5) -> str | None:
    """
    Fetches transcripts for a list of video IDs and returns combined text.
    Tries up to 'max_to_try' videos, stops if 'target_num' are found.
    Returns None if fewer than 'min_needed' transcripts are successfully fetched.
    """
    transcripts_found = []
    tried_count = 0

    print(f"  Attempting to fetch transcripts for up to {max_to_try} videos (aiming for {target_num})...")

    for video_id in video_ids:
        if not video_id: continue # Skip if video_id is None or empty

        tried_count += 1
        if tried_count > max_to_try or len(transcripts_found) >= target_num:
            break

        try:
            # Fetch transcript (English preferred, fallback to available)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_generated_transcript(['en']) # Prioritize English
            # transcript = transcript_list.find_manually_created_transcript(['en']) # Or try manual first

            full_transcript = transcript.fetch()
            text = " ".join([entry['text'] for entry in full_transcript])
            transcripts_found.append(text)
            print(f"    ✓ Transcript found for {video_id}")

        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"    - No transcript available for {video_id}")
            continue
        except Exception as e:
            print(f"    ✗ Error fetching transcript for {video_id}: {e}")
            continue

    print(f"  Found {len(transcripts_found)} transcripts out of {tried_count} attempts.")

    if len(transcripts_found) >= min_needed:
        return " ".join(transcripts_found)
    else:
        return None


def extract_keywords_gemini(text_content: str, max_chars: int = 18000) -> list[str]:
    """
    Uses the configured Gemini model to extract keywords from the provided text.
    Limits input text length.
    """
    if not model:
        print("  Error: Gemini model not initialized. Cannot extract keywords.")
        return []
    if not text_content:
        print("  Warning: No text content provided to Gemini.")
        return []

    # Limit input size to control cost/performance
    truncated_content = text_content[:max_chars]

    prompt = f"""Analyze the following text content from multiple videos of a single YouTube channel. Extract a list of 12-15 specific and descriptive keywords or short phrases (2-3 words max) that represent the core topics, themes, and niche of the channel.

Focus on:
- Subject matter (e.g., 'quantum physics', 'ancient history', 'deep sea exploration', 'game development', 'financial independence')
- Specific recurring elements, technologies, or concepts discussed.
- The overall style or genre if distinctive (e.g., 'educational animation', 'investigative documentary', 'comedy sketch')

Avoid:
- Generic YouTube terms ('video', 'channel', 'subscribe', 'tutorial', 'guide', 'tips', 'best', 'top')
- Vague words ('world', 'people', 'life', 'thing', 'make', 'get')
- Calls to action or boilerplate text.

Content:
\"\"\"
{truncated_content}
\"\"\"

Return ONLY a comma-separated list of the keywords/phrases. Do not include numbering, explanations, or any other text. Example: quantum physics, black holes, string theory, educational animation, space exploration"""

    try:
        response = model.generate_content(prompt)
        # Handle potential safety blocks or empty responses
        if not response.parts:
             print("  Warning: Gemini response blocked or empty.")
             # Check candidate.safety_ratings for block reasons if needed
             # print(response.candidates[0].safety_ratings)
             return []

        keywords_text = response.text
        keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip()]
        # Filter potentially empty strings again after stripping
        keywords = [kw for kw in keywords if kw]
        return keywords[:15] # Limit to 15

    except Exception as e:
        print(f"  Error during Gemini API call: {e}")
        return []


# --- Main Tiered Function ---

def create_channel_fingerprint_gemini(
    channel_id: str,
    video_ids: list[str],
    video_titles: list[str],
    video_descriptions: list[str]
) -> list[str]:
    """
    AUTOMATED 2-TIER APPROACH using Gemini.
    Tier 1: Use transcripts if available.
    Tier 2: Use titles + descriptions as fallback.
    """
    print(f"\n🔍 Generating Gemini Fingerprint for channel: {channel_id}")

    # ===== TIER 1: Try transcripts + Gemini =====
    print("  → Tier 1: Attempting Transcript Analysis...")
    transcript_text = get_transcripts_text(video_ids, max_to_try=15, min_needed=3, target_num=5)

    if transcript_text:
        print("  Processing transcripts with Gemini...")
        keywords_tier1 = extract_keywords_gemini(transcript_text)
        if keywords_tier1: # Check if Gemini returned valid keywords
             print(f"  ✅ Tier 1 SUCCESS (using transcripts)")
             print(f"  Keywords: {keywords_tier1}")
             return keywords_tier1
        else:
             print("  ⚠️ Tier 1 Warning: Gemini failed to extract keywords from transcripts. Proceeding to Tier 2.")
    else:
        print(f"  ❌ Tier 1 FAILED (Not enough transcripts found)")

    # ===== TIER 2: Use titles + descriptions + Gemini (Fallback) =====
    print("  → Tier 2: Using Titles + Descriptions with Gemini...")
    # Combine titles (weighted slightly more by repeating) and descriptions
    # Ensure lists are of the same length or handle appropriately
    # Simple combination assuming titles and descriptions correspond
    combined_texts = []
    num_videos = min(len(video_titles), len(video_descriptions))
    for i in range(num_videos):
         # Give titles a bit more presence
         combined_texts.append(str(video_titles[i]) * 2 + " " + str(video_descriptions[i]))

    full_text_blob = " ".join(combined_texts)

    if not full_text_blob.strip():
        print("  ❌ Tier 2 FAILED (No titles or descriptions available)")
        return [] # Return empty list if no text content at all

    keywords_tier2 = extract_keywords_gemini(full_text_blob)

    if keywords_tier2:
        print(f"  ✅ Tier 2 SUCCESS (using titles/descriptions)")
        print(f"  Keywords: {keywords_tier2}")
        return keywords_tier2
    else:
         print("  ❌ Tier 2 FAILED: Gemini failed to extract keywords from titles/descriptions.")
         return [] # Return empty if Gemini fails completely