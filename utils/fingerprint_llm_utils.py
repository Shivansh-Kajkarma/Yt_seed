import os
import re
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import time
# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Configure the Gemini client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = genai.GenerationConfig(
        max_output_tokens=200,
        temperature=0.1
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    # ✅ FIXED: Use correct model name
    model = genai.GenerativeModel(
        'gemini-2.0-flash-exp',  # Changed from 'gemini-1.5-flash-latest'
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print("✅ Gemini model configured: gemini-2.0-flash-exp")
    
except Exception as e:
    print(f"❌ Error configuring Gemini model: {e}")
    model = None
    print("Warning: Gemini model could not be initialized.")


# --- Text Preprocessing ---
def preprocess_text_for_llm(text: str) -> str:
    """Basic cleaning: lowercase, remove URLs, consolidate whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- LLM Keyword Extraction ---
def extract_keywords_llm(combined_text: str, channel_name: str = "", max_chars: int = 40000) -> list[str]:
    """Uses Gemini to extract keywords from combined text."""
    if not model:
        print("  ❌ Gemini model not initialized.")
        return []
    if not combined_text:
        print("  ⚠️  No text content provided.")
        return []

    truncated_content = combined_text[:max_chars]
    print(f"  📤 Sending {len(truncated_content)} chars to Gemini...")

    prompt = f"""Act as an expert YouTube channel analyst. Analyze the following combined text from recent video titles and descriptions of a single channel. Your goal is to identify the core niche and recurring topics.

    Based ONLY on the text provided, extract a list of the 15 most specific and descriptive keywords or short phrases (2-4 words max) that accurately represent the primary subject matter and unique focus of this channel.

    Prioritize:
    - Specific topics, technologies, games, scientific concepts, historical periods, locations, food types, or artistic styles frequently mentioned.
    - Terms that define the channel's unique angle or genre (e.g., 'animated science explainer', 'retro game review', 'budget travel guide', 'investigative business documentary').

    CRITICALLY AVOID (Do NOT include these):
    - The channel's own name {channel_name} or variations of it .
    - Names of collaborators or guests UNLESS they are the primary subject of multiple videos.
    - Generic YouTube/internet words ('video', 'channel', 'subscribe', 'shorts', 'vlog', 'new', 'official', 'trailer', 'gameplay', 'review', 'unboxing', 'live stream', 'website', 'merch', 'patreon', 'twitter', 'instagram', 'facebook').
    - Vague fillers ('world', 'life', 'people', 'thing', 'amazing', 'ultimate', 'best', 'top', 'how to', 'why', 'what', 'insane', 'crazy', 'real', 'every').
    - Calls to action, sponsor mentions, social media links/handles, or website URLs.
    - Music/voice credits (e.g., 'steve taylor', 'epic mountain').
    - Extremely common single English words (e.g., 'make', 'get', 'part', 'using', 'hai'). Only use single words if they are specific technical terms (e.g., 'thermodynamics', 'photosynthesis').

    Combined Titles and Descriptions:
    \"\"\"
    {truncated_content}
    \"\"\"

    Return ONLY a comma-separated list of the 10 keywords/phrases, strictly adhering to the AVOID list. Do not add explanations, numbering, or any other text."""

    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)

            # Check for blocked response
            if not response.parts:
                safety_feedback = response.candidates[0].safety_ratings if response.candidates else "No ratings"
                print(f"  ⚠️  Response blocked. Safety: {safety_feedback}")
                return []

            # Get response text
            keywords_text = response.text.strip()
            
            # FIX 1: Split by both commas AND newlines (handles both formats)
            keywords = [kw.strip().lower() for kw in re.split(r'[,\n]+', keywords_text) if kw.strip()]
            
            # Filter short keywords
            keywords = [kw for kw in keywords if len(kw) > 1]

            # FIX 2: Sleep BEFORE return (to avoid rate limits on next call)
            time.sleep(4)  # 15 RPM = 1 request per 4 seconds

            if keywords:
                print(f"  ✅ Got {len(keywords)} keywords from Gemini")
                return keywords[:10]
            else:
                print("  ⚠️  Gemini returned empty list")
                return []

        except Exception as e:
            print(f"  ❌ API error (Attempt {attempt + 1}/{retries}): {str(e)[:100]}")
            
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = 60  # Wait 1 minute for rate limit (not 10s - that's too short)
                print(f"    💤 Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print("  ❌ Failed after all retries")
                return []

    return []



# --- Main Fingerprint Creation ---
def create_channel_fingerprint_llm(video_df: pd.DataFrame, channel_name: str = "", top_n: int = 15) -> list[str]:
    """Generates channel fingerprint using Gemini."""
    if video_df.empty:
        print("  ⚠️  Empty DataFrame")
        return []

    print(f"  📊 Processing {len(video_df)} videos...")

    combined_texts = []
    for _, row in video_df.iterrows():
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))

        # Clean
        clean_title = preprocess_text_for_llm(title)
        
        # NEW: Only take first 200 chars of description (avoid social spam)
        clean_desc = preprocess_text_for_llm(description)
        
        # NEW: Remove channel name from text
        if channel_name:
            clean_title = clean_title.replace(channel_name.lower(), '')
            clean_desc = clean_desc.replace(channel_name.lower(), '')

        combined_texts.append(clean_title * 2)  # Weight title 2x
        combined_texts.append(clean_desc)

    full_text_blob = "\n---\n".join(filter(None, combined_texts))

    if not full_text_blob:
        print("  ⚠️  No text content")
        return []

    keywords = extract_keywords_llm(full_text_blob, channel_name = channel_name)  # Pass channel_name
    return keywords

