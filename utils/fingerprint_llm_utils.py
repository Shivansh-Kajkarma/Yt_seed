# /utils/fingerprint_llm_utils.py

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
def extract_keywords_llm(combined_text: str, max_chars: int = 40000) -> list[str]:
    """Uses Gemini to extract keywords from combined text."""
    if not model:
        print("  ❌ Gemini model not initialized.")
        return []
    if not combined_text:
        print("  ⚠️  No text content provided.")
        return []

    truncated_content = combined_text[:max_chars]
    print(f"  📤 Sending {len(truncated_content)} chars to Gemini...")

    prompt = f"""Act as an expert YouTube channel analyst. Analyze the following combined text from recent video titles and descriptions of a single channel.

Extract 15 specific keywords/phrases (2-4 words max) that represent the channel's niche.

PRIORITIZE:
- Specific topics, technologies, games, concepts, people, places
- Channel's unique angle/genre (e.g., 'animated science', 'tech reviews')

AVOID:
- Generic YouTube words: video, channel, subscribe, new, official, trailer, gameplay, review, shorts, vlog
- Vague fillers: world, life, people, thing, amazing, ultimate, best, top, how to
- Social media links, CTAs, sponsor mentions
- Repeated concepts (pick most representative)

Combined Titles and Descriptions:
\"\"\"
{truncated_content}
\"\"\"

Return ONLY comma-separated keywords. No explanations."""

    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)

            if not response.parts:
                safety_feedback = response.candidates[0].safety_ratings if response.candidates else "No ratings"
                print(f"  ⚠️  Response blocked. Safety: {safety_feedback}")
                return []

            keywords_text = response.text.strip()
            keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip()]
            keywords = [kw for kw in keywords if len(kw) > 1]

            if keywords:
                print(f"  ✅ Got {len(keywords)} keywords from Gemini")
                return keywords[:15]
            else:
                print("  ⚠️  Gemini returned empty list")
                return []

        except Exception as e:
            print(f"  ❌ API error (Attempt {attempt + 1}/{retries}): {str(e)[:100]}")
            
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = 10 * (attempt + 1)
                print(f"    💤 Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print("  ❌ Failed after all retries")
                return []

    return []


# --- Main Fingerprint Creation ---
def create_channel_fingerprint_llm(video_df: pd.DataFrame, top_n: int = 15) -> list[str]:
    """Generates channel fingerprint using Gemini on titles & descriptions."""
    if video_df.empty:
        print("  ⚠️  Empty DataFrame")
        return []

    print(f"  📊 Processing {len(video_df)} videos...")

    combined_texts = []
    for _, row in video_df.iterrows():
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))

        clean_title = preprocess_text_for_llm(title)
        clean_desc = preprocess_text_for_llm(description)

        combined_texts.append(clean_title * 2)  # Weight title 2x
        combined_texts.append(clean_desc)

    full_text_blob = "\n---\n".join(filter(None, combined_texts))

    if not full_text_blob:
        print("  ⚠️  No text content after preprocessing")
        return []

    keywords = extract_keywords_llm(full_text_blob)
    return keywords
