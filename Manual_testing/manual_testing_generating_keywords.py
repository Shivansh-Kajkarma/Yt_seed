import pandas as pd
import json
import re
import time
from pathlib import Path
import os
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# --- Import necessary utils ---
# Import the actual LLM client objects from your utility file
from utils.fingerprint_llm_utils import gemini_model, gpt_client

# === Configuration ===
INPUT_CSV_PATH = Path("./sample_videos.csv") # <-- Load from this file
TARGET_CHANNEL_NAME = "Ali Abdaal"
# Use a strong model for dynamic categorization
MODEL_FOR_KEYWORDS = "gpt-4o" # Or "gemini-1.5-pro" if you have access

# --- Cache Setup ---
CACHE_DIR = Path(__file__).resolve().parent / "manual_outputs" / "cache_keywords_dynamic_only" # New cache folder
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cache_key(*args) -> str:
    s = str(args)
    return hashlib.md5(s.encode()).hexdigest()

def load_from_cache(filename: str):
    cache_file = CACHE_DIR / f"{filename}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                print(f"   CACHE HIT: Loading {filename}.json")
                return json.load(f)
        except Exception as e:
            print(f"   ⚠️ Cache load error for {filename}.json: {e}")
    print(f"   CACHE MISS: {filename}.json")
    return None

def save_to_cache(filename: str, data):
    cache_file = CACHE_DIR / f"{filename}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"   CACHE SAVE: Saved to {filename}.json")
    except Exception as e:
        print(f"   ⚠️ Cache save error for {filename}.json: {e}")
# --- End Cache Setup ---

# --- Utility: Text Cleaning (from your utils, but defined here for clarity) ---
# --- CORRECTED: Removed max_len parameter ---
def preprocess_text_for_llm(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text # No max_len slice

# === Adapted LLM Function (Niche Function Removed) ===

# --- MODIFIED: Takes video_titles list, no description, no niche ---
def extract_dynamic_categorized_keywords_llm_cached(
    video_titles: List[str], # Expects list of title strings
    channel_name: str,
    model_name: str = MODEL_FOR_KEYWORDS,
    max_chars: int = 40000 # Max chars for the *final* blob
) -> Dict[str, List[str]]:
    """ Extracts DYNAMICALLY categorized keywords using LLM, with caching. """
    # Create cache key based only on video titles and channel name
    titles_hash = _get_cache_key(video_titles, channel_name) # Hash the list of titles
    cache_filename = f"dynamic_cat_keywords_{channel_name}_{titles_hash}"
    cached_keywords = load_from_cache(cache_filename)
    if cached_keywords is not None and isinstance(cached_keywords, dict):
        return cached_keywords

    # --- Prepare combined text (Titles Only) ---
    combined_texts = []
    for title in video_titles:
        clean_title = preprocess_text_for_llm(title) # No max_len
        if channel_name:
            clean_title = clean_title.replace(channel_name.lower(), '')
        combined_texts.append(clean_title)
    
    # Format as a newline-separated string
    full_text_blob = "\n".join(filter(None, combined_texts))
    truncated_content = full_text_blob[:max_chars]
    
    blob_cache_filename = f"input_blob_{channel_name}_{titles_hash}"
    blob_cache_file = CACHE_DIR / f"{blob_cache_filename}.txt"
    try:
        with open(blob_cache_file, 'w', encoding='utf-8') as f:
            f.write(f"--- Input Blob for {channel_name} ({len(truncated_content)} chars) ---\n--- Titles Used: {len(video_titles)} ---\n\n")
            f.write(truncated_content)
        print(f"   DEBUG: Saved input text blob to {blob_cache_file.name}")
    except Exception as e:
        print(f"   ⚠️ DEBUG: Error saving input text blob: {e}")

    
    if not truncated_content: 
        print("⚠️ No text blob for keyword extraction."); 
        return {}
    # --- End Prepare ---
    

    print(f"📤 Sending {len(truncated_content)} chars to {model_name} for DYNAMIC categorized keywords...")

  
    # prompt = f"""You are a YouTube competitor research analyst. Your goal is to extract keywords that will help FIND SIMILAR CHANNELS to "{channel_name}" when searched on YouTube.

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHANNEL: "{channel_name}"
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # CONTENT TO ANALYZE:
    # \"\"\"
    # {truncated_content}
    # \"\"\"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OBJECTIVE:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Generate keywords that will surface COMPETITOR CHANNELS when searched on YouTube - channels that create similar content for the same audience.

    # IMPORTANT: These keywords should describe the TYPE of content this channel makes, NOT just the topics they cover.

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TASK:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 1. Identify 3-6 main thematic categories based on recurring content themes
    # 2. For each category, extract 4-6 specific searchable keyword phrases (2-5 words)
    # 3. Ensure 60% are TOPIC keywords (what they discuss) and 40% are FORMAT/STYLE keywords (how they present)

    # KEYWORD QUALITY CRITERIA:

    # ✓ GOOD KEYWORDS (What to include):
    # - Specific actionable phrases: "build lifestyle business", "passive income strategies"
    # - Niche descriptors: "productivity for students", "business documentaries"  
    # - Format indicators: "tutorial style", "case study breakdown", "cinematic storytelling"
    # - Skill/method names: "notion productivity system", "time blocking method"
    # - Audience-specific: "entrepreneurship for beginners", "creator monetization"
    # - Search-intent phrases: What someone types to find this content type

    # ✗ BAD KEYWORDS (What to avoid):
    # - Channel name: '{channel_name}'
    # - Generic words: "how to", "best", "top", "tips", "guide", "tutorial", "hacks", "new"
    # - Platform names: "youtube", "instagram", "tiktok"
    # - Vague terms: "life", "people", "world", "things", "amazing", "ultimate", "awesome"
    # - Single broad topics: "business", "productivity", "technology" (always add specificity!)
    # - Calls to action: "subscribe", "like", "watch"
    # - Specific names/events: Celebrity names, brand names, event names, dates

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REFERENCE EXAMPLES (Match this style):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Example 1 - Educational Business Channel (like Ali Abdaal):
    # {{
    #     "Business & Entrepreneurship": [
    #         "build lifestyle business",
    #         "boring business ideas",
    #         "entrepreneurship 2025",
    #         "make money online",
    #         "business ideas for beginners"
    #     ],
    #     "Financial Freedom": [
    #         "financial freedom",
    #         "passive income",
    #         "wealth building",
    #         "millionaire habits",
    #         "money mindset"
    #     ],
    #     "Productivity & Life Design": [
    #         "productivity tips",
    #         "self improvement",
    #         "time management",
    #         "habits and routines",
    #         "discipline and motivation"
    #     ],
    #     "Creator Economy": [
    #         "youtube growth",
    #         "start youtube channel",
    #         "creator business",
    #         "content creator",
    #         "personal brand"
    #     ]
    # }}

    # Example 2 - Business Documentary Channel (like MagnatesMedia):
    # {{
    #     "Corporate History": [
    #         "company rise and fall",
    #         "business empire story",
    #         "corporate scandal",
    #         "brand failure analysis",
    #         "startup bankruptcy"
    #     ],
    #     "Documentary Storytelling": [
    #         "business documentary",
    #         "cinematic business story",
    #         "company history explained",
    #         "entrepreneur biography",
    #         "corporate investigation"
    #     ],
    #     "Business Analysis": [
    #         "business case study",
    #         "company strategy breakdown",
    #         "business model analysis",
    #         "market disruption story"
    #     ]
    # }}

    # Example 3 - Student Productivity Channel (like Thomas Frank):
    # {{
    #     "Study Techniques": [
    #         "evidence based study tips",
    #         "active recall method",
    #         "spaced repetition",
    #         "study strategies",
    #         "exam preparation"
    #     ],
    #     "Productivity Systems": [
    #         "notion productivity",
    #         "time blocking method",
    #         "second brain building",
    #         "productivity system",
    #         "task management"
    #     ],
    #     "Student Life": [
    #         "college productivity",
    #         "student morning routine",
    #         "study motivation",
    #         "academic success"
    #     ]
    # }}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDATION CHECKLIST:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Before including any keyword, verify:
    # 1. Would someone SEARCH this exact phrase to find similar content?
    # 2. Is it specific enough to filter to a niche? (Not just "business" but "lifestyle business")
    # 3. Does it describe content TYPE, not just a topic?
    # 4. Is it 2-5 words? (Not too short, not too long)
    # 5. Does it avoid all terms in the BAD KEYWORDS list?
    # 6. Each keyword must accurately represent what the channel’s content and narrative style communicate — not clickbait SEO terms.

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OUTPUT FORMAT:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Return ONLY a valid JSON object where:
    # - Keys = Category names (2-4 words, Title Case)
    # - Values = Lists of 4-6 lowercase keyword phrases

    # Ensure JSON is properly formatted with double quotes and correct syntax.

    # Example structure:
    # {{
    #     "Category Name One": ["keyword phrase one", "keyword phrase two", "keyword phrase three", "keyword phrase four"],
    #     "Category Name Two": ["keyword phrase five", "keyword phrase six", "keyword phrase seven", "keyword phrase eight"]
    # }}
    # """

    prompt = f"""You are a YouTube competitor research analyst. Your goal is to extract search keywords that will surface COMPETITOR CHANNELS when searched on YouTube.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        CHANNEL: "{channel_name}"
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        CONTENT TO ANALYZE:
        \"\"\"
        {truncated_content}
        \"\"\"

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        CRITICAL OBJECTIVE:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Generate keywords that REAL USERS actually type into YouTube search to find channels like this one.

        Priority: SEARCHABILITY over specificity. Use the exact phrases people search, even if they seem generic.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        TASK:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        1. Identify 3-6 main content categories (consider: Business, Finance, Productivity, Creator Economy, AI/Tech, Education, Documentary Style, Career, Marketing)
        2. For each category, extract 5-7 keyword phrases that users search to find this content type
        3. Keep phrases SHORT (2-3 words preferred, max 4 words)

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        KEYWORD PRINCIPLES:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ✓ PRIORITIZE (High search volume phrases):
        - Common search queries: "make money online", "financial freedom", "youtube growth"
        - Popular topics: "passive income", "productivity tips", "self improvement"
        - Specific methods: "time blocking", "notion system", "active recall"
        - Audience-specific: "for beginners", "for students", "2025"
        - Natural language: How people actually talk/search

        ✓ ALLOW SELECTIVELY (When it's part of a real search):
        - "how to" phrases: "how to start a business", "how to make money"
        - "tips": "productivity tips", "study tips" (standalone searches)
        - Numbers/years: "business ideas 2025", "ai tools 2025"

        ✗ STRICTLY AVOID:
        - Channel name: '{channel_name}'
        - Unnecessary modifiers: Don't add "strategies", "techniques", "methods", "journey", "guide" unless in original content
        - Platform names: "youtube", "instagram", "tiktok" (unless part of search like "youtube growth")
        - Pure clickbait: "ultimate", "insane", "crazy", "amazing", "shocking"
        - Too generic alone: "business", "productivity", "success" (add context)
        - Calls to action: "subscribe", "like", "watch now"

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        KEYWORD GUIDELINES:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        LENGTH: Prefer 2-3 words. Use 4+ words ONLY if it's a complete search phrase.
        ✓ "financial freedom" (2 words)
        ✓ "make money online" (3 words)
        ✓ "how to start a business" (5 words - complete phrase)
        ✗ "financial freedom journey" (don't add "journey")
        ✗ "passive income strategies" (don't add "strategies")

        NATURALNESS: Use conversational search terms, not formal/academic language.
        ✓ "get rich" (what people search)
        ✗ "wealth accumulation practices" (too formal)
        ✓ "productivity tips" (common search)
        ✗ "productivity optimization methodologies" (too academic)

        SPECIFICITY: Add context to broad terms, but keep it searchable.
        ✗ "business" (too broad)
        ✓ "lifestyle business", "online business", "small business ideas"
        ✗ "content" (too broad)
        ✓ "content creator", "content creation", "content strategy"

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        REFERENCE EXAMPLES (MATCH THIS EXACT STYLE):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Example 1 - Educational Business Channel (Ali Abdaal style):
        {{
            "Business & Entrepreneurship": [
                "build a business",
                "lifestyle business",
                "boring business ideas",
                "entrepreneurship 2025",
                "make money online",
                "business ideas for beginners",
                "how to start a business"
            ],
            "Financial Freedom": [
                "financial freedom",
                "get rich",
                "passive income",
                "how to make money",
                "wealth building",
                "millionaire habits",
                "money mindset"
            ],
            "Productivity & Life Design": [
                "productivity tips",
                "how to change your life",
                "self improvement",
                "time management",
                "habits and routines",
                "discipline and motivation",
                "overthinking"
            ],
            "Creator Economy": [
                "youtube growth",
                "how to start a youtube channel",
                "creator business",
                "content creator",
                "personal brand",
                "solopreneur"
            ],
            "AI & Technology": [
                "ai for entrepreneurs",
                "ai productivity tools",
                "how to use ai for business",
                "ai workflow"
            ]
        }}

        Example 2 - Business Documentary Channel (MagnatesMedia style):
        {{
            "Corporate History": [
                "company rise and fall",
                "business empire",
                "corporate scandal",
                "brand failure",
                "startup bankruptcy",
                "company history"
            ],
            "Documentary Storytelling": [
                "business documentary",
                "cinematic documentary",
                "company story",
                "entrepreneur story",
                "business breakdown"
            ],
            "Business Analysis": [
                "business case study",
                "company analysis",
                "business strategy",
                "how companies failed",
                "business investigation"
            ]
        }}

        Example 3 - Student Productivity Channel (Thomas Frank style):
        {{
            "Study Techniques": [
                "study tips",
                "how to study better",
                "active recall",
                "spaced repetition",
                "exam preparation",
                "study strategies"
            ],
            "Productivity Systems": [
                "notion productivity",
                "time blocking",
                "second brain",
                "productivity system",
                "task management",
                "note taking"
            ],
            "Student Life": [
                "college productivity",
                "student morning routine",
                "study motivation",
                "productive student"
            ]
        }}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        VALIDATION (Check each keyword):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Before including a keyword, ask:
        1. ✓ Would real users type this into YouTube search?
        2. ✓ Is it 2-4 words maximum? (Shorter = better)
        3. ✓ Does it avoid unnecessary modifiers (strategies, techniques, journey, guide)?
        4. ✓ Is it natural/conversational, not academic?
        5. ✓ Will it surface similar channels, not just similar videos?

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        OUTPUT FORMAT:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Return ONLY valid JSON:
        - Keys: Category names (2-4 words, Title Case)
        - Values: Lists of 5-7 lowercase keyword phrases (2-4 words each)

        {{
            "Category Name": ["keyword one", "keyword two", "keyword three", "keyword four", "keyword five"]
        }}
        """

    # --- END PROMPT PLACEHOLDER ---


    keywords_categorized = {}
    retries = 3
    for attempt in range(retries):
        try:
            raw_response = ""
            current_model_provider = 'gemini' if 'gemini' in model_name.lower() else 'gpt'

            if current_model_provider == "gemini":
                if not gemini_model: print("❌ Gemini not initialized."); break
                # Ensure you are using a model capable of this, like gemini-1.5-pro
                # dynamic_cat_model = genai.GenerativeModel(model_name) 
                # generation_config = genai.GenerationConfig(response_mime_type="application/json", max_output_tokens=1000, temperature=0.2)
                # response = dynamic_cat_model.generate_content(prompt, generation_config=generation_config)
                # raw_response = response.text.strip()
                pass

            elif current_model_provider == "gpt":
                if not gpt_client: print("❌ GPT not initialized."); break
                response = gpt_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=1000,
                )
                raw_response = response.choices[0].message.content.strip()
            else: print(f"❌ Unknown model type derived from '{model_name}'."); break

            # --- Parse JSON ---
            try:
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
                json_str = json_match.group(1) if json_match else raw_response
                parsed_json = json.loads(json_str)

                if isinstance(parsed_json, dict) and all(isinstance(v, list) and all(isinstance(s, str) for s in v) for v in parsed_json.values()):
                     keywords_categorized = {k.strip(): [kw.strip().lower() for kw in v if kw.strip()]
                                             for k, v in parsed_json.items() if k.strip() and v}
                     if keywords_categorized:
                         print(f"✅ Got {sum(len(v) for v in keywords_categorized.values())} keywords across {len(keywords_categorized)} DYNAMIC categories.");
                         save_to_cache(cache_filename, keywords_categorized)
                         return keywords_categorized
                     else: print(f"⚠️ LLM returned valid JSON but no categories/keywords.")
                else: print(f"⚠️ LLM output was not a valid Dict[str, List[str]] structure: {raw_response[:100]}...")
            except Exception as parse_e: print(f"⚠️ Error parsing/validating LLM JSON (attempt {attempt+1}): {parse_e}")

        except Exception as e:
            print(f"❌ Error ({model_name} attempt {attempt+1}/{retries}): {str(e)[:100]}")
            if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e):
                print("💤 Rate limit hit, waiting 60s...")
                time.sleep(60)
            elif attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print("❌ All keyword extraction retries failed.")
                break

    save_to_cache(cache_filename, keywords_categorized)
    return keywords_categorized


# === Main Execution ===
def main():
    print("="*70)
    print(f"MANUAL DYNAMIC KEYWORD GENERATION TEST for: {TARGET_CHANNEL_NAME}")
    print(f"Loading data from: {INPUT_CSV_PATH.name}")
    print(f"Using Model for Keywords: {MODEL_FOR_KEYWORDS}")
    print("="*70)

    # --- Step 1: Load Video Data from CSV ---
    print("\nSTEP 1: Loading video data from CSV...")
    try:
        df_all_videos = pd.read_csv(INPUT_CSV_PATH)
        df_channel = df_all_videos[df_all_videos['Channel_Name'] == TARGET_CHANNEL_NAME].copy()
        if df_channel.empty: print(f"❌ ERROR: No data for '{TARGET_CHANNEL_NAME}'."); return
        
        # --- MODIFIED: Only need title ---
        required_cols = ['Channel_Name', 'title']
        if not all(col in df_channel.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df_channel.columns]; print(f"❌ ERROR: CSV missing: {missing}"); return

        # --- MODIFIED: Create list of titles ---
        video_titles = df_channel['title'].tolist()
        
        print(f"   Loaded {len(video_titles)} titles for {TARGET_CHANNEL_NAME}")

    except Exception as e: print(f"❌ ERROR loading CSV: {e}"); return

    # --- STEP 2: Extract DYNAMIC Categorized Keywords (uses cache) ---
    print("\nSTEP 2: Extracting DYNAMIC categorized keywords...")
    categorized_keywords = extract_dynamic_categorized_keywords_llm_cached(
        video_titles=video_titles, # Pass the list of titles
        channel_name=TARGET_CHANNEL_NAME,
        model_name=MODEL_FOR_KEYWORDS
    )
    keywords_output_path = CACHE_DIR.parent / f"{TARGET_CHANNEL_NAME}_dynamic_categorized_keywords_2.json"
    save_to_cache(keywords_output_path.stem, categorized_keywords)
    print(f"   Saved dynamically categorized keywords to {keywords_output_path.name}")

    # --- Final Summary ---
    print("\n" + "="*70)
    print("MANUAL DYNAMIC KEYWORD GENERATION COMPLETE")
    print("="*70)
    print(f"Seed Channel: {TARGET_CHANNEL_NAME}")
    # --- REMOVED Niche printout ---
    print("\nDynamically Generated Categories & Keywords:")
    if categorized_keywords:
        print(json.dumps(categorized_keywords, indent=2))
        total_kws = sum(len(v) for v in categorized_keywords.values())
        print(f"\nTotal Keywords: {total_kws}")
    else:
        print("   No keywords generated.")
    print(f"\nOutputs saved in: {CACHE_DIR.parent.name}")

if __name__ == "__main__":
    main()