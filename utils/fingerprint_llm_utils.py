import os
import re
import time
import google.generativeai as genai
from openai import OpenAI
import pandas as pd
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import json
import pprint
import numpy as np

# ============================================
# 1️⃣ Load Environment & API Keys
# ============================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


try:
    # Using a reliable, efficient model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence Transformer model loaded ('all-MiniLM-L6-v2').")
except Exception as e:
    print(f"❌ ERROR loading Sentence Transformer model: {e}")
    embedding_model = None


# ============================================
# 2️⃣ Configure Gemini (if key present)
# ============================================
gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # --- MODIFIED: Upped tokens slightly for niche/keyword prompts ---
        generation_config = genai.GenerationConfig(
            max_output_tokens=300, # Increased from 200
            temperature=0.1
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        gemini_model = genai.GenerativeModel(
            'gemini-2.0-flash-exp', # --- MODIFIED: Using gemini-pro for better reasoning
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        print("✅ Gemini model configured: gemini-pro")
    except Exception as e:
        print(f"❌ Gemini configuration failed: {e}")
else:
    print("⚠️ No GOOGLE_API_KEY found — Gemini disabled.")


# ============================================
# 3️⃣ Configure OpenAI GPT (if key present)
# ============================================
gpt_client = None
if OPENAI_API_KEY:
    try:
        gpt_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client configured.")
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
else:
    print("⚠️ No OPENAI_API_KEY found — GPT disabled.")


# ============================================
# 4️⃣ Utility: Text Cleaning
# ============================================
def preprocess_text_for_llm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_embedding_similarity_hybrid(keywords1: List[str], keywords2: List[str]) -> float:
    """
    Hybrid approach: Combines average similarity + max similarity.
    Most robust method.
    """
    if not embedding_model or not keywords1 or not keywords2:
        return 0.0
    
    try:
        embeddings1 = embedding_model.encode(keywords1)
        embeddings2 = embedding_model.encode(keywords2)
        
        # 1. Average embedding similarity (overall niche)
        avg_emb1 = embeddings1.mean(axis=0)
        avg_emb2 = embeddings2.mean(axis=0)
        avg_similarity = cosine_similarity(
            avg_emb1.reshape(1, -1),
            avg_emb2.reshape(1, -1)
        )[0][0]
        
        # 2. Maximum pairwise similarity (best keyword matches)
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        max_similarity = similarity_matrix.max()
        
        # 3. Weighted combination (70% average, 30% max)
        final_score = 0.7 * avg_similarity + 0.3 * max_similarity
        
        return round(float(final_score), 3)
        
    except Exception as e:
        print(f"  ❌ Embedding error: {e}")
        return 0.0
    
# --- Helper for the new prompt in niche finding ---
# def _get_keyword_sample(kw_dict: dict, max_sample=5) -> str:
#     """Gets a representative sample of keywords, one from each category."""
#     if not isinstance(kw_dict, dict): 
#         return "N/A"
#     sample = []
#     # Get first keyword from up to 5 categories
#     for cat, kws in kw_dict.items():
#         if kws and isinstance(kws, list) and kws[0]:
#             sample.append(kws[0])
#         if len(sample) >= max_sample:
#             break
#     return ", ".join(sample)

def _get_keyword_sample(kw_dict: dict, max_categories=3, keywords_per_category=3) -> str:
    """
    Gets top keywords from each category for LLM validation.
    
    Args:
        kw_dict: Dictionary of {category: [keywords]}
        max_categories: Maximum number of categories to sample (default: 3)
        keywords_per_category: Keywords to take from each category (default: 3)
    
    Returns:
        Formatted string: "cat1: kw1, kw2, kw3 | cat2: kw1, kw2, kw3"
        
    Example:
        Input: {
            "Business Scandals": ["corporate scandals", "business failures", "evil corporations"],
            "Success Stories": ["successful entrepreneurs", "startup success"]
        }
        Output: "Business Scandals: corporate scandals, business failures, evil corporations | Success Stories: successful entrepreneurs, startup success"
    """
    if not isinstance(kw_dict, dict) or not kw_dict:
        return "N/A"
    
    samples = []
    categories_processed = 0
    
    for category, keywords in kw_dict.items():
        # Stop if we've processed enough categories
        if categories_processed >= max_categories:
            break
        
        # Validate keywords list
        if not keywords or not isinstance(keywords, list):
            continue
        
        # Get top N keywords from this category
        top_keywords = [kw for kw in keywords[:keywords_per_category] if kw]
        
        if top_keywords:
            # Format: "Category: keyword1, keyword2, keyword3"
            category_sample = f"{category}: {', '.join(top_keywords)}"
            samples.append(category_sample)
            categories_processed += 1
    
    # Join categories with " | " separator
    return " | ".join(samples) if samples else "N/A"

# --- MODIFIED: Function updated to include Niche for context ---
def calculate_llm_similarity(
    seed_keywords: dict,        # <-- This is now a DICT
    candidate_keywords: dict,   # <-- This is now a DICT
    seed_channel_name: str,
    candidate_channel_name: str,
    seed_niche: str = "",       # <-- This is "Niche - Format"
    candidate_niche: str = "",  # <-- This is "Niche - Format"
    model_type: str = "gpt"
) -> float:
    """
    Uses LLM to directly score channel similarity, now using Niche as a key factor.
    Returns float 0.0-1.0
    """
    
    seed_categories = list(seed_keywords.keys()) if isinstance(seed_keywords, dict) else ["Unknown"]
    candidate_categories = list(candidate_keywords.keys()) if isinstance(candidate_keywords, dict) else ["Unknown"]
    
    seed_kw_sample = _get_keyword_sample(seed_keywords)
    candidate_kw_sample = _get_keyword_sample(candidate_keywords)

    # --- MODIFIED: Prompt now includes the niche ---
    prompt = f"""You are an expert YouTube channel analyst evaluating channel similarity for content discovery.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SEED CHANNEL: "{seed_channel_name}"
    Niche-Format: {seed_niche or "Unknown - Unknown"}
    Categories: {", ".join(seed_categories)}
    Keyword Sample: {seed_kw_sample}...

    CANDIDATE CHANNEL: "{candidate_channel_name}"
    Niche-Format: {candidate_niche or "Unknown - Unknown"}
    Categories: {", ".join(candidate_categories)}
    Keyword Sample: {candidate_kw_sample}...
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    TASK: Calculate similarity score using this weighted formula:
    **Final Score = (0.7 x Niche Similarity) + (0.3 x Keyword Similarity)**

    SCORING FRAMEWORK:

    STEP 1 - Evaluate NICHE Similarity (70% weight):
    Focus on the PRIMARY NICHE and INTENT (education vs storytelling vs motivation)

    NICHE MATCH LEVELS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1.0 = Identical niche AND intent
    Example: "Product management education" vs "Product management education"

    0.9 = Same niche, slightly different focus
    Example: "Product management education" vs "Tech career education"

    0.8 = Very similar niche, same intent
    Example: "Business documentaries" vs "Corporate history documentaries"

    0.7 = Related niche, same intent
    Example: "YouTube creator economy" vs "Content creator business strategies"

    0.6 = Related niche, different intent
    Example: "Product management education" vs "Product management podcasts"
    (Both PM, but education vs storytelling intent)

    0.5 = Overlapping topics, different niche
    Example: "Digital marketing education" vs "Social media creator tips"
    (Both use social media, but marketing vs creator economy)

    0.4 = Loosely related
    Example: "Business documentaries" vs "Entrepreneurship motivation"
    (Both business, but storytelling vs motivation)

    0.3 = Somewhat related topics
    Example: "Product management" vs "Software engineering"
    (Both tech careers, different roles)

    0.2 = Different industries, minimal overlap
    Example: "Tech reviews" vs "Personal finance"

    0.1 = Completely different niches
    Example: "Product management" vs "Gaming commentary"

    STEP 2 - Evaluate KEYWORD Similarity (30% weight):
    Compare the supporting keywords for semantic overlap

    KEYWORD MATCH LEVELS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1.0 = 80%+ keywords are synonyms or direct matches
    0.8 = 60-80% keywords overlap semantically
    0.6 = 40-60% keywords overlap or relate
    0.4 = 20-40% keywords share themes
    0.2 = <20% keyword overlap
    0.0 = No keyword overlap

    SPECIAL CASES:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    If seed_niche = "Unknown": Weight keywords more (60% keywords, 40% niche guess)
    If candidate_niche = "Unknown": Penalize by 0.1 (uncertainty penalty)
    If BOTH = "Unknown": Use 100% keyword similarity

    INTENT DISTINCTION (Critical for niche scoring):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Education/Career (teaching, how-to, tutorials):
    - "Product management education", "SEO strategies", "Coding tutorials"

    Storytelling/Analysis (documentaries, news, case studies):
    - "Business documentaries", "Geopolitical analysis", "Corporate history"

    Motivation/Mindset (inspiration, personal growth):
    - "Entrepreneurial mindset", "Personal development", "Leadership philosophy"

    CRITICAL RULE:
    Same topic + Different intent = MAX 0.6 similarity
    Example: "Content creation education" vs "Content creation documentaries"

    EXAMPLE CALCULATIONS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Case 1: Lenny's Podcast vs PM Dojo
    Niche: "Product management education" vs "Product management education"
    → Niche Score: 1.0 (identical)
    Keywords: "product strategy", "PM interviews" vs "product roadmaps", "PM careers"
    → Keyword Score: 0.9 (high overlap)
    Final: (0.7 × 1.0) + (0.3 × 0.9) = 0.97 ✅

    Case 2: Colin & Samir vs Neil Patel
    Niche: "Creator economy storytelling" vs "Digital marketing education"
    → Niche Score: 0.5 (different niches, some keyword overlap)
    Keywords: "youtube growth" vs "SEO strategies"
    → Keyword Score: 0.4 (some overlap on "content")
    Final: (0.7 × 0.5) + (0.3 × 0.4) = 0.47 ❌ (below 0.7 threshold)

    Case 3: Vox vs Johnny Harris
    Niche: "News analysis storytelling" vs "Geopolitical storytelling"
    → Niche Score: 0.8 (related storytelling, similar intent)
    Keywords: "investigative journalism" vs "documentary journalism"
    → Keyword Score: 0.85 (high overlap)
    Final: (0.7 × 0.8) + (0.3 × 0.85) = 0.82 ✅

    OUTPUT FORMAT:
    Return ONLY a decimal number between 0.0 and 1.0 (e.g., 0.82)
    No explanations, no text, just the number.
    """

    
    try:
        if model_type.lower() == "gemini":
            if not gemini_model:
                print("❌ Gemini not initialized for similarity.")
                return 0.0
            
            response = gemini_model.generate_content(prompt)
            result_text = response.text.strip()
            
        else:  # OpenAI
            if not gpt_client:
                 print("❌ GPT not initialized for similarity.")
                 return 0.0
            
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini", # Using your specified model
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result_text = response.choices[0].message.content.strip()
        
        # Extract number from response
        import re
        match = re.search(r'0?\.\d+|1\.0|0|1', result_text)
        
        if match:
            score = float(match.group())
            return round(min(max(score, 0.0), 1.0), 3)  # Clamp 0-1
        else:
            print(f"  ⚠️ LLM returned non-numeric: {result_text}")
            return 0.0
            
    except Exception as e:
        print(f"  ❌ LLM similarity error: {str(e)[:50]}")
        return 0.0


# ============================================
# NEW Niche-Format Extraction (niche+format)
# ============================================
def extract_niche_llm(
    channel_description: str,
    video_titles: List[str],
    channel_name: str,
    model_type: str = "gemini",
    max_chars: int = 2000
) -> str:
    """
    HYBRID APPROACH:
    1. Try channel description first.
    2. Fallback to video titles if description is poor.
    
    Returns a single string: "Primary Niche - Primary Format"
    """

    # Step 1: Check if description is good quality
    text_for_niche = ""
    text_source = ""
    if channel_description and len(channel_description) >= 100:
        if channel_description.count("http") / max(len(channel_description), 1) < 0.3:
            text_for_niche = channel_description[:max_chars]
            text_source = "channel description"
            print(f"  ℹ️ Using channel description for {channel_name}")
    
    # Step 2: Fallback to video titles
    if not text_for_niche and video_titles:
        text_for_niche = " | ".join(video_titles[:10])
        text_source = "video titles"
        print(f"  ⚠️ Using video titles fallback for {channel_name}")
    
    # Step 3: No data
    if not text_for_niche:
        print(f"  ❌ No data for {channel_name}, returning 'General - Unknown'")
        return "General - Unknown"
    
    truncated_text = preprocess_text_for_llm(text_for_niche)[:max_chars]
    
    # --- THIS IS THE NEW, STRICT PROMPT ---
    prompt = f"""You are an expert YouTube channel analyst.
    Analyze the following {text_source} for the channel "{channel_name}" and identify its single primary niche and single primary format.

    {text_source.upper()}:
    \"\"\"
    {truncated_text}
    \"\"\"

    TASK: You must identify TWO things:
    1.  **PRIMARY NICHE:** The channel's main TOPIC (e.g., "Productivity", "Business Case Studies", "Entrepreneurship", "Financial Education").
    2.  **PRIMARY FORMAT:** The channel's main STYLE (e.g., "Educational Tutorials", "Documentary", "Podcast/Interviews", "Talking-Head Analysis", "Vlog").

    CRITICAL RULE:
    - Distinguish "Educational Tutorials" (how-to guides) from "Documentary" (storytelling) from "Podcast/Interviews" (conversations).
    - "Ali Abdaal" is "Productivity - Educational Tutorials".
    - "The Diary Of A CEO" is "Entrepreneurship - Podcast/Interviews".
    - "MagnatesMedia" is "Business - Documentary".
    - "Johnny Harris" is "Geopolitics - Documentary".

    OUTPUT FORMAT:
    Return ONLY the Niche and Format separated by a hyphen.
    
    FORMAT: "Primary Niche - Primary Format"

    EXAMPLES:
    - "Productivity - Educational Tutorials"
    - "Business Case Studies - Documentary"
    - "Entrepreneurship - Podcast/Interviews"
    - "Tech Careers - Talking-Head Analysis"
    - "Financial Education - Educational Tutorials"
    - "Creator Economy - Podcast/Interviews"
    - "Geopolitics - Documentary"
    

    If multiple subtopics appear, choose the most dominant recurring one based on frequency or emphasis.
    Return only one concise answer in the format: “Niche - Format”. 
    Do not add explanations or any other text.
    """
    
    try:
        if model_type.lower() == "gemini":
            if not gemini_model: return "General - Unknown"
            response = gemini_model.generate_content(prompt)
            niche_format = response.text.strip().replace('"', '')
            
        elif model_type.lower() == "gpt":
            if not gpt_client: return "General - Unknown"
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini", # Use your specified model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )
            niche_format = response.choices[0].message.content.strip().replace('"', '')
            
        else:
            print(f"❌ Unknown model type '{model_type}'.")
            return "General - Unknown"
            
        # Validate the "Niche - Format" structure
        if " - " not in niche_format or len(niche_format) < 7:
            print(f"⚠️ LLM returned invalid format: '{niche_format}'. Defaulting.")
            # Try to salvage, or just default
            if niche_format:
                return f"{niche_format} - Unknown"
            return "General - Unknown"
            
        print(f"  ✅ Niche/Format for {channel_name}: {niche_format} (from {text_source})")
        time.sleep(2) # Keep your rate limit
        return niche_format

    except Exception as e:
        print(f"  ❌ LLM niche extraction error for {channel_name}: {str(e)[:50]}")
        return "General - Unknown"

# ============================================
# 6️⃣ LLM Keyword Extraction (Gemini / GPT Modular)
# ============================================
# --- MODIFIED: Function updated to accept 'niche' for context ---
def extract_keywords_llm(
    combined_text: str, 
    channel_name: str = "", 
    niche: str = "", # <-- ADDED
    model_type: str = "gemini", 
    max_chars: int = 40000
) -> Dict[str, List[str]]:
    """Extracts keywords using Gemini or GPT, now guided by the channel's niche."""
    if not combined_text:
        print("⚠️ No text provided.")
        return {}


    truncated_content = combined_text[:max_chars]
    print(f"📤 Sending {len(truncated_content)} chars to {model_type.upper()}...")
    channel_name_cleaned = channel_name.lower().strip() if channel_name else "[Channel Name Unavailable]"
    
    print(channel_name, "\n\n")
    
    # --- MODIFIED: Prompt now includes the niche ---
    prompt = f"""You are a YouTube SEO and competitor research analyst.
    Your task is to analyze the provided content for "{channel_name}" and identify its **2-3 most dominant content categories**.

    CHANNEL NICHE-FORMAT: {niche or "Unknown"}
    CONTENT TO ANALYZE:
    \"\"\"
    {truncated_content}
    \"\"\"

    CRITICAL TASK:
    1.  Identify ONLY the **2 or 3** most central, high-volume categories for this channel.
    2.  **IGNORE** minor or occasional topics (like "AI" for a business channel). Focus on the channel's "heartland" content.
    3.  For each of these 2-3 categories, extract **10-15 high-intent** YouTube search keywords that a user would type to find **other channels just like this one**.
    4.  Keywords must be 2-4 words. Avoid generic single words.

    ---
    REFERENCE EXAMPLES (MATCH THIS EXACT STYLE):
    ---

    Example 1 - Business Documentary Channel (Niche: Business - Documentary):
    (Notice: No "AI" or "Creator Economy" categories, as they are not the *major* focus)
    {{
      "Corporate History & Scandals": [
        "company rise and fall", "business empire", "corporate scandal", "brand failure",
        "startup bankruptcy", "company history", "how companies failed", 
        "business investigation", "corporate greed", "failed businesses",
        "business stories", "corporate analysis"
      ],
      "Business Documentaries": [
        "business documentary", "cinematic documentary", "company story", 
        "entrepreneur story", "business breakdown", "business case study",
        "corporate deep dive", "business analysis", "brand story", "ceo story",
        "economic documentary"
      ]
    }}

    Example 2 - Student Productivity Channel (Niche: Productivity - Educational Tutorials):
    {{
      "Productivity Systems": [
        "productivity tips", "time management", "habits and routines", "self discipline",
        "focus strategies", "overcoming procrastination", "how to wake up early",
        "productivity hacks", "self improvement", "deep work", "getting things done"
      ],
      "Learning & Study Skills": [
        "study tips", "how to learn faster", "active recall", "spaced repetition",
        "note taking methods", "how to read more", "best learning resources",
        "exam preparation", "how to organize", "study habits"
      ],
      "Productivity Tech & Tools": [
        "best productivity apps", "notion tutorial", "best apps for students",
        "task management apps", "digital organization", "tech for productivity",
        "notion setup", "best note taking apps", "obsidian tutorial"
      ]
    }}

    Example 3 - Educational Business Channel (Niche: Entrepreneurship - Educational Tutorials):
    {{
      "Business & Entrepreneurship": [
        "build a business", "lifestyle business", "boring business ideas",
        "entrepreneurship 2025", "make money online", "business ideas for beginners",
        "how to start a business", "small business ideas", "online business"
      ],
      "Financial Freedom": [
        "financial freedom", "get rich", "passive income", "how to make money",
        "wealth building", "millionaire habits", "money mindset", "investing for beginners"
      ]
    }}

    OUTPUT FORMAT:
    Return ONLY valid JSON with 2-3 keys (highle related categories) and 10-15 keywords per key.
    {{
      "Primary Category 1": ["keyword one", "keyword two", ...],
      "Primary Category 2": ["keyword one", "keyword two", ...]
    }}
    """
    
    retries = 3
    for attempt in range(retries):
        try:
            # ========== GEMINI MODE ==========
            if model_type.lower() == "gemini":
                if not gemini_model:
                    print("❌ Gemini not initialized.")
                    return []
                response = gemini_model.generate_content(prompt)
                if not response.parts:
                    print("⚠️ Gemini blocked or empty response.")
                    return []
                text_out = response.text.strip()


            # ========== GPT MODE ==========
            elif model_type.lower() == "gpt":
                if not gpt_client:
                    print("❌ GPT client not initialized.")
                    return []
                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a YouTube keyword expert."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1000,
                )
                text_out = response.choices[0].message.content.strip()


            else:
                print(f"❌ Unknown model type '{model_type}'. Use 'gemini' or 'gpt'.")
                return []


            # ========== Parse Keywords ==========
            # ========== Parse JSON Keywords ==========
            keywords_categorized = {} # Default to empty dict
            try:
                # Clean potential markdown wrappers
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", text_out, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text_out # Assume raw output is JSON

                parsed_json = json.loads(json_str)

                # Validate structure: Dict where values are lists of strings
                if isinstance(parsed_json, dict) and all(isinstance(v, list) and all(isinstance(s, str) for s in v) for v in parsed_json.values()):
                     # Clean keys and keywords
                     keywords_categorized = {k.strip(): [kw.strip().lower() for kw in v if kw.strip()]
                                             for k, v in parsed_json.items() if k.strip() and v} # Keep only non-empty categories/lists

                     if keywords_categorized:
                         total_kws = sum(len(v) for v in keywords_categorized.values())
                         print(f"✅ Parsed {total_kws} keywords across {len(keywords_categorized)} categories from {model_type.upper()}.")
                         # NO sleep needed here, already slept after API call potentially
                         return keywords_categorized # <-- Return the dictionary
                     else:
                         print(f"⚠️ LLM returned valid JSON but no categories/keywords.")
                         # Fall through to return empty dict outside try block if needed
                else:
                    print(f"⚠️ LLM output was not a valid Dict[str, List[str]] structure: {text_out[:100]}...")
                    # Fall through to retry or return empty

            except json.JSONDecodeError:
                print(f"⚠️ LLM output was not valid JSON (attempt {attempt+1}): {text_out[:100]}...")
                # Fall through to retry or return empty
            except Exception as parse_e:
                # Catch any other unexpected parsing errors
                print(f"⚠️ Error parsing/validating LLM JSON (attempt {attempt+1}): {parse_e}")
                # Fall through to retry or return empty

            # If parsing failed, keywords_categorized is still {}, loop will retry if possible


        except Exception as e:
            print(f"❌ Error ({model_type.upper()} attempt {attempt+1}/{retries}): {str(e)[:100]}")
            if "429" in str(e) or "quota" in str(e).lower():
                print("💤 Rate limit hit, waiting 60s...")
                time.sleep(60)
            elif attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print("❌ All retries failed.")
                return {}
    return {}


# ============================================
# 7️⃣ Channel Fingerprint Builder
# ============================================
# --- MODIFIED: Function updated to accept 'niche' for context ---
def create_channel_fingerprint_llm(
    video_df: pd.DataFrame, 
    channel_name: str = "", 
    niche: str = "", # <-- ADDED
    top_n: int = 15, 
    model_type: str = "gemini"
) -> Dict[str, List[str]]:
    """Creates channel fingerprint using either Gemini or GPT, guided by the niche."""
    if video_df.empty:
        print("⚠️ Empty DataFrame")
        return {}


    print(f"📊 Processing {len(video_df)} videos for {channel_name}...")


    combined_texts = []
    for _, row in video_df.iterrows():
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))
        clean_title = preprocess_text_for_llm(title)
        clean_desc = preprocess_text_for_llm(description[:200])
        if channel_name:
            clean_title = clean_title.replace(channel_name.lower(), '')
            clean_desc = clean_desc.replace(channel_name.lower(), '')
        combined_texts.append(clean_title * 2)  # weight titles higher
        combined_texts.append(clean_desc)


    full_text_blob = "\n---\n".join(filter(None, combined_texts))
    if not full_text_blob:
        print("⚠️ No text to analyze.")
        return {}

    # --- MODIFIED: Pass the niche to the keyword extractor ---
    keywords = extract_keywords_llm(
        full_text_blob, 
        channel_name=channel_name, 
        niche=niche,  # <-- Pass the niche here
        model_type=model_type
    )
    return keywords


# ============================================
# NEW FUNCTION: Language Detection 
# ============================================
def detect_channel_language_llm(
    channel_description: str, 
    video_titles: List[str], 
    channel_name: str, 
    model_type: str = "gemini",
    max_chars: int = 1500
) -> str:
    """
    Analyzes channel text to detect the primary language.
    Returns a 2-letter ISO 639-1 code (e.g., 'en', 'hi', 'es') or 'un' for unknown.
    """
    
    # Combine the most telling pieces of text
    text_blob = f"Channel Name: {channel_name}\n\n"
    text_blob += f"Description: {channel_description}\n\n"
    text_blob += "Recent Video Titles:\n- " + "\n- ".join(video_titles[:10])
    
    truncated_text = preprocess_text_for_llm(text_blob)[:max_chars]

    prompt = f"""You are an expert language detection system.
    Analyze the following text from a YouTube channel {channel_name}:

    \"\"\"
    {truncated_text}
    \"\"\"

    TASK: Identify the single **primary language** used in the text.
    - Respond with ONLY the two-letter ISO 639-1 code.
    - Examples: 'en' (English), 'hi' (Hindi), 'es' (Spanish), 'de' (German).
    - If the language is a mix (e.g., 'Hinglish'), return the code for the dominant spoken language ('hi').
    - If you are completely uncertain, return 'un'.

    OUTPUT:
    """

    try:
        if model_type.lower() == "gemini":
            if not gemini_model: return "un"
            # Use a config with fewer tokens for this simple task
            simple_config = genai.GenerationConfig(max_output_tokens=10, temperature=0.0)
            response = gemini_model.generate_content(prompt, generation_config=simple_config)
            lang_code = response.text.strip().lower()
            
        elif model_type.lower() == "gpt":
            if not gpt_client: return "un"
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini", # gpt-3.5-turbo could also work here
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            lang_code = response.choices[0].message.content.strip().lower()
            
        else:
            return "un"

        # Basic validation of the 2-letter code
        if len(lang_code) == 2 and re.match(r'^[a-z]{2}$', lang_code):
            return lang_code
        else:
            print(f"  ⚠️  Language detector returned invalid code: {lang_code}")
            return "un"

    except Exception as e:
        print(f"  ❌ LLM language detection error: {str(e)[:50]}")
        return "un"


# (Make sure json, re, time, and gpt_client are available)

# This is a simple helper just to trim the long keyword string for the prompt
def _get_keyword_sample_from_string(kw_string: str, max_sample=20) -> str:
    """Gets a representative sample of keywords from a flat string."""
    if not isinstance(kw_string, str): 
        return "N/A"
    kws = [k.strip() for k in kw_string.split(',') if k.strip()]
    return ", ".join(kws[:max_sample])

# ============================================
# NEW: Final "Extra Call" Competitor Check (v5 - Raw Data Only)
# ============================================
def is_direct_competitor_llm_final_check(
    seed_name: str,
    seed_keywords_str: str,     # <-- NEW: Pass seed's flat keyword string
    candidate_name: str,
    candidate_description: str,
    candidate_keywords_str: str, # <-- Pass candidate's flat keyword string
    model_provider: str = "gpt",
    retries: int = 2
) -> dict:
    """
    Uses GPT-4o-mini for a final, strict "Yes/No" check.
    This version IGNORES Niche/Format labels and infers from raw data.
    """

    # Get keyword samples from the flat strings
    seed_kw_sample = _get_keyword_sample_from_string(seed_keywords_str, max_sample=20)
    cand_kw_sample = _get_keyword_sample_from_string(candidate_keywords_str, max_sample=20)
    cand_desc_snippet = (candidate_description[:1000] if candidate_description else "N/A")

    prompt = f"""You are an expert YouTube analyst. Your job is to make a final "Yes" or "No" decision on whether two channels are DIRECT content competitors.
    
    **CRITICAL INSTRUCTION: IGNORE any Niche/Format labels. Infer EVERYTHING from the raw data provided below.**

    SEED CHANNEL (RAW DATA):
    - Name: "{seed_name}"
    - Keyword Sample: "{seed_kw_sample}"

    CANDIDATE CHANNEL (RAW DATA):
    - Name: "{candidate_name}"
    - Description: "{cand_desc_snippet}..."
    - Keyword List: "{cand_kw_sample}"

    YOUR TASK:
    1.  **Infer** the Seed's true Format (e.g., Documentary, Podcast, Tutorial) and Niche (e.g., Business, Productivity) from its Name and Keyword Sample.
    2.  **Infer** the Candidate's true Format and Niche from its Name, Description, and Keyword List.
    3.  **Compare** your *inferred* data. Are they direct competitors (same format/intent, similar niche)?

    DEFINITION: Direct competitors create content on very similar TOPICS using the same primary FORMAT/INTENT. Would a typical viewer of the SEED channel subscribe to the CANDIDATE channel because it serves the exact same need?

    CRITICAL EVALUATION (Answer YES only if BOTH are true):

    1. FORMAT MATCH? (Primary Check - Must be identical or extremely similar)
       - "Educational Tutorials" vs "Educational Tutorials" = YES
       - "Documentary" vs "Documentary" = YES
       - "Educational Tutorials" vs "Talking-Head Analysis" = YES (Similar Intent)
       ----------------------------------------------------
       - "Educational Tutorials" vs "Podcast/Interviews" = NO (Different Intent)
       - "Documentary" vs "Educational Tutorials" = NO (Different Intent)

    2. NICHE (TOPIC) MATCH? (Secondary Check - Must be highly relevant)
       - "Business Case Studies" vs "Corporate History" = YES (High Relevance)
       - "Productivity" vs "Study Skills" = YES (High Relevance)
       ----------------------------------------------------
       - "Business" vs "Personal Finance" = NO (Related, but Different Focus)

    EXAMPLES (How you should think):

    Seed: "Ali Abdaal" (Keywords: "productivity tips, notion, study hacks, self improvement")
    Cand: "The Diary Of A CEO" (Desc: "host of the #1 podcast...", Keywords: "podcast, interview, mindset")
    Inferred Seed Format: Educational Tutorials
    Inferred Cand Format: Podcast/Interviews
    Decision: NO (Format mismatch is critical)

    Seed: "MagnatesMedia" (Keywords: "business documentary, company rise and fall, corporate scandal")
    Cand: "Company Man" (Desc: "business documentaries...", Keywords: "company analysis, business stories, documentary")
    Inferred Seed Format: Documentary
    Inferred Cand Format: Documentary
    Decision: YES (Perfect match)

    Seed: "MagnatesMedia" (Keywords: "business documentary, corporate scandal")
    Cand: "Jake Tran" (Desc: "documentaries on money and power", Keywords: "geopolitics, business, war, history")
    Inferred Seed Format: Business Documentary
    Inferred Cand Format: Geopolitics/Business Documentary
    Decision: YES (Format matches, Niches are related "big picture" analysis)

    **Output Format** (Return ONLY a single, valid JSON object):
    {{
      "is_competitor": [true/false],
      "confidence": "[High/Medium/Low]",
      "reason": "[One sentence explaining your *inferred* format and niche match/mismatch]"
    }}
    """

    for attempt in range(retries):
        try:
            if model_provider.lower() == "gpt":
                if not gpt_client:
                    print("  ❌ GPT client not initialized for final check.")
                    return {"is_competitor": False, "confidence": "Low", "reason": "GPT client not initialized."}
                
                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini", # Use the cheap mini model
                    response_format={"type": "json_object"}, # Force JSON output
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                result_text = response.choices[0].message.content.strip()
                
                try:
                    result_json = json.loads(result_text)
                    if 'is_competitor' in result_json:
                        print(f"     ✅ LLM Final Check: {'Yes' if result_json['is_competitor'] else 'No'}. Reason: {result_json.get('reason', 'N/A')}")
                        return result_json
                except json.JSONDecodeError:
                    print(f"  ⚠️ LLM Final Check returned invalid JSON: '{result_text}' (Attempt {attempt+1})")

            else:
                 print(f"  ❌ Unknown model provider '{model_provider}'")
                 return {"is_competitor": False, "confidence": "Low", "reason": "Unknown model provider."}

        except Exception as e:
            print(f"  ❌ LLM Final Check Error (Attempt {attempt+1}/{retries}): {str(e)[:100]}")
            time.sleep(5 * (attempt + 1))
            
    return {"is_competitor": False, "confidence": "Low", "reason": "All API retries failed."}

# ============================================
# NEW: "One-Shot" Fingerprint Function
# (Replaces extract_niche_llm AND create_channel_fingerprint_llm)
# ============================================
# Make sure 'gpt_client' is initialized at the top of your file
# (e.g., gpt_client = OpenAI(api_key=OPENAI_API_KEY))


def _get_openai_embedding(text_list: list, model="text-embedding-3-small") -> list:
    """
    Helper to get embeddings from OpenAI.
    Returns list of embedding vectors.
    """
    if not text_list:
        return []
    
    # Clean input
    text_list = [str(text).strip() for text in text_list if str(text).strip()]
    if not text_list:
        return []
        
    try:
        response = gpt_client.embeddings.create(input=text_list, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"  ❌ OpenAI Embedding Error: {e}")
        return []


def calculate_keyword_score_openai(
    seed_keywords: dict, 
    candidate_keywords: dict,
    max_keywords_per_channel: int = 50,
    verbose: bool = False) -> float:
    """
    OPTIMIZED Keyword Similarity Score using OpenAI embeddings.
    
    Features:
    - Single batched API call (2x faster)
    - Token limit safety
    - Optional verbose logging
    - Averages all keywords into semantic vector
    
    Returns: 0.0 - 1.0
    """
    if not gpt_client:
        if verbose:
            print("  ❌ OpenAI client not initialized.")
        return 0.0
        
    if not seed_keywords or not candidate_keywords:
        return 0.0

    try:
        # --- 1. Flatten Keywords ---
        seed_kw_list = [kw for kws in seed_keywords.values() for kw in kws]
        cand_kw_list = [kw for kws in candidate_keywords.values() for kw in kws]

        # Safety: Limit keywords to avoid token overflow
        seed_kw_list = seed_kw_list[:max_keywords_per_channel]
        cand_kw_list = cand_kw_list[:max_keywords_per_channel]

        if verbose:
            print(f"  📊 Seed: {len(seed_kw_list)} keywords | Candidate: {len(cand_kw_list)} keywords")

        if not seed_kw_list or not cand_kw_list:
            return 0.0

        # --- 2. Batch Embedding (Single API Call) ---
        all_keywords = seed_kw_list + cand_kw_list
        all_vectors = _get_openai_embedding(all_keywords)
        
        if not all_vectors or len(all_vectors) != len(all_keywords):
            if verbose:
                print("  ⚠️ Embedding generation failed.")
            return 0.0

        # Split vectors back
        seed_count = len(seed_kw_list)
        seed_vectors = all_vectors[:seed_count]
        cand_vectors = all_vectors[seed_count:]

        # --- 3. Average Vectors ---
        avg_seed_vec = np.mean(seed_vectors, axis=0)
        avg_cand_vec = np.mean(cand_vectors, axis=0)

        # --- 4. Cosine Similarity ---
        final_score = cosine_similarity(
            avg_seed_vec.reshape(1, -1),
            avg_cand_vec.reshape(1, -1)
        )[0][0]
        
        if verbose:
            print(f"  ✅ Keyword score: {final_score:.3f}")
        
        return round(float(final_score), 3)

    except Exception as e:
        print(f"  ❌ Error in keyword scoring: {str(e)[:100]}")
        return 0.0



def get_channel_fingerprint_oneshot(
    channel_name: str,
    channel_description: str,
    video_df: pd.DataFrame, # Pass in the DataFrame of 20 videos
    model_provider: str = "gpt",
    max_chars: int = 40000,
    retries: int = 3
) -> dict:
    """
    Performs a single, "one-shot" LLM call to extract BOTH the
    detailed channel profile and the focused SEO keywords.
    (Version 2: Includes fix for nan/float values and ad detection)
    """
    
    # --- 1. Combine all text for context ---
    
    # --- FIX for nan/float in channel_description ---
    safe_channel_desc = str(channel_description) if pd.notna(channel_description) else "N/A - No description provided"
    
    combined_text = f"CHANNEL NAME: {channel_name}\n"
    combined_text += f"CHANNEL DESCRIPTION:\n{safe_channel_desc}\n\n"
    
    # --- FIX for nan/float in video_titles ---
    video_titles = video_df['title'].tolist()
    safe_titles = [str(t) for t in video_titles if pd.notna(t)] # Convert all valid titles to string
    combined_text += "--- RECENT VIDEO TITLES (Sample) ---\n"
    combined_text += "\n".join(safe_titles) + "\n\n"
    
    # --- FIX for nan/float in video_descs + Ad Detection ---
    video_descs = video_df['description'].tolist()
    safe_descs = [str(d) for d in video_descs if pd.notna(d) and isinstance(d, str)]
    
    # Heuristic: If > 70% of descriptions start with "http" or "Go to", they are ads.
    ad_count = 0
    for d in safe_descs:
        d_low = d.lower()
        if d_low.startswith("http") or d_low.startswith("go to") or "tryfum.com" in d_low or "buyraycon.com" in d_low:
            ad_count += 1
            
    if safe_descs and (ad_count / len(safe_descs)) > 0.7:
        print("  ⚠️  CONTEXT DETECTED: Video descriptions are sponsor ads. Telling LLM to IGNORE them.")
        combined_text += "--- RECENT VIDEO DESCRIPTIONS (Sample) ---\n"
        combined_text += "[Video descriptions are all sponsor ads and have been ignored]\n"
        # We will also add this instruction to the main prompt
    else:
        # If they are not ads, add them.
        combined_text += "--- RECENT VIDEO DESCRIPTIONS (Sample) ---\n"
        for i, desc in enumerate(video_descs):
            # This is the simple fix: convert to string first, THEN slice.
            safe_desc_str = str(desc)
            if safe_desc_str.lower() == 'nan':
                safe_desc_str = "[No Description]"
            combined_text += f"Video {i+1} Desc: {safe_desc_str[:300]}...\n" 
                
    truncated_content = combined_text
    # truncated_content = combined_text[:max_chars]
    print(f"📤 Sending {len(truncated_content)} chars to {model_provider.upper()} for one-shot analysis...")
    
    # --- 2. The New "Master" Prompt (Now with Title-Focus) ---
    prompt = f"""You are an expert YouTube channel analyst.
    Analyze the provided raw data (channel name, description, video titles, video descriptions) 
    and extract a complete channel profile and its SEO keywords.

    RAW DATA TO ANALYZE:
    \"\"\"
    {truncated_content}
    \"\"\"

    TASK: Return a single, valid JSON object with two top-level keys: "profile" and "keywords".

    ---
    PART 1: "profile"
    ---
    This key must contain an object with these 6 sub-keys:
    1.  "niche": The channel's primary TOPIC (e.g., "Productivity", "Business Case Studies", "Scam Investigation").
    2.  "format": The primary STYLE (e.g., "Educational Tutorial", "Explainer Documentary", "Video Essay", "Podcast/Interviews", "Talking-Head Analysis").
    3.  "intent": The channel's main GOAL. Must be one of: ["To Explain", "To Persuade", "To Report News", "To Entertain", "To Educate (Tutorial)", "To Inspire"].
    4.  "speaker": The primary point of view. Must be one of: ["Solo Creator", "Brand/Corporation", "Media Company", "Anonymous"].
    5.  "ideology": The channel's political bias. Must be one of: ["Progressive/Left", "Conservative/Right", "Libertarian", "Neutral/Academic"].
    6.  "target_audience": The primary demographic (e.g., "Curious Learners", "Political Activists", "Students", "Entrepreneurs").

    ### CRITICAL RULE ###
        - Do NOT default to "N/A" for any of them.
    ---
    PART 2: "keywords"
    ---
    Generate **2-3 most dominant** content categories with 10-15 high-intent keywords each.
    **CRITICAL**: Prioritize EMOTIONAL and ACTION-ORIENTED keywords over generic topic keywords.
    **CRITICAL**: "niche" and "format" are very crucial. So analyze the data of {channel_name} very deeply and then answer.
    **CRITICAL**: "intent" of channel is very very important. So please rethink and reanalyze the "intent" that is it accurate and then answer.

    Examples of GOOD keywords: 
    ✅ "exposed as fraud" (emotional + action)
    ✅ "conspiracy revealed" (emotional + action)
    ✅ "company destroyed" (emotional + action)

    Examples of BAD keywords:
    ❌ "business news" (too generic)
    ❌ "political analysis" (too generic)

    ---
    EXAMPLE OUTPUT (Do not copy it, use it to learn.):
    {{
    "profile": {{
        "niche": "Political & Cultural Commentary",
        "format": "Talking-Head Analysis (High Production)",
        "intent": "Controversy & Exposé",
        "speaker": "Solo Creator",
        "ideology": "Progressive-leaning",
        "target_audience": "Young adults (18-35), progressive, internet-native"
    }},
    "keywords": {{
        "Controversial Exposés": [
        "celebrity exposed", "hollywood exposed", "everything wrong with",
        "exposed as fraud", "exposed as monster", "career destroyed",
        "brutal truth exposed", "fake celebrity exposed", "celebrity lies exposed",
        "controversial documentary"
        ],
        "Conspiracy & Government Critique": [
        "government conspiracy", "secret plot exposed", "government corruption",
        "political conspiracy", "deep state exposed", "government lies",
        "conspiracy documentary", "hidden truth exposed", "shadow government",
        "corruption exposed"
        ]
    }}
    }}

    --- (End of Example) ---

    OUTPUT:
    Return ONLY the valid JSON for the channel in the "RAW DATA" section.
    """
    
    for attempt in range(retries):
        try:
            # ========== GPT MODE ==========
            if model_provider.lower() == "gpt":
                if not gpt_client:
                    print("❌ GPT client not initialized.")
                    return {}
                
                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini", # Use the cheap mini model
                    response_format={"type": "json_object"}, # Force JSON
                    messages=[
                        {"role": "system", "content": "You are a YouTube channel analyst outputting JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000 # Allow for larger JSON output
                )
                result_text = response.choices[0].message.content.strip()

            # ========== (Add Gemini mode here if needed) ==========
            else:
                print(f"❌ Unknown model provider '{model_provider}'.")
                return {}

            # ========== Parse JSON Output ==========
            try:
                parsed_json = json.loads(result_text)
                
                # Validate the complex structure
                if "profile" in parsed_json and "keywords" in parsed_json and \
                   isinstance(parsed_json["profile"], dict) and \
                   isinstance(parsed_json["keywords"], dict) and \
                   "niche" in parsed_json["profile"]:
                    
                    print(f"✅ One-shot analysis successful for {channel_name}.")
                    return parsed_json # Return the full JSON object
                else:
                    print(f"⚠️ LLM returned invalid JSON structure: {result_text[:100]}... (Attempt {attempt+1})")

            except json.JSONDecodeError:
                print(f"⚠️ LLM output was not valid JSON: {result_text[:100]}... (Attempt {attempt+1})")
                
        except Exception as e:
            print(f"❌ LLM One-Shot Error (Attempt {attempt+1}/{retries}): {str(e)[:100]}")
            time.sleep(5 * (attempt + 1))
            
    print(f"❌ All retries failed for {channel_name}.")
    return {} # Return empty dict if all retries fail

def calculate_profile_score_llm(
    seed_profile: dict,
    candidate_profile: dict,
    seed_keywords: dict,
    candidate_keywords: dict,
    seed_channel_name: str,
    candidate_channel_name: str,
    model_provider: str = "gpt") -> float:
    """
    OPTIMIZED Profile Score with Target Audience.
    
    Scoring cascade:
    1. Ideology Filter (HARD)
    2. Format/Intent Filter (HARD)
    3. Niche + Audience Similarity (SOFT)
    4. Speaker Penalty (MINOR)
    """
    
    # --- Extract profile fields ---
    s_niche = seed_profile.get("niche", "Unknown")
    s_format = seed_profile.get("format", "Unknown")
    s_intent = seed_profile.get("intent", "Unknown")
    s_speaker = seed_profile.get("speaker", "Unknown")
    s_ideology = seed_profile.get("ideology", "N/A")
    s_audience = seed_profile.get("target_audience", "General Audience")

    c_niche = candidate_profile.get("niche", "Unknown")
    c_format = candidate_profile.get("format", "Unknown")
    c_intent = candidate_profile.get("intent", "Unknown")
    c_speaker = candidate_profile.get("speaker", "Unknown")
    c_ideology = candidate_profile.get("ideology", "N/A")
    c_audience = candidate_profile.get("target_audience", "General Audience")

    pprint.pprint(candidate_keywords)

    pprint.pprint(seed_keywords)
    
    prompt = f"""You are an expert YouTube channel analyst evaluating channel similarity for content discovery.
        Calculate a similarity score (0.0 to 1.0) between these two channels based ONLY on their profiles.

        SEED CHANNEL: "{seed_channel_name}"
        Profile: {{
        "niche": "{s_niche}",
        "format": "{s_format}",
        "intent": "{s_intent}",
        "speaker": "{s_speaker}",
        "ideology": "{s_ideology}",
        "target_audience": "{s_audience}"
        }}

        CANDIDATE CHANNEL: "{candidate_channel_name}"
        Profile: {{
        "niche": "{c_niche}",
        "format": "{c_format}",
        "intent": "{c_intent}",
        "speaker": "{c_speaker}",
        "ideology": "{c_ideology}",
        "target_audience": "{c_audience}"
        }}

        **SCORING RULES (Apply in order):**

        1. **IDEOLOGY FILTER (CRITICAL - Can cause instant rejection):**
            - Direct opposites (Progressive/Left ↔ Conservative/Right): **SCORE 0.1**. Stop.
            - One political, one N/A (e.g., Political ↔ N/A): **MAX SCORE 0.35**. Proceed but cap at 0.35.
            - Compatible or both non-political: Proceed normally.
            
            Examples:
            - Channel A (Progressive) vs Channel B (Conservative): 0.1 
            - Channel A (Progressive) vs Channel B (N/A): Max 0.35 
            - Channel A (Libertarian) vs Channel B (Progressive): Compatible 

        2. **FORMAT/INTENT COMPATIBILITY (40% of final score):**
            - Identical formats: 1.0
            - Highly compatible (Video Essay ↔ Explainer Documentary): 0.9
            - Compatible (Documentary ↔ Podcast/Interviews): 0.7
            - Partially compatible (Educational Tutorial ↔ Talking-Head): 0.5
            - Incompatible (Vlog ↔ Educational Tutorial): 0.2
            
            Intent compatibility:
            - "To Persuade" ↔ "To Explain": 0.8 (compatible)
            - "To Explain" ↔ "To Entertain": 0.4 (less compatible)

        3. **NICHE SIMILARITY (40% of final score):**
            - Identical niche: 1.0
            - Very similar (Political Commentary ↔ Social Commentary): 0.9
            - Similar (Business Analysis ↔ Economic Analysis): 0.8
            - Loosely related (Tech ↔ Business): 0.5
            - Unrelated (Cooking ↔ Fitness): 0.1

        4. **TARGET AUDIENCE OVERLAP (15% of final score):**
            - Identical or highly overlapping audiences: 1.0
            - Partially overlapping (Students ↔ Young Professionals): 0.7
            - Different but compatible (Curious Learners ↔ Critical Thinkers): 0.8
            - Very different (Students ↔ Retirees): 0.3
            
            Examples:
            - "Curious Learners" ↔ "Socially Conscious Individuals": 0.8 
            - "Students" ↔ "Entrepreneurs": 0.5 

        5. **SPEAKER TYPE ADJUSTMENT (5% penalty if mismatch):**
            - Solo Creator ↔ Solo Creator: No penalty
            - Media Company ↔ Media Company: No penalty
            - Solo ↔ Media Company: -0.05 penalty (minor!)
            - Solo ↔ Brand/Corporation: -0.10 penalty
            - Anonymous ↔ Known personality: No penalty

        **CALCULATION:**
        - Start with base score from Format/Intent (40%) + Niche (40%) + Audience (15%)
        - Apply Speaker penalty (if any)
        - Cap at MAX SCORE from ideology filter (if applicable)

        **EXAMPLES:**
        - Channel A (Libertarian, Video Essay, Persuade, Societal Critique, Skeptics) 
        vs Channel B (Progressive, Explainer Doc, Explain, Political Commentary, Curious Learners):
        → Format: 0.9, Intent: 0.8 → Format/Intent: 0.85
        → Niche: 0.9
        → Audience: 0.75 (Skeptics vs Curious = compatible)
        → Speaker: -0.05 (Anonymous vs Media)
        → Final: (0.85 × 0.4) + (0.9 × 0.4) + (0.75 × 0.15) - 0.05 = 0.76 

        - Channel A (N/A, Educational Tutorial, Educate, Productivity, Students)
        vs Channel B (N/A, Motivational Talks, Inspire, Self-Help, Executives):
        → Niche: 0.6 (Productivity vs Self-Help = related)
        → Audience: 0.3 (Students vs Executives = very different!)
        → Final: ~0.45 

        Return ONLY a decimal number (e.g., 0.76). No explanation.
    """
    
    try:
        if model_provider.lower() == "gpt":
            if not gpt_client:
                 print("  ❌ GPT client not initialized.")
                 return 0.0
            
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            result_text = response.choices[0].message.content.strip()
        else:
            print(f"  ❌ Unknown model provider.")
            return 0.0

        # Extract score
        match = re.search(r'0?\.\d+|1\.0', result_text)
        
        if match:
            score = float(match.group())
            return round(min(max(score, 0.0), 1.0), 3)
        else:
            print(f"  ⚠️ Non-numeric result: {result_text}")
            return 0.0
            
    except Exception as e:
        print(f"  ❌ Profile score error: {str(e)[:100]}")
        return 0.0
