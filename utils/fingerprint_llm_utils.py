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
def _get_keyword_sample(kw_dict: dict, max_sample=5) -> str:
    """Gets a representative sample of keywords, one from each category."""
    if not isinstance(kw_dict, dict): 
        return "N/A"
    sample = []
    # Get first keyword from up to 5 categories
    for cat, kws in kw_dict.items():
        if kws and isinstance(kws, list) and kws[0]:
            sample.append(kws[0])
        if len(sample) >= max_sample:
            break
    return ", ".join(sample)

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
                model="gpt-4o", # Using your specified model
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


# def extract_niche_llm(
#     channel_description: str, 
#     video_titles: List[str],  # <-- ADD THIS
#     channel_name: str, 
#     model_type: str = "gemini", 
#     max_chars: int = 2000
# ) -> str:
#     """
#     HYBRID APPROACH: 
#     1. Try channel description first (90% of cases)
#     2. Fallback to video titles (10% of cases)
#     """
    
#     # Step 1: Check if description is good quality
#     text_for_niche = ""
#     text_source = ""
    
#     if channel_description and len(channel_description) >= 100:
#         # Check if not mostly URLs
#         url_count = channel_description.count("http")
#         if url_count / max(len(channel_description), 1) < 0.3:  # Less than 30% URLs
#             text_for_niche = channel_description[:max_chars]
#             text_source = "channel description"
#             print(f"  ℹ️ Using channel description for {channel_name}")
    
#     # Step 2: Fallback to video titles if description is poor
#     if not text_for_niche and video_titles:
#         # Concatenate top 10 video titles
#         text_for_niche = " | ".join(video_titles[:10])
#         text_source = "video titles"
#         print(f"  ⚠️ Using video titles fallback for {channel_name}")
    
#     # Step 3: No data available
#     if not text_for_niche:
#         print(f"  ❌ No data for {channel_name}, returning 'General'")
#         return "General"
    
#     # Preprocess
#     truncated_text = preprocess_text_for_llm(text_for_niche)[:max_chars]
    
#     # Updated prompt that mentions the source
#     prompt = f"""You are an expert YouTube channel analyst.
#         Your task is to analyze the following {text_source} for the channel "{channel_name}" and identify its single primary niche.

#         {text_source.upper()}:
#         \"\"\"
#         {truncated_text}
#         \"\"\"

#         TASK: Identify the channel's **primary niche** in a single, concise phrase (3-7 words).
#         The niche must describe the **content's PURPOSE and TOPIC** for a viewer.

#         **CRITICAL RULE: Distinguish the channel's INTENT.**
#         - Is it **"Career / Education"** (teaching a skill, 'how to', job prep)?
#         - Is it **"Storytelling / Analysis"** (documentaries, case studies, news, entertainment)?
#         - Is it **"Mindset / Motivation"** (self-improvement, leadership advice)?

#         **EXAMPLES OF GOOD NICHES (Note the INTENT):**

#         # Example 1: The "Business Analysis" Problem
#         - **GOOD Niche:** "Business analysis career prep" (This is Career/Education)
#         - **GOOD Niche:** "Business documentaries and case studies" (This is Storytelling/Analysis)
#         - **BAD Niche:** "Business analysis" (This is too vague)

#         # Example 2: The "Creator" Problem
#         - **GOOD Niche:** "Creator economy news and interviews" (This is Storytelling/Analysis)
#         - **GOOD Niche:** "YouTube growth tips and tutorials" (This is Career/Education)

#         # Example 3: Other Good Niches
#         - "Product management and tech careers" (Career/Education)
#         - "Consumer tech reviews and unboxings" (Storytelling/Analysis)
#         - "Entrepreneurial mindset and leadership" (Mindset/Motivation)

#         Return ONLY the single niche phrase, nothing else.
#         """
    
#     try:
#         if model_type.lower() == "gemini":
#             if not gemini_model:
#                 print("❌ Gemini not initialized for niche.")
#                 return "General"
#             response = gemini_model.generate_content(prompt)
#             niche = response.text.strip().replace('"', '')
            
#         elif model_type.lower() == "gpt":
#             if not gpt_client:
#                 print("❌ GPT not initialized for niche.")
#                 return "General"
#             response = gpt_client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.1,
#                 max_tokens=50,
#             )
#             niche = response.choices[0].message.content.strip().replace('"', '')
            
#         else:
#             print(f"❌ Unknown model type '{model_type}'.")
#             return "General"
            
#         if not niche:
#             print(f"⚠️ LLM returned empty niche for {channel_name}")
#             return "General"
            
#         print(f"  ✅ Niche for {channel_name}: {niche} (from {text_source})")
#         time.sleep(2)
#         return niche

#     except Exception as e:
#         print(f"  ❌ LLM niche extraction error for {channel_name}: {str(e)[:50]}")
#         return "General"

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
                model="gpt-4o", # Use your specified model
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
    
    # --- MODIFIED: Prompt now includes the niche ---
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
                    model="gpt-4o",
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
                model="gpt-4o", # gpt-3.5-turbo could also work here
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
    



import time
import re
import json
# Add any other necessary imports if not already present at the top
# (like genai, OpenAI client etc.)

# ... (keep all your existing functions like extract_niche_llm, calculate_llm_similarity, etc.) ...

# ============================================
# NEW: Final Competitor Check LLM Call
# ============================================
def is_direct_competitor_llm(
    seed_name: str,
    seed_niche_format: str, # e.g., "Productivity - Educational Tutorials"
    candidate_name: str,
    candidate_niche_format: str, # e.g., "Entrepreneurship - Podcast/Interviews"
    model_provider: str = "gpt", # Ensure this matches your main script config
    retries: int = 2
) -> str:
    """
    Uses GPT-4o-mini for a final check: Are these channels direct competitors
    based *primarily* on Niche (Topic) and Format (Intent)?

    Returns: "Yes" or "No" (or "Error" on failure)
    """

    # Basic check for valid inputs
    if not seed_niche_format or not candidate_niche_format or " - " not in seed_niche_format or " - " not in candidate_niche_format:
        print(f"  ⚠️ Invalid Niche-Format input for competitor check ({seed_name} vs {candidate_name}). Skipping.")
        return "Error"

    seed_parts = seed_niche_format.split(" - ", 1)
    cand_parts = candidate_niche_format.split(" - ", 1)
    seed_niche = seed_parts[0]
    seed_format = seed_parts[1]
    cand_niche = cand_parts[0]
    cand_format = cand_parts[1]

    prompt = f"""You are an expert YouTube analyst determining if two channels are DIRECT content competitors.

    DEFINITION: Direct competitors create content on very similar TOPICS using the same primary FORMAT/INTENT. Would a typical viewer of the SEED channel likely subscribe to the CANDIDATE channel because the content serves the exact same need?

    SEED CHANNEL: "{seed_name}"
    - Primary Niche (Topic): "{seed_niche}"
    - Primary Format (Intent): "{seed_format}"

    CANDIDATE CHANNEL: "{candidate_name}"
    - Primary Niche (Topic): "{cand_niche}"
    - Primary Format (Intent): "{cand_format}"

    CRITICAL EVALUATION (Answer YES only if BOTH are true):

    1. FORMAT MATCH? (Primary Check - Must be identical or extremely similar)
       - "Educational Tutorials" vs "Educational Tutorials" = YES
       - "Documentary" vs "Documentary" = YES
       - "Podcast/Interviews" vs "Podcast/Interviews" = YES
       - "Educational Tutorials" vs "Talking-Head Analysis" = YES (Similar Intent)
       - "Documentary" vs "Video Essay" = YES (Similar Intent)
       ----------------------------------------------------
       - "Educational Tutorials" vs "Podcast/Interviews" = NO (Different Intent)
       - "Documentary" vs "Educational Tutorials" = NO (Different Intent)
       - "Podcast/Interviews" vs "Vlog" = NO (Different Intent)

    2. NICHE (TOPIC) MATCH? (Secondary Check - Must be highly relevant)
       - "Business Case Studies" vs "Corporate History" = YES (High Relevance)
       - "Productivity" vs "Study Skills" = YES (High Relevance)
       - "Entrepreneurship" vs "Startup Growth" = YES (High Relevance)
       ----------------------------------------------------
       - "Business" vs "Personal Finance" = NO (Related, but Different Focus)
       - "Productivity" vs "Tech Reviews" = NO (Different Niches)
       - "Creator Economy" vs "Digital Marketing" = NO (Overlapping, but Different Focus)

    EXAMPLES:

    Seed: "Ali Abdaal", Niche: "Productivity", Format: "Educational Tutorials"
    Cand: "The Diary Of A CEO", Niche: "Entrepreneurship", Format: "Podcast/Interviews"
    Decision: NO (Format mismatch is critical)

    Seed: "MagnatesMedia", Niche: "Business", Format: "Documentary"
    Cand: "Business Breakdown", Niche: "Business Case Studies", Format: "Documentary"
    Decision: YES (Format matches, Niches are highly relevant)

    Seed: "Fireship", Niche: "Web Development", Format: "Educational Tutorials"
    Cand: "Traversy Media", Niche: "Web Development", Format: "Educational Tutorials"
    Decision: YES (Format matches, Niches match)

    Seed: "Johnny Harris", Niche: "Geopolitics", Format: "Documentary"
    Cand: "Vox", Niche: "News Analysis", Format: "Video Essay"
    Decision: YES (Formats are similar storytelling/analysis, Niches are related enough)

    Seed: "Ali Abdaal", Niche: "Productivity", Format: "Educational Tutorials"
    Cand: "Thomas Frank", Niche: "Study Skills", Format: "Educational Tutorials"
    Decision: YES (Format matches, Niches highly relevant)

    Seed: "MrBeast", Niche: "Entertainment", Format: "Challenge/Vlog"
    Cand: "Dude Perfect", Niche: "Entertainment", Format: "Challenge/Stunts"
    Decision: YES (Formats similar, Niches match)


    FINAL QUESTION: Based *only* on the Niche (Topic) and Format (Intent), are these two channels DIRECT competitors? Answer with only "Yes" or "No".

    ANSWER:
    """

    for attempt in range(retries):
        try:
            # Using GPT-4o-mini as requested
            if model_provider.lower() == "gpt":
                if not gpt_client:
                    print("  ❌ GPT client not initialized for competitor check.")
                    return "Error"
                
                # --- USE GPT-4o-mini ---
                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini", # Use the mini model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, # Low temp for direct answer
                    max_tokens=10 # Expecting only "Yes" or "No"
                )
                result_text = response.choices[0].message.content.strip().capitalize()
                
                if result_text in ["Yes", "No"]:
                    print(f"     ✅ LLM Competitor Check: {result_text}")
                    return result_text
                else:
                    print(f"  ⚠️ LLM Competitor Check returned unexpected text: '{result_text}' (Attempt {attempt+1})")
                    # Fall through to retry

            # Add Gemini or other providers if needed, ensure they use a comparable small model
            # elif model_provider.lower() == "gemini":
            #     # ... use gemini flash ...
            #     pass

            else:
                 print(f"  ❌ Unknown model provider '{model_provider}' for competitor check.")
                 return "Error"

        except Exception as e:
            print(f"  ❌ LLM Competitor Check Error (Attempt {attempt+1}/{retries}): {str(e)[:100]}")
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                print("     Rate limit hit, waiting 30s...")
                time.sleep(30)
            elif attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print("  ❌ All retries failed for LLM Competitor Check.")
                return "Error" # Failed after retries

    return "Error" # Should not be reached, but safety return