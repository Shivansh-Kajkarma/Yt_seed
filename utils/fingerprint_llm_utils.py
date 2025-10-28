# import os
# import re
# import time
# import google.generativeai as genai
# from openai import OpenAI
# import pandas as pd
# from typing import List
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from dotenv import load_dotenv

# # ============================================
# # 1️⃣ Load Environment & API Keys
# # ============================================
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# try:
#     # Using a reliable, efficient model
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     print("✅ Sentence Transformer model loaded ('all-MiniLM-L6-v2').")
# except Exception as e:
#     print(f"❌ ERROR loading Sentence Transformer model: {e}")
#     embedding_model = None

# # ============================================
# # 2️⃣ Configure Gemini (if key present)
# # ============================================
# gemini_model = None
# if GOOGLE_API_KEY:
#     try:
#         genai.configure(api_key=GOOGLE_API_KEY)
#         generation_config = genai.GenerationConfig(
#             max_output_tokens=200,
#             temperature=0.1
#         )
#         safety_settings = [
#             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         ]
#         gemini_model = genai.GenerativeModel(
#             'gemini-2.0-flash-exp',
#             generation_config=generation_config,
#             safety_settings=safety_settings
#         )
#         print("✅ Gemini model configured: gemini-2.0-flash-exp")
#     except Exception as e:
#         print(f"❌ Gemini configuration failed: {e}")
# else:
#     print("⚠️ No GOOGLE_API_KEY found — Gemini disabled.")

# # ============================================
# # 3️⃣ Configure OpenAI GPT (if key present)
# # ============================================
# gpt_client = None
# if OPENAI_API_KEY:
#     try:
#         gpt_client = OpenAI(api_key=OPENAI_API_KEY)
#         print("✅ OpenAI client configured.")
#     except Exception as e:
#         print(f"❌ OpenAI initialization failed: {e}")
# else:
#     print("⚠️ No OPENAI_API_KEY found — GPT disabled.")

# # ============================================
# # 4️⃣ Utility: Text Cleaning
# # ============================================
# def preprocess_text_for_llm(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#     text = re.sub(r'\S+@\S+', '', text)
#     text = re.sub(r'[\n\t]+', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# # --- ADDED: Jaccard Similarity Function ---
# # def calculate_jaccard_similarity(keywords1: List[str], keywords2: List[str]) -> float:
# #     """
# #     Calculates Jaccard similarity (intersection / union) between two lists of keywords.
# #     Returns a float between 0.0 and 1.0.
# #     """
# #     # Ensure inputs are lists and handle potential None/empty keywords within them
# #     list1 = keywords1 if isinstance(keywords1, list) else []
# #     list2 = keywords2 if isinstance(keywords2, list) else []

# #     set1 = set(kw.lower() for kw in list1 if kw and isinstance(kw, str)) # Lowercase, filter None/empty
# #     set2 = set(kw.lower() for kw in list2 if kw and isinstance(kw, str))

# #     intersection = len(set1.intersection(set2))
# #     union = len(set1.union(set2))

# #     return round(intersection / union, 3) if union > 0 else 0.0
# # # --- END ADDED ---

# def calculate_embedding_similarity_hybrid(keywords1: List[str], keywords2: List[str]) -> float:
#     """
#     Hybrid approach: Combines average similarity + max similarity.
#     Most robust method.
#     """
#     if not embedding_model or not keywords1 or not keywords2:
#         return 0.0
    
#     try:
#         embeddings1 = embedding_model.encode(keywords1)
#         embeddings2 = embedding_model.encode(keywords2)
        
#         # 1. Average embedding similarity (overall niche)
#         avg_emb1 = embeddings1.mean(axis=0)
#         avg_emb2 = embeddings2.mean(axis=0)
#         avg_similarity = cosine_similarity(
#             avg_emb1.reshape(1, -1),
#             avg_emb2.reshape(1, -1)
#         )[0][0]
        
#         # 2. Maximum pairwise similarity (best keyword matches)
#         similarity_matrix = cosine_similarity(embeddings1, embeddings2)
#         max_similarity = similarity_matrix.max()
        
#         # 3. Weighted combination (70% average, 30% max)
#         final_score = 0.7 * avg_similarity + 0.3 * max_similarity
        
#         return round(float(final_score), 3)
        
#     except Exception as e:
#         print(f"  ❌ Embedding error: {e}")
#         return 0.0
    


# def calculate_llm_similarity(
#     seed_keywords: List[str], 
#     candidate_keywords: List[str],
#     seed_channel_name: str,
#     candidate_channel_name: str,
#     model_type: str = "gemini"
# ) -> float:
#     """
#     Uses LLM to directly score channel similarity.
#     Returns float 0.0-1.0
#     """
    
#     # Create prompt for LLM to score similarity
#     prompt = f"""You are analyzing YouTube channel similarity.

#         SEED CHANNEL: "{seed_channel_name}"
#         Niche keywords: {", ".join(seed_keywords)}

#         CANDIDATE CHANNEL: "{candidate_channel_name}"
#         Niche keywords: {", ".join(candidate_keywords)}

#         TASK: Rate how similar these channels are in terms of content niche and style.

#         SCORING GUIDELINES:
#         - 0.9-1.0: Almost identical niche (e.g., both make business documentary investigations)
#         - 0.7-0.8: Very similar niche (e.g., business docs vs startup stories)
#         - 0.5-0.6: Related niche (e.g., business docs vs economics education)
#         - 0.3-0.4: Somewhat related (e.g., business docs vs tech reviews)
#         - 0.0-0.2: Different niches (e.g., business docs vs gaming)

#         Consider:
#         - Content TYPE (documentary vs news vs tutorial vs entertainment)
#         - Topic FOCUS (business, tech, science, etc.)
#         - Production STYLE (storytelling vs reporting vs teaching)

#         Return ONLY a number between 0.0 and 1.0, nothing else.
#     """
    
#     try:
#         if model_type.lower() == "gemini":
#             import google.generativeai as genai
#             model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
#             response = model.generate_content(prompt)
#             result_text = response.text.strip()
            
#         else:  # OpenAI
#             from openai import OpenAI
#             client = OpenAI()
            
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0
#             )
#             result_text = response.choices[0].message.content.strip()
        
#         # Extract number from response
#         import re
#         match = re.search(r'0?\.\d+|1\.0|0|1', result_text)
        
#         if match:
#             score = float(match.group())
#             return round(min(max(score, 0.0), 1.0), 3)  # Clamp 0-1
#         else:
#             print(f"  ⚠️ LLM returned non-numeric: {result_text}")
#             return 0.0
            
#     except Exception as e:
#         print(f"  ❌ LLM similarity error: {str(e)[:50]}")
#         return 0.0


# # ============================================
# # 5️⃣ LLM Keyword Extraction (Gemini / GPT Modular)
# # ============================================
# def extract_keywords_llm(combined_text: str, channel_name: str = "", model_type: str = "gemini", max_chars: int = 40000) -> list[str]:
#     """Extracts keywords using Gemini or GPT based on model_type."""
#     if not combined_text:
#         print("⚠️ No text provided.")
#         return []

#     truncated_content = combined_text[:max_chars]
#     print(f"📤 Sending {len(truncated_content)} chars to {model_type.upper()}...")
#     channel_name_cleaned = channel_name.lower().strip() if channel_name else "[Channel Name Unavailable]"

#     prompt = f"""Act as an expert YouTube channel analyst specializing in content categorization and thematic analysis. Analyze the following combined text from recent video titles and descriptions of the channel named '{channel_name}'.

#     Your primary goal is to identify and list the **recurring themes, content categories, genres, and overall niche** of this channel. Focus on the **conceptual essence** and **TYPE** of content the channel produces consistently, not the specifics of individual videos.

#     Extract a list of **10 concise phrases** (strictly **1 to 3 words each**). These phrases should describe **WHAT KIND** of content this channel makes. Think in terms of categories, subjects, styles, and formats.

#     **Follow the style and specificity of these examples:**
#     - For business documentaries: "business documentary", "corporate scandal", "company downfall", "historical analysis", "entrepreneur stories"
#     - For tech reviews: "tech review", "smartphone comparison", "gadget testing", "consumer electronics", "pc building"
#     - For cooking: "recipe tutorial", "cooking techniques", "food preparation", "home cooking", "baking guide"

#     **CRITICALLY AVOID (Do NOT include these specific items):**
#     - The channel's own name ('{channel_name_cleaned}') or creator/host names.
#     - Specific, one-off video titles, individual event names (e.g., 'jump over lava'), specific product models (e.g., 'iphone 17 pro'), or specific people/company names mentioned only once or twice. **Instead, use the broader category** (e.g., 'challenge video' instead of 'jump over lava', 'smartphone review' instead of 'iphone 17 pro review').
#     - Generic YouTube/internet jargon ('video', 'channel', 'subscribe', 'shorts', 'new', 'official', 'review', 'unboxing', 'gameplay', 'vlog', 'live stream', 'patreon', 'merch', social media platform names).
#     - Vague filler words ('world', 'life', 'people', 'thing', 'amazing', 'ultimate', 'best', 'top', 'how to', 'why', 'what', 'insane', 'crazy', 'real', 'every', 'using', 'tips', 'tricks').
#     - Calls to action, sponsor mentions, boilerplate text (like intro/outro phrases, affiliate links).
#     - Numbers, dates, or time references (e.g., '2025 update', 'part 3').

#     Combined Titles and Descriptions:
#     \"\"\"
#     {truncated_content}
#     \"\"\"

#     Return ONLY a comma-separated list of the 10 niche phrases (1-3 words each), strictly adhering to the guidelines and AVOID list. Do not add explanations, numbering, or any other text."""
#     retries = 3
#     for attempt in range(retries):
#         try:
#             # ========== GEMINI MODE ==========
#             if model_type.lower() == "gemini":
#                 if not gemini_model:
#                     print("❌ Gemini not initialized.")
#                     return []
#                 response = gemini_model.generate_content(prompt)
#                 if not response.parts:
#                     print("⚠️ Gemini blocked or empty response.")
#                     return []
#                 text_out = response.text.strip()

#             # ========== GPT MODE ==========
#             elif model_type.lower() == "gpt":
#                 if not gpt_client:
#                     print("❌ GPT client not initialized.")
#                     return []
#                 response = gpt_client.chat.completions.create(
#                     model="gpt-4o-mini",  # economical + strong for keyword reasoning
#                     messages=[
#                         {"role": "system", "content": "You are a YouTube keyword expert."},
#                         {"role": "user", "content": prompt},
#                     ],
#                     temperature=0.2,
#                     max_tokens=150,
#                 )
#                 text_out = response.choices[0].message.content.strip()

#             else:
#                 print(f"❌ Unknown model type '{model_type}'. Use 'gemini' or 'gpt'.")
#                 return []

#             # ========== Parse Keywords ==========
#             keywords = [kw.strip().lower() for kw in re.split(r'[,\n]+', text_out) if kw.strip()]
#             keywords = [kw for kw in keywords if len(kw) > 1]

#             if keywords:
#                 print(f"✅ Got {len(keywords)} keywords from {model_type.upper()}")
#                 time.sleep(4)
#                 return keywords[:10]
#             else:
#                 print(f"⚠️ Empty keyword list from {model_type.upper()}")
#                 return []

#         except Exception as e:
#             print(f"❌ Error ({model_type.upper()} attempt {attempt+1}/{retries}): {str(e)[:100]}")
#             if "429" in str(e) or "quota" in str(e).lower():
#                 print("💤 Rate limit hit, waiting 60s...")
#                 time.sleep(60)
#             elif attempt < retries - 1:
#                 time.sleep(5 * (attempt + 1))
#             else:
#                 print("❌ All retries failed.")
#                 return []
#     return []

# # ============================================
# # 6️⃣ Channel Fingerprint Builder
# # ============================================
# def create_channel_fingerprint_llm(video_df: pd.DataFrame, channel_name: str = "", top_n: int = 15, model_type: str = "gemini") -> list[str]:
#     """Creates channel fingerprint using either Gemini or GPT."""
#     if video_df.empty:
#         print("⚠️ Empty DataFrame")
#         return []

#     print(f"📊 Processing {len(video_df)} videos...")

#     combined_texts = []
#     for _, row in video_df.iterrows():
#         title = str(row.get('title', ''))
#         description = str(row.get('description', ''))
#         clean_title = preprocess_text_for_llm(title)
#         clean_desc = preprocess_text_for_llm(description[:200])
#         if channel_name:
#             clean_title = clean_title.replace(channel_name.lower(), '')
#             clean_desc = clean_desc.replace(channel_name.lower(), '')
#         combined_texts.append(clean_title * 2)  # weight titles higher
#         combined_texts.append(clean_desc)

#     full_text_blob = "\n---\n".join(filter(None, combined_texts))
#     if not full_text_blob:
#         print("⚠️ No text to analyze.")
#         return []

#     keywords = extract_keywords_llm(full_text_blob, channel_name=channel_name, model_type=model_type)
#     return keywords



import os
import re
import time
import google.generativeai as genai
from openai import OpenAI
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


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
    


# --- MODIFIED: Function updated to include Niche for context ---
def calculate_llm_similarity(
    seed_keywords: List[str], 
    candidate_keywords: List[str],
    seed_channel_name: str,
    candidate_channel_name: str,
    seed_niche: str = "",          # <-- ADDED
    candidate_niche: str = "",     # <-- ADDED
    model_type: str = "gemini"
) -> float:
    """
    Uses LLM to directly score channel similarity, now using Niche as a key factor.
    Returns float 0.0-1.0
    """
    
    # --- MODIFIED: Prompt now includes the niche ---
    prompt = f"""You are an expert YouTube channel analyst evaluating channel similarity for content discovery.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SEED CHANNEL: "{seed_channel_name}"
    Primary Niche: {seed_niche or "Unknown"}
    Keywords: {", ".join(seed_keywords[:5])}...

    CANDIDATE CHANNEL: "{candidate_channel_name}"
    Primary Niche: {candidate_niche or "Unknown"}
    Keywords: {", ".join(candidate_keywords[:5])}...
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
                model="gpt-4o",
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
# 5️⃣ --- ADDED: New Function for Niche Extraction ---
# ============================================
# def extract_niche_llm(
#     channel_description: str, 
#     channel_name: str, 
#     model_type: str = "gemini", 
#     max_chars: int = 2000
# ) -> str:
#     """
#     Analyzes a channel's "About" description to determine its primary niche.
#     Returns a single string (e.g., "Business documentaries and corporate analysis").
#     """
#     if not channel_description:
#         print(f"⚠️ No description for {channel_name}, cannot extract niche.")
#         return "General" # Return a default
        
#     truncated_desc = preprocess_text_for_llm(channel_description)[:max_chars]
    
#     prompt = f"""
#     Analyze the following "About" page description for the YouTube channel "{channel_name}".

#     Description:
#     \"\"\"
#     {truncated_desc}
#     \"\"\"

#     TASK: Identify the channel's **primary niche** in a single, concise phrase.
#     This phrase should be 3-7 words. Examples:
#     - "Creator economy news and interviews"
#     - "Product management and tech careers"
#     - "Business documentaries and corporate analysis"
#     - "Consumer tech reviews and gadget unboxings"
#     - "Self-improvement and entrepreneurial mindset"

#     Return ONLY the niche phrase, nothing else.
#     """
    
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
#                 model="gpt-4o-mini",
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
            
#         print(f"✅ Niche for {channel_name}: {niche}")
#         time.sleep(2) # Short sleep after call
#         return niche

#     except Exception as e:
#         print(f"❌ LLM niche extraction error for {channel_name}: {str(e)[:50]}")
#         return "General"

def extract_niche_llm(
    channel_description: str, 
    video_titles: List[str],  # <-- ADD THIS
    channel_name: str, 
    model_type: str = "gemini", 
    max_chars: int = 2000
) -> str:
    """
    HYBRID APPROACH: 
    1. Try channel description first (90% of cases)
    2. Fallback to video titles (10% of cases)
    """
    
    # Step 1: Check if description is good quality
    text_for_niche = ""
    text_source = ""
    
    if channel_description and len(channel_description) >= 100:
        # Check if not mostly URLs
        url_count = channel_description.count("http")
        if url_count / max(len(channel_description), 1) < 0.3:  # Less than 30% URLs
            text_for_niche = channel_description[:max_chars]
            text_source = "channel description"
            print(f"  ℹ️ Using channel description for {channel_name}")
    
    # Step 2: Fallback to video titles if description is poor
    if not text_for_niche and video_titles:
        # Concatenate top 10 video titles
        text_for_niche = " | ".join(video_titles[:10])
        text_source = "video titles"
        print(f"  ⚠️ Using video titles fallback for {channel_name}")
    
    # Step 3: No data available
    if not text_for_niche:
        print(f"  ❌ No data for {channel_name}, returning 'General'")
        return "General"
    
    # Preprocess
    truncated_text = preprocess_text_for_llm(text_for_niche)[:max_chars]
    
    # Updated prompt that mentions the source
    prompt = f"""You are an expert YouTube channel analyst.
        Your task is to analyze the following {text_source} for the channel "{channel_name}" and identify its single primary niche.

        {text_source.upper()}:
        \"\"\"
        {truncated_text}
        \"\"\"

        TASK: Identify the channel's **primary niche** in a single, concise phrase (3-7 words).
        The niche must describe the **content's PURPOSE and TOPIC** for a viewer.

        **CRITICAL RULE: Distinguish the channel's INTENT.**
        - Is it **"Career / Education"** (teaching a skill, 'how to', job prep)?
        - Is it **"Storytelling / Analysis"** (documentaries, case studies, news, entertainment)?
        - Is it **"Mindset / Motivation"** (self-improvement, leadership advice)?

        **EXAMPLES OF GOOD NICHES (Note the INTENT):**

        # Example 1: The "Business Analysis" Problem
        - **GOOD Niche:** "Business analysis career prep" (This is Career/Education)
        - **GOOD Niche:** "Business documentaries and case studies" (This is Storytelling/Analysis)
        - **BAD Niche:** "Business analysis" (This is too vague)

        # Example 2: The "Creator" Problem
        - **GOOD Niche:** "Creator economy news and interviews" (This is Storytelling/Analysis)
        - **GOOD Niche:** "YouTube growth tips and tutorials" (This is Career/Education)

        # Example 3: Other Good Niches
        - "Product management and tech careers" (Career/Education)
        - "Consumer tech reviews and unboxings" (Storytelling/Analysis)
        - "Entrepreneurial mindset and leadership" (Mindset/Motivation)

        Return ONLY the single niche phrase, nothing else.
        """
    
    try:
        if model_type.lower() == "gemini":
            if not gemini_model:
                print("❌ Gemini not initialized for niche.")
                return "General"
            response = gemini_model.generate_content(prompt)
            niche = response.text.strip().replace('"', '')
            
        elif model_type.lower() == "gpt":
            if not gpt_client:
                print("❌ GPT not initialized for niche.")
                return "General"
            response = gpt_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )
            niche = response.choices[0].message.content.strip().replace('"', '')
            
        else:
            print(f"❌ Unknown model type '{model_type}'.")
            return "General"
            
        if not niche:
            print(f"⚠️ LLM returned empty niche for {channel_name}")
            return "General"
            
        print(f"  ✅ Niche for {channel_name}: {niche} (from {text_source})")
        time.sleep(2)
        return niche

    except Exception as e:
        print(f"  ❌ LLM niche extraction error for {channel_name}: {str(e)[:50]}")
        return "General"



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
) -> list[str]:
    """Extracts keywords using Gemini or GPT, now guided by the channel's niche."""
    if not combined_text:
        print("⚠️ No text provided.")
        return []


    truncated_content = combined_text[:max_chars]
    print(f"📤 Sending {len(truncated_content)} chars to {model_type.upper()}...")
    channel_name_cleaned = channel_name.lower().strip() if channel_name else "[Channel Name Unavailable]"
    
    # --- MODIFIED: Prompt now includes the niche ---
    prompt = f"""You are an expert YouTube content analyst specializing in channel categorization and discovery. Your task is to extract HIGH-QUALITY search keywords for finding similar channels.

    CHANNEL INFORMATION:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Channel Name: "{channel_name}"
    Primary Niche: **{niche}**
    Data Source: Video titles and descriptions

    CRITICAL CONTEXT:
    These keywords will be used to SEARCH for similar channels. They must:
    1. Reflect the niche: "{niche}"
    2. Be SEARCHABLE on YouTube (what users would type)
    3. Describe content TYPES, not specific instances
    4. Match the channel's INTENT (education/storytelling/motivation)

    🎯 KEYWORD MIX TARGET (Important!):
    - **60% TOPIC keywords** (WHAT they discuss: "corporate scandals", "PM strategies")
    - **40% FORMAT keywords** (HOW they present: "documentary style", "interview format")

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    VIDEO CONTENT TO ANALYZE:
    \"\"\"
    {truncated_content}
    \"\"\"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    TASK: Extract 10 SPECIFIC, SEARCHABLE keywords (2-4 words each) that:

    ✅ MUST DO:
    1. Align with the niche "{niche}" (70%+ relevance)
    2. Follow 60/40 split: ~6 TOPIC keywords, ~4 FORMAT keywords
    3. Use terms people SEARCH for (not internal jargon)
    4. Describe RECURRING themes (not one-off topics)
    5. Be specific enough to filter similar channels

    ❌ MUST AVOID:
    1. Channel name: "{channel_name_cleaned}"
    2. Generic terms: "video", "content", "channel", "tips", "guide", "tutorial", "how to", "best", "top", "new", "official"
    3. Platform names: "youtube", "instagram", "tiktok", "facebook", "twitter"
    4. Vague fillers: "amazing", "ultimate", "insane", "crazy", "life", "people", "world"
    5. Specific instances: Event names, product models, dates, people's names (use categories instead)
    6. Action words alone: "unboxing", "review", "gameplay" (combine with topic: "smartphone unboxing")
    7. Too broad: "business", "technology", "education" (add specificity)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📌 NICHE-ALIGNED EXAMPLES (Notice Topic vs Format):

    🎯 Niche: "Product management and tech careers"
    Good Keywords (60/40 mix):
    ✅ "product roadmap planning" (Topic)
    ✅ "PM interview preparation" (Topic)
    ✅ "tech career advancement" (Topic)
    ✅ "user research methods" (Topic)
    ✅ "product strategy frameworks" (Topic)
    ✅ "startup product insights" (Topic)
    ✅ "whiteboard explainers" (Format)
    ✅ "interview format discussions" (Format)
    ✅ "deep-dive analysis" (Format)
    ✅ "case study breakdowns" (Format)

    🎯 Niche: "Business documentaries and corporate history"
    Good Keywords (60/40 mix):
    ✅ "corporate scandal analysis" (Topic)
    ✅ "business failure case studies" (Topic)
    ✅ "company history narratives" (Topic)
    ✅ "entrepreneur rise and fall" (Topic)
    ✅ "brand evolution stories" (Topic)
    ✅ "financial crisis analysis" (Topic)
    ✅ "cinematic documentary style" (Format)
    ✅ "narrative storytelling" (Format)
    ✅ "archival footage presentations" (Format)
    ✅ "investigative deep-dives" (Format)

    🎯 Niche: "YouTube creator economy and business"
    Good Keywords (60/40 mix):
    ✅ "creator monetization strategies" (Topic)
    ✅ "youtube growth tactics" (Topic)
    ✅ "content creator business" (Topic)
    ✅ "platform algorithm insights" (Topic)
    ✅ "influencer marketing trends" (Topic)
    ✅ "creator economy news" (Topic)
    ✅ "creator interviews" (Format)
    ✅ "podcast-style discussions" (Format)
    ✅ "news and commentary" (Format)
    ✅ "behind-the-scenes insights" (Format)

    🎯 Niche: "Digital marketing and SEO education"
    Good Keywords (60/40 mix):
    ✅ "SEO optimization techniques" (Topic)
    ✅ "conversion rate strategies" (Topic)
    ✅ "marketing funnel analysis" (Topic)
    ✅ "content marketing tactics" (Topic)
    ✅ "paid advertising campaigns" (Topic)
    ✅ "email marketing automation" (Topic)
    ✅ "tutorial-style teaching" (Format)
    ✅ "screen recording walkthroughs" (Format)
    ✅ "live Q&A sessions" (Format)
    ✅ "step-by-step guides" (Format)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    🎯 QUALITY CHECKLIST (Apply to each keyword):
    1. ✓ Is it 2-4 words?
    2. ✓ Does it match the niche "{niche}"?
    3. ✓ Is it a TOPIC (what) or FORMAT (how)?
    4. ✓ Would someone SEARCH this on YouTube?
    5. ✓ Is it specific but not too narrow?

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    OUTPUT FORMAT:
    Return EXACTLY 10 keywords as a comma-separated list.
    Aim for ~6 TOPIC keywords, ~4 FORMAT keywords.
    No numbering, no explanations, no quotes, no extra text.

    Example output format:
    product roadmap planning, PM interview preparation, tech career advancement, user research methods, product strategy frameworks, startup product insights, whiteboard explainers, interview format discussions, deep-dive analysis, case study breakdowns
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
                    max_tokens=150,
                )
                text_out = response.choices[0].message.content.strip()


            else:
                print(f"❌ Unknown model type '{model_type}'. Use 'gemini' or 'gpt'.")
                return []


            # ========== Parse Keywords ==========
            keywords = [kw.strip().lower() for kw in re.split(r'[,\n]+', text_out) if kw.strip()]
            keywords = [kw for kw in keywords if len(kw) > 1]


            if keywords:
                print(f"✅ Got {len(keywords)} keywords from {model_type.upper()}")
                time.sleep(4)
                return keywords[:10]
            else:
                print(f"⚠️ Empty keyword list from {model_type.upper()}")
                return []


        except Exception as e:
            print(f"❌ Error ({model_type.upper()} attempt {attempt+1}/{retries}): {str(e)[:100]}")
            if "429" in str(e) or "quota" in str(e).lower():
                print("💤 Rate limit hit, waiting 60s...")
                time.sleep(60)
            elif attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print("❌ All retries failed.")
                return []
    return []


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
) -> list[str]:
    """Creates channel fingerprint using either Gemini or GPT, guided by the niche."""
    if video_df.empty:
        print("⚠️ Empty DataFrame")
        return []


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
        return []

    # --- MODIFIED: Pass the niche to the keyword extractor ---
    keywords = extract_keywords_llm(
        full_text_blob, 
        channel_name=channel_name, 
        niche=niche,  # <-- Pass the niche here
        model_type=model_type
    )
    return keywords