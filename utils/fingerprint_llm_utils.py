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
        gemini_model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        print("✅ Gemini model configured: gemini-2.0-flash-exp")
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

# --- ADDED: Jaccard Similarity Function ---
# def calculate_jaccard_similarity(keywords1: List[str], keywords2: List[str]) -> float:
#     """
#     Calculates Jaccard similarity (intersection / union) between two lists of keywords.
#     Returns a float between 0.0 and 1.0.
#     """
#     # Ensure inputs are lists and handle potential None/empty keywords within them
#     list1 = keywords1 if isinstance(keywords1, list) else []
#     list2 = keywords2 if isinstance(keywords2, list) else []

#     set1 = set(kw.lower() for kw in list1 if kw and isinstance(kw, str)) # Lowercase, filter None/empty
#     set2 = set(kw.lower() for kw in list2 if kw and isinstance(kw, str))

#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))

#     return round(intersection / union, 3) if union > 0 else 0.0
# # --- END ADDED ---

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
    


def calculate_llm_similarity(
    seed_keywords: List[str], 
    candidate_keywords: List[str],
    seed_channel_name: str,
    candidate_channel_name: str,
    model_type: str = "gemini"
) -> float:
    """
    Uses LLM to directly score channel similarity.
    Returns float 0.0-1.0
    """
    
    # Create prompt for LLM to score similarity
    prompt = f"""You are analyzing YouTube channel similarity.

        SEED CHANNEL: "{seed_channel_name}"
        Niche keywords: {", ".join(seed_keywords)}

        CANDIDATE CHANNEL: "{candidate_channel_name}"
        Niche keywords: {", ".join(candidate_keywords)}

        TASK: Rate how similar these channels are in terms of content niche and style.

        SCORING GUIDELINES:
        - 0.9-1.0: Almost identical niche (e.g., both make business documentary investigations)
        - 0.7-0.8: Very similar niche (e.g., business docs vs startup stories)
        - 0.5-0.6: Related niche (e.g., business docs vs economics education)
        - 0.3-0.4: Somewhat related (e.g., business docs vs tech reviews)
        - 0.0-0.2: Different niches (e.g., business docs vs gaming)

        Consider:
        - Content TYPE (documentary vs news vs tutorial vs entertainment)
        - Topic FOCUS (business, tech, science, etc.)
        - Production STYLE (storytelling vs reporting vs teaching)

        Return ONLY a number between 0.0 and 1.0, nothing else.
    """
    
    try:
        if model_type.lower() == "gemini":
            import google.generativeai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
        else:  # OpenAI
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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
# 5️⃣ LLM Keyword Extraction (Gemini / GPT Modular)
# ============================================
def extract_keywords_llm(combined_text: str, channel_name: str = "", model_type: str = "gemini", max_chars: int = 40000) -> list[str]:
    """Extracts keywords using Gemini or GPT based on model_type."""
    if not combined_text:
        print("⚠️ No text provided.")
        return []

    truncated_content = combined_text[:max_chars]
    print(f"📤 Sending {len(truncated_content)} chars to {model_type.upper()}...")

    # prompt = f"""Act as an expert YouTube channel analyst. Analyze the following combined text from recent video titles and descriptions of a single channel. Your goal is to identify the core niche and recurring topics.

    # Based ONLY on the text provided, extract a list of the 15 most specific and descriptive keywords or short phrases (2-4 words max) that accurately represent the primary subject matter and unique focus of this channel.

    # Prioritize:
    # - Specific topics, technologies, games, scientific concepts, historical periods, locations, food types, or artistic styles frequently mentioned.
    # - Terms that define the channel's unique angle or genre (e.g., 'animated science explainer', 'retro game review', 'budget travel guide', 'investigative business documentary').

    # CRITICALLY AVOID (Do NOT include these):
    # - The channel's own name {channel_name} or variations of it .
    # - Names of collaborators or guests UNLESS they are the primary subject of multiple videos.
    # - Generic YouTube/internet words ('video', 'channel', 'subscribe', 'shorts', 'vlog', 'new', 'official', 'trailer', 'gameplay', 'review', 'unboxing', 'live stream', 'website', 'merch', 'patreon', 'twitter', 'instagram', 'facebook').
    # - Vague fillers ('world', 'life', 'people', 'thing', 'amazing', 'ultimate', 'best', 'top', 'how to', 'why', 'what', 'insane', 'crazy', 'real', 'every').
    # - Calls to action, sponsor mentions, social media links/handles, or website URLs.
    # - Music/voice credits (e.g., 'steve taylor', 'epic mountain').
    # - Extremely common single English words (e.g., 'make', 'get', 'part', 'using', 'hai'). Only use single words if they are specific technical terms (e.g., 'thermodynamics', 'photosynthesis').

    # Combined Titles and Descriptions:
    # \"\"\"
    # {truncated_content}
    # \"\"\"

    # Return ONLY a comma-separated list of the 10 keywords/phrases, strictly adhering to the AVOID list. Do not add explanations, numbering, or any other text."""

    channel_name_cleaned = channel_name.lower().strip() if channel_name else "[Channel Name Unavailable]"

    prompt = f"""Act as an expert YouTube channel analyst specializing in content categorization and thematic analysis. Analyze the following combined text from recent video titles and descriptions of the channel named '{channel_name}'.

    Your primary goal is to identify and list the **recurring themes, content categories, genres, and overall niche** of this channel. Focus on the **conceptual essence** and **TYPE** of content the channel produces consistently, not the specifics of individual videos.

    Extract a list of **10 concise phrases** (strictly **1 to 3 words each**). These phrases should describe **WHAT KIND** of content this channel makes. Think in terms of categories, subjects, styles, and formats.

    **Follow the style and specificity of these examples:**
    - For business documentaries: "business documentary", "corporate scandal", "company downfall", "historical analysis", "entrepreneur stories"
    - For tech reviews: "tech review", "smartphone comparison", "gadget testing", "consumer electronics", "pc building"
    - For cooking: "recipe tutorial", "cooking techniques", "food preparation", "home cooking", "baking guide"

    **CRITICALLY AVOID (Do NOT include these specific items):**
    - The channel's own name ('{channel_name_cleaned}') or creator/host names.
    - Specific, one-off video titles, individual event names (e.g., 'jump over lava'), specific product models (e.g., 'iphone 17 pro'), or specific people/company names mentioned only once or twice. **Instead, use the broader category** (e.g., 'challenge video' instead of 'jump over lava', 'smartphone review' instead of 'iphone 17 pro review').
    - Generic YouTube/internet jargon ('video', 'channel', 'subscribe', 'shorts', 'new', 'official', 'review', 'unboxing', 'gameplay', 'vlog', 'live stream', 'patreon', 'merch', social media platform names).
    - Vague filler words ('world', 'life', 'people', 'thing', 'amazing', 'ultimate', 'best', 'top', 'how to', 'why', 'what', 'insane', 'crazy', 'real', 'every', 'using', 'tips', 'tricks').
    - Calls to action, sponsor mentions, boilerplate text (like intro/outro phrases, affiliate links).
    - Numbers, dates, or time references (e.g., '2025 update', 'part 3').

    Combined Titles and Descriptions:
    \"\"\"
    {truncated_content}
    \"\"\"

    Return ONLY a comma-separated list of the 10 niche phrases (1-3 words each), strictly adhering to the guidelines and AVOID list. Do not add explanations, numbering, or any other text."""
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
                    model="gpt-4o-mini",  # economical + strong for keyword reasoning
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
# 6️⃣ Channel Fingerprint Builder
# ============================================
def create_channel_fingerprint_llm(video_df: pd.DataFrame, channel_name: str = "", top_n: int = 15, model_type: str = "gemini") -> list[str]:
    """Creates channel fingerprint using either Gemini or GPT."""
    if video_df.empty:
        print("⚠️ Empty DataFrame")
        return []

    print(f"📊 Processing {len(video_df)} videos...")

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

    keywords = extract_keywords_llm(full_text_blob, channel_name=channel_name, model_type=model_type)
    return keywords
