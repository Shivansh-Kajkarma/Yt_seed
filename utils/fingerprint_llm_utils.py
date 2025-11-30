import os
import re
import time
from openai import OpenAI
import pandas as pd
from typing import Dict, List

# from sentence_transformers import SentenceTransformer  # DEPRECATED: Now using OpenAI embeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import json
import pprint
import numpy as np

# ============================================
# 1️⃣ Load Environment & API Keys
# ============================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FORMAT_CONFIG = {
    "Podcast": {
        "keywords": ["podcast", "interview", "show", "talk", "conversation"],
        "rules": "Keywords MUST imply audio/dialogue. Focus on 'Who is speaking' and 'Topic'.",
        "examples": "✓ 'startup founder podcast'\n   ✓ 'tech leader interviews'",
    },
    "Documentary": {
        "keywords": [
            "documentary",
            "doc",
            "story of",
            "history of",
            "explained",
            "breakdown",
        ],
        "rules": "Keywords MUST imply narrative/storytelling. Focus on 'Subject' + 'Documentary'.",
        "examples": "✓ 'tech startup documentary'\n   ✓ 'business failure explained'",
    },
    "Tutorial": {
        "keywords": ["tutorial", "guide", "how to", "course", "learn", "walkthrough"],
        "rules": "Keywords MUST imply utility/education. Focus on 'Skill' + 'Guide'.",
        "examples": "✓ 'python programming tutorial'\n   ✓ 'seo guide for beginners'",
    },
    "Talking Head": {
        "keywords": ["analysis", "review", "opinion", "commentary", "video essay"],
        "rules": "Keywords MUST imply analysis/opinion. Focus on 'Topic' + 'Analysis'.",
        "examples": "✓ 'apple vision pro review'\n   ✓ 'market crash analysis'",
    },
    # Fallback for unknown formats
    "General": {
        "keywords": [],
        "rules": "Focus on the core topic and content style.",
        "examples": "✓ 'tech news'\n   ✓ 'funny cat videos'",
    },
}

FORMAT_VALIDATION_CONFIG = {
    "Podcast": {
    "positive_signals": 
        """A podcast transcript typically has: 
        • Conversational dialogue between two or more speakers, OR a single host delivering long-form commentary. 
        • Turn-taking or naturally flowing responses, questions, and follow-ups. 
        • Host-driven framing ('Today we're talking about...', 'My guest is...', 'Let's dive in'). 
        • Long uninterrupted segments where the speaker explores one topic deeply (not action-based instructions). 
        • No strict procedural steps; instead: opinions, stories, insights, experiences, interviews. 
        • Language patterns consistent with spoken conversation: fillers, pauses, rhetorical questions, personal references, anecdotes.""",

    "structural_cue": 
        """Expected structure markers: 
        • Opening context-setting by host. 
        • Guest introduction or topic framing. 
        • Multi-minute segments exploring one theme before shifting. 
        • Smooth transitions between ideas ('Now let’s shift to...', 'Moving on...'). 
        • Ending with summary, takeaway, or sign-off. 
        The content should clearly resemble spoken audio intended for listening rather than watching."""
},

    "Documentary": {
        "positive_signals": "Look for scripted voiceover narration (often 3rd person). Past tense storytelling ('He started the company in...'). High production value descriptions.",
        "negative_constraints": "REJECT if: It is a 'Reaction Video' (watching someone else). REJECT if it is 'Commentary' (just a guy talking at a desk without narrative b-roll).",
        "structural_cue": "Structure must be: Narrative Hook -> Chronological Story -> Conclusion.",
    },
    "Tutorial": {
        "positive_signals": "Look for imperative verbs ('Click here', 'Do this', 'Type that'). Sequential steps ('Step 1', 'Next'). Instructional tone.",
        "negative_constraints": "REJECT if: It is a 'Review' (giving opinion vs teaching how). REJECT if it is a 'Speedbuild' (music only, no instruction).",
        "structural_cue": "Structure must be: Problem Statement -> Step-by-Step Solution -> Result.",
    },
    "Talking Head": {
        "positive_signals": "Look for direct eye contact (implied text). Opinions, Analysis, or Commentary on a specific topic. Single dominant speaker.",
        "negative_constraints": "REJECT if: It is an Interview (Podcast). REJECT if it is a Skit/Comedy Sketch.",
        "structural_cue": "Structure must be: Thesis Statement -> Arguments/Analysis -> Conclusion.",
    },
    "General": {
        "positive_signals": "Check if content matches the general theme.",
        "negative_constraints": "None.",
        "structural_cue": "Any.",
    },
}

# DEPRECATED: Sentence Transformers model (replaced by OpenAI embeddings in Phase 3 Step 3B)
# try:
#     # Using a reliable, efficient model
#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#     print("✅ Sentence Transformer model loaded ('all-MiniLM-L6-v2').")
# except Exception as e:
#     print(f"❌ ERROR loading Sentence Transformer model: {e}")
#     embedding_model = None
embedding_model = None  # Not used in current pipeline


# ============================================
# 2️⃣ Configure OpenAI GPT
# ============================================
gpt_client = None
if OPENAI_API_KEY:
    try:
        gpt_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI GPT client configured.")
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
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# DEPRECATED: This function used local Sentence Transformers (now using OpenAI embeddings)
def calculate_embedding_similarity_hybrid(
    keywords1: List[str], keywords2: List[str]
) -> float:
    """
    Hybrid approach: Combines average similarity + max similarity.
    Most robust method.
    DEPRECATED: Replaced by OpenAI embeddings in Phase 3 Step 3B
    """
    # if not embedding_model or not keywords1 or not keywords2:
    #     return 0.0
    #
    # try:
    #     embeddings1 = embedding_model.encode(keywords1)
    #     embeddings2 = embedding_model.encode(keywords2)
    #
    #     # 1. Average embedding similarity (overall niche)
    #     avg_emb1 = embeddings1.mean(axis=0)
    #     avg_emb2 = embeddings2.mean(axis=0)
    #     avg_similarity = cosine_similarity(
    #         avg_emb1.reshape(1, -1), avg_emb2.reshape(1, -1)
    #     )[0][0]
    #
    #     # 2. Maximum pairwise similarity (best keyword matches)
    #     similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    #     max_similarity = similarity_matrix.max()
    #
    #     # 3. Weighted combination (70% average, 30% max)
    #     final_score = 0.7 * avg_similarity + 0.3 * max_similarity
    #
    #     return round(float(final_score), 3)
    #
    # except Exception as e:
    #     print(f"  ❌ Embedding error: {e}")
    return 0.0  # Not implemented for current pipeline


def _get_keyword_sample(
    kw_dict: dict, max_categories=3, keywords_per_category=3
) -> str:
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
    seed_keywords: dict,  # <-- This is now a DICT
    candidate_keywords: dict,  # <-- This is now a DICT
    seed_channel_name: str,
    candidate_channel_name: str,
    seed_niche: str = "",  # <-- This is "Niche - Format"
    candidate_niche: str = "",  # <-- This is "Niche - Format"
    model_type: str = "gpt",
) -> float:
    """
    Uses LLM to directly score channel similarity, now using Niche as a key factor.
    Returns float 0.0-1.0
    """

    seed_categories = (
        list(seed_keywords.keys()) if isinstance(seed_keywords, dict) else ["Unknown"]
    )
    candidate_categories = (
        list(candidate_keywords.keys())
        if isinstance(candidate_keywords, dict)
        else ["Unknown"]
    )

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
        if not gpt_client:
            print("❌ GPT not initialized for similarity.")
            return 0.0

        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result_text = response.choices[0].message.content.strip()

        match = re.search(r"0?\.\d+|1\.0|0|1", result_text)

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
    max_chars: int = 2000,
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
        if not gpt_client:
            return "General - Unknown"

        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        niche_format = response.choices[0].message.content.strip().replace('"', "")

        # Validate the "Niche - Format" structure
        if " - " not in niche_format or len(niche_format) < 7:
            print(f"⚠️ LLM returned invalid format: '{niche_format}'. Defaulting.")
            # Try to salvage, or just default
            if niche_format:
                return f"{niche_format} - Unknown"
            return "General - Unknown"

        print(
            f"  ✅ Niche/Format for {channel_name}: {niche_format} (from {text_source})"
        )
        time.sleep(2)  # Keep your rate limit
        return niche_format

    except Exception as e:
        print(f"  ❌ LLM niche extraction error for {channel_name}: {str(e)[:50]}")
        return "General - Unknown"


# ============================================
# 6️⃣ LLM Keyword Extraction (Gemini / GPT Modular)
# ============================================
# --- MODIFIED: Function updated to accept 'niche' for context ---
def extract_keywords_llm(
    combined_text: str, channel_name: str = "", niche: str = "", max_chars: int = 40000
) -> Dict[str, List[str]]:
    """Extracts keywords using Gemini or GPT, now guided by the channel's niche."""
    if not combined_text:
        print("⚠️ No text provided.")
        return {}

    truncated_content = combined_text[:max_chars]
    print(f"📤 Sending {len(truncated_content)} chars to GPT...")

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
            if not gpt_client:
                print("❌ GPT client not initialized.")
                return {}

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

            # ========== Parse Keywords ==========
            # ========== Parse JSON Keywords ==========
            keywords_categorized = {}  # Default to empty dict
            try:
                # Clean potential markdown wrappers
                json_match = re.search(
                    r"```json\s*(\{.*?\})\s*```", text_out, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text_out  # Assume raw output is JSON

                parsed_json = json.loads(json_str)

                # Validate structure: Dict where values are lists of strings
                if isinstance(parsed_json, dict) and all(
                    isinstance(v, list) and all(isinstance(s, str) for s in v)
                    for v in parsed_json.values()
                ):
                    # Clean keys and keywords
                    keywords_categorized = {
                        k.strip(): [kw.strip().lower() for kw in v if kw.strip()]
                        for k, v in parsed_json.items()
                        if k.strip() and v
                    }  # Keep only non-empty categories/lists

                    if keywords_categorized:
                        total_kws = sum(len(v) for v in keywords_categorized.values())
                        print(
                            f"✅ Parsed {total_kws} keywords across {len(keywords_categorized)} categories from GPT."
                        )
                        return keywords_categorized
                    else:
                        print("⚠️ LLM returned valid JSON but no categories/keywords.")
                        # Fall through to return empty dict outside try block if needed
                else:
                    print(
                        f"⚠️ LLM output was not a valid Dict[str, List[str]] structure: {text_out[:100]}..."
                    )
                    # Fall through to retry or return empty

            except json.JSONDecodeError:
                print(
                    f"⚠️ LLM output was not valid JSON (attempt {attempt + 1}): {text_out[:100]}..."
                )
                # Fall through to retry or return empty
            except Exception as parse_e:
                # Catch any other unexpected parsing errors
                print(
                    f"⚠️ Error parsing/validating LLM JSON (attempt {attempt + 1}): {parse_e}"
                )
                # Fall through to retry or return empty

            # If parsing failed, keywords_categorized is still {}, loop will retry if possible

        except Exception as e:
            print(f"❌ Error (GPT attempt {attempt + 1}/{retries}): {str(e)[:100]}")
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
    video_df: pd.DataFrame, channel_name: str = "", niche: str = "", top_n: int = 15
) -> Dict[str, List[str]]:
    """Creates channel fingerprint using either Gemini or GPT, guided by the niche."""
    if video_df.empty:
        print("⚠️ Empty DataFrame")
        return {}

    print(f"📊 Processing {len(video_df)} videos for {channel_name}...")

    combined_texts = []
    for _, row in video_df.iterrows():
        title = str(row.get("title", ""))
        description = str(row.get("description", ""))
        clean_title = preprocess_text_for_llm(title)
        clean_desc = preprocess_text_for_llm(description[:200])
        if channel_name:
            clean_title = clean_title.replace(channel_name.lower(), "")
            clean_desc = clean_desc.replace(channel_name.lower(), "")
        combined_texts.append(clean_title * 2)  # weight titles higher
        combined_texts.append(clean_desc)

    full_text_blob = "\n---\n".join(filter(None, combined_texts))
    if not full_text_blob:
        print("⚠️ No text to analyze.")
        return {}

    # --- MODIFIED: Pass the niche to the keyword extractor ---
    keywords = extract_keywords_llm(
        full_text_blob, channel_name=channel_name, niche=niche
    )
    return keywords


# ============================================
# NEW FUNCTION: Language Detection
# ============================================
def detect_channel_language_llm(
    channel_description: str,
    video_titles: List[str],
    channel_name: str,
    max_chars: int = 1500,
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
        if not gpt_client:
            return "un"

        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        lang_code = response.choices[0].message.content.strip().lower()

        # Basic validation of the 2-letter code
        if len(lang_code) == 2 and re.match(r"^[a-z]{2}$", lang_code):
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
    kws = [k.strip() for k in kw_string.split(",") if k.strip()]
    return ", ".join(kws[:max_sample])


# ============================================
# NEW: Final "Extra Call" Competitor Check (v5 - Raw Data Only)
# ============================================
def is_direct_competitor_llm_final_check(
    seed_name: str,
    seed_keywords_str: str,  # <-- NEW: Pass seed's flat keyword string
    candidate_name: str,
    candidate_description: str,
    candidate_keywords_str: str,  # <-- Pass candidate's flat keyword string
    model_provider: str = "gpt",
    retries: int = 2,
) -> dict:
    """
    Uses GPT-4o-mini for a final, strict "Yes/No" check.
    This version IGNORES Niche/Format labels and infers from raw data.
    """

    # Get keyword samples from the flat strings
    seed_kw_sample = _get_keyword_sample_from_string(seed_keywords_str, max_sample=20)
    cand_kw_sample = _get_keyword_sample_from_string(
        candidate_keywords_str, max_sample=20
    )
    cand_desc_snippet = candidate_description[:1000] if candidate_description else "N/A"

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
                    return {
                        "is_competitor": False,
                        "confidence": "Low",
                        "reason": "GPT client not initialized.",
                    }

                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini",  # Use the cheap mini model
                    response_format={"type": "json_object"},  # Force JSON output
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150,
                )
                result_text = response.choices[0].message.content.strip()

                try:
                    result_json = json.loads(result_text)
                    if "is_competitor" in result_json:
                        print(
                            f"     ✅ LLM Final Check: {'Yes' if result_json['is_competitor'] else 'No'}. Reason: {result_json.get('reason', 'N/A')}"
                        )
                        return result_json
                except json.JSONDecodeError:
                    print(
                        f"  ⚠️ LLM Final Check returned invalid JSON: '{result_text}' (Attempt {attempt + 1})"
                    )

            else:
                print(f"  ❌ Unknown model provider '{model_provider}'")
                return {
                    "is_competitor": False,
                    "confidence": "Low",
                    "reason": "Unknown model provider.",
                }

        except Exception as e:
            print(
                f"  ❌ LLM Final Check Error (Attempt {attempt + 1}/{retries}): {str(e)[:100]}"
            )
            time.sleep(5 * (attempt + 1))

    return {
        "is_competitor": False,
        "confidence": "Low",
        "reason": "All API retries failed.",
    }


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
    verbose: bool = False,
) -> float:
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
            print(
                f"  📊 Seed: {len(seed_kw_list)} keywords | Candidate: {len(cand_kw_list)} keywords"
            )

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
            avg_seed_vec.reshape(1, -1), avg_cand_vec.reshape(1, -1)
        )[0][0]

        if verbose:
            print(f"  ✅ Keyword score: {final_score:.3f}")

        return round(float(final_score), 3)

    except Exception as e:
        print(f"  ❌ Error in keyword scoring: {str(e)[:100]}")
        return 0.0


def get_format_rules(client_format):
    """Retrieves specific prompt rules based on the checkbox input."""
    # Default to General if format not found
    config = FORMAT_CONFIG.get(client_format, FORMAT_CONFIG["General"])

    # Format the required keywords into a string
    req_kws = ", ".join([f'"{k}"' for k in config["keywords"]])

    return f"""
    3. **FORMAT ENFORCEMENT ({client_format}):**
       - {config["rules"]}
       - Every keyword MUST include one of these identifiers (or synonyms): {req_kws}
       
    EXAMPLES for {client_format}:
    {config["examples"]}
    """


def get_channel_fingerprint_oneshot(
    channel_name: str,
    dossier: str,
    client_format: str = "General",  # <--- NEW: User constraint (e.g. "Podcast")
    client_intent: str = "General",  # <--- NEW: User constraint (e.g. "Interviews")
    model_provider: str = "gpt-4o",  # <--- Default to high-intellect model
    retries: int = 3,
) -> dict:
    """
    Performs a one-shot analysis using the 'Meta-Block' data (Transcripts + Tags).
    Steered by Client Constraints to generate SOTA search keywords.
    """

    # --- 1. PREPARE THE "DOSSIER" (Input Construction) ---
    # We assume video_df['description'] already contains the "Meta-Block"
    # (Title + Category + Tags + Transcript Slice) prepared by the Phase 1 script.

    dynamic_format_rules = get_format_rules(client_format)

    prompt = f"""You are a YouTube Search Algorithm Expert specializing in competitor discovery.

    GOAL: Reverse engineer this channel to find its **Core Identity** and **SOTA Search Keywords**.

    --- CLIENT CONSTRAINTS ---
    TARGET FORMAT: "{client_format}" (PRIORITY #1)
    TARGET INTENT: "{client_intent}" (PRIORITY #2)

    --- DOSSIER ---
    \"\"\"
    {dossier}
    \"\"\"

    --- TASK PART 1: IDENTITY EXTRACTION ---
    Analyze the transcripts/metadata to define the channel's identity.
    1. **generated_niche**: The specific topic (e.g., "B2B Sales", "True Crime"). NOT "General".
    2. **generated_intent**: The goal like what the channel aims to achieve for viewers.

    --- TASK PART 2: SOTA KEYWORD GENERATION ---
    Generate 15-20 search keywords that maximize FORMAT precision.

    KEYWORD RULES:
    1. **OPTIMAL LENGTH:** 2-3 words maximum.
    2. **ARCHITECTURE:** [Broad Niche] + [Format Identifier].
    {dynamic_format_rules}

    --- OUTPUT FORMAT ---
    Return a SINGLE JSON object:
    {{
      "generated_niche": "...",
      "generated_intent": "...",
      "search_keywords": ["...", "..."]
    }}
    """

    for attempt in range(retries):
        try:
            if not gpt_client:
                print("❌ GPT client not initialized.")
                return {}

            response = gpt_client.chat.completions.create(
                model=model_provider,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a YouTube Search Algorithm.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            result_text = response.choices[0].message.content.strip()
            data = json.loads(result_text)

            if "search_keywords" in data and "generated_niche" in data:
                print(f"✅ SOTA Analysis successful for {channel_name}")
                return data

            print(f"⚠️ Invalid JSON structure. Retrying...")

        except Exception as e:
            print(f"  ⚠️ LLM Error (Attempt {attempt + 1}): {e}")
            time.sleep(2)

    print(f"❌ All retries failed for {channel_name}")
    return {}


def calculate_profile_score_llm_holistic(
    seed_profile: dict,
    candidate_profile: dict,
    seed_channel_name: str,
    candidate_channel_name: str,
    model_provider: str = "gpt",
) -> dict:  # --- CHANGED: Returns dict ---
    """
    V3 - "Holistic Analyst" Model (Replaces rigid calculator)
    ...
    """

    # --- Default error return ---
    error_output = {
        "similarity_score": 0.0,
        "competitor_score": 0.0,
        "audience_overlap_score": 0.0,
        "reason": "API or parsing error.",
    }

    # --- Extract profile fields (anonymized for the prompt) ---
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

    # --- Build the new "Holistic Analyst" Prompt ---
    prompt = f"""You are an expert YouTube analyst. Your job is to compare two anonymized channel profiles and provide a holistic competitor analysis.
    
    **CRITICAL RULE:** Do NOT use any outside knowledge. Base your analysis *ONLY* on the profile data provided for "{seed_channel_name}" and "{candidate_channel_name}".

    ---
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
    ---

    **TASK:**
    Provide a JSON object with four keys:
    1.  `"similarity_score"`: (0.0-1.0) How *alike* are they on paper? (Based on niche, format, intent).
    2.  `"competitor_score"`: (0.0-1.0) How *directly* do they compete for the same user need? (Penalized heavily by ideology/intent mismatch).
    3.  `"audience_overlap_score"`: (0.0-1.0) How likely is Channel A's audience to *also* watch Channel B? (Driven by audience, format, and niche compatibility).
    4.  `"reason"`: A one-sentence explanation for your scores, comparing the key profile points.

    **REASONING PRINCIPLES (Apply in this order):**

    1.  **IDEOLOGY (CRITICAL):**
        -   If `ideology` is a direct opposite (e.g., "Progressive/Left" vs "Conservative/Right") -> They are antagonists. `competitor_score` and `audience_overlap_score` MUST be ~0.1. `similarity_score` can still be high (e.g., 0.7) if they are both political commentators.
        -   If one is political (Left/Right) and one is "N/A - Non-Political" -> They are not direct competitors. `competitor_score` MUST be low (~0.3). `audience_overlap_score` can be high (0.5-0.8) if `format` and `target_audience` match (e.g., "Curious Learners").
        -   If ideologies are compatible or both are "N/A", proceed.

    2.  **FORMAT & INTENT (Drives `competitor_score`):**
        -   Incompatible formats (e.g., "Podcast" vs. "Educational Tutorial") MUST have a low `competitor_score`.
        -   Highly compatible (e.g., "Video Essay" ↔ "Explainer Documentary") means a high `competitor_score` (if ideology also matches).

    3.  **NICHE & AUDIENCE (Drives `audience_overlap_score`):**
        -   If Formats match but Niches are different (e.g., "Scam Investigation" vs "Geopolitics"), `competitor_score` is low, but `audience_overlap_score` can be very high if `target_audience` (e.g., "Curious Learners") is the same.

    **EXAMPLE REASONING (This is how you must think):**

    -   **Case 1: (Progressive vs. Progressive)**
        -   A: {{ "format": "Video Essay", "ideology": "Progressive/Left", "niche": "Media Critique" }}
        -   B: {{ "format": "Video Essay", "ideology": "Progressive/Left", "niche": "Political Analysis" }}
        -   *JSON Output:* {{ "similarity_score": 0.9, "competitor_score": 0.9, "audience_overlap_score": 0.95, "reason": "Perfect ideology and format match; they are direct competitors serving a very similar audience." }}

    -   **Case 2: (Progressive vs. Conservative)**
        -   A: {{ "format": "Video Essay", "ideology": "Progressive/Left" }}
        -   B: {{ "format": "Talking-Head Analysis", "ideology": "Conservative/Right" }}
        -   *JSON Output:* {{ "similarity_score": 0.7, "competitor_score": 0.1, "audience_overlap_score": 0.1, "reason": "Ideological opposites. They are antagonists, not competitors, and serve different audiences." }}

    -   **Case 3: (Progressive vs. Non-Political)**
        -   A: {{ "format": "Video Essay", "ideology": "Progressive/Left", "niche": "Media Critique" }}
        -   B: {{ "format": "Investigative Documentary", "ideology": "N/A - Non-Political", "niche": "Scam Investigation" }}
        -   *JSON Output:* {{ "similarity_score": 0.8, "competitor_score": 0.35, "audience_overlap_score": 0.85, "reason": "Formats and audiences are very similar, but they are not direct competitors as one is political and the other is non-political investigation." }}

    Return ONLY a single, valid JSON object.
    """

    try:
        if not gpt_client:
            print("  ❌ GPT client not initialized.")
            return error_output

        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=250,
        )
        result_text = response.choices[0].message.content.strip()

        # Parse the full JSON object
        try:
            result_json = json.loads(result_text)
            # Validate the required keys
            if (
                "competitor_score" in result_json
                and "audience_overlap_score" in result_json
                and "reason" in result_json
                and "similarity_score" in result_json
            ):
                return result_json
            else:
                print(f"  ⚠️ LLM JSON missing required keys: {result_text}")
                return error_output
        except json.JSONDecodeError:
            print(f"  ⚠️ LLM returned invalid JSON: {result_text}")
            return error_output

    except Exception as e:
        print(f"  ❌ Profile score error: {str(e)[:100]}")
        return error_output


def calculate_profile_score_llm_with_keywords(
    seed_profile: dict,
    candidate_profile: dict,
    seed_keywords: dict,
    candidate_keywords: dict,
    seed_channel_name: str,
    candidate_channel_name: str,
    model_provider: str = "gpt",
) -> dict:
    """
    NEW V4 - "Audience Match" Model

    Asks the LLM to act as a content strategist, deciding if the
    Seed audience would *love* the Candidate channel, considering
    profile, format, and keyword mismatches.

    Returns a dict, e.g.,
    {
        "audience_match_score": 0.85,
        "reason": "High match. Formats and ideologies align,
                   and keywords show a strong thematic overlap."
    }
    """

    # --- Default error return ---
    error_output = {"audience_match_score": 0.0, "reason": "API or parsing error."}

    # Use pprint to format the dicts nicely for the prompt
    s_keywords_str = pprint.pformat(seed_keywords)
    c_keywords_str = pprint.pformat(candidate_keywords)

    prompt = f"""You are an expert YouTube Content Strategist. Your goal is to find new channels for your audience.
    
    You are given data for two channels:
    1.  **Seed Channel ("{seed_channel_name}")**: This is the channel our audience *already loves*.
    2.  **Candidate Channel ("{candidate_channel_name}")**: This is a new channel we are thinking of recommending.
    
    **TASK:**
    Based *only* on the data below, provide a JSON object with two keys:
    1.  `"audience_match_score"`: (0.0 - 1.0) How likely is the Seed Channel's audience to *also love* the Candidate Channel?
    2.  `"reason"`: A one-sentence explanation for your score.
    
    ---
    DATA FOR SEED CHANNEL ("{seed_channel_name}")
    ---
    
    **Keywords:**
    {s_keywords_str}
    
    ---
    DATA FOR CANDIDATE CHANNEL ("{candidate_channel_name}")
    ---
    
    **Keywords:**
    {c_keywords_str}
    
    ---
    **SCORING RULES (CRITICAL):**
    Analyze indepth the keywords of both seed channel and candidate channel and then score them and give a reason that how likely are they related? And target same audience? 
        
    Return ONLY a single, valid JSON object.
    """

    try:
        if not gpt_client:
            print("  ❌ GPT client not initialized.")
            return error_output

        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )
        result_text = response.choices[0].message.content.strip()

        try:
            result_json = json.loads(result_text)
            if "audience_match_score" in result_json and "reason" in result_json:
                return result_json
            else:
                print(f"  ⚠️ LLM JSON missing required keys: {result_text}")
                return error_output
        except json.JSONDecodeError:
            print(f"  ⚠️ LLM returned invalid JSON: {result_text}")
            return error_output

    except Exception as e:
        print(f"  ❌ Profile score error: {str(e)[:100]}")
        return error_output


def get_vector_from_texts(texts: list[str]) -> np.ndarray | None:
    """
    Takes a list of text strings, cleans them, gets embeddings,
    and returns the single averaged vector.
    """
    if not texts:
        return None
    cleaned_texts = [preprocess_text_for_llm(text) for text in texts if text]
    if not cleaned_texts:
        return None
    try:
        embeddings = embedding_model.encode(cleaned_texts)
        avg_vector = np.mean(embeddings, axis=0)
        return avg_vector
    except Exception as e:
        print(f"  ⚠️ Error encoding texts: {e}")
        return None


def calculate_cosine_similarity(vec_a, vec_b) -> float:
    """Calculates cosine similarity between two averaged vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    try:
        return cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0]
    except Exception as e:
        print(f"  ⚠️ Error in cosine similarity: {e}")
        return 0.0


# DEPRECATED: This function used local Sentence Transformers (now using OpenAI embeddings)
def calculate_matrix_average_similarity(
    texts_a: list[str], texts_b: list[str]
) -> float:
    """
    Computes the full pairwise matrix and returns the average of all scores.
    DEPRECATED: Replaced by OpenAI embeddings in Phase 3 Step 3B
    """
    if not embedding_model or not texts_a or not texts_b:
        return 0.0

    # Clean texts
    texts_a = [preprocess_text_for_llm(t) for t in texts_a if t]
    texts_b = [preprocess_text_for_llm(t) for t in texts_b if t]
    if not texts_a or not texts_b:
        return 0.0

    # try:
    #     embed_a = embedding_model.encode(texts_a)
    #     embed_b = embedding_model.encode(texts_b)
    #
    #     # This creates the (e.g.) 30x20 matrix
    #     similarity_matrix = cosine_similarity(embed_a, embed_b)
    #
    #     # Take the mean of the entire matrix
    #     avg_score = np.mean(similarity_matrix)
    #     return float(avg_score)
    # except Exception as e:
    #     print(f"  ⚠️ Error in matrix avg: {e}")
    return 0.0  # Not implemented for current pipeline


def get_channel_tier_gpt(
    seed_name: str,
    candidate_name: str,
    candidate_titles: list,
    seed_profile_str: str,
    seed_categories_str: str,
    seed_keywords_str: str,
    retries: int = 3,
) -> dict:
    """
    Calls Gemini to perform the "Client's Gut Check" and assign a tier.
    """

    # Create the text blob of candidate titles
    titles_blob = "\n- ".join(candidate_titles)

    # --- This is the new SOTA prompt ---
    # --- PROMPT START ---

    prompt = f"""You are an expert YouTube analyst and content strategist.
    Your goal is to help me find *exact* competitors for my seed channel by analyzing a candidate.

    ---
    **SEED CHANNEL CONTEXT**
    ---
    My seed channel's name is **"{seed_name}"**.

    My Seed Channel's **Profile** (its core identity):
    {seed_profile_str}

    My Seed Channel's **Keywords** (the topics it covers):
    {seed_keywords_str}

    ---
    **CANDIDATE CHANNEL ANALYSIS**
    ---
    I will now provide the most recent video titles from a candidate channel.
    Candidate Name: **"{candidate_name}"**
    Candidate Titles:
    - {titles_blob}

    ---
    **YOUR TASK (Must follow these 3 steps):**
    ---
    1.  **Candidate Profile Generation:** Based *only* on the candidate's titles, infer its 'Niche', 'Format', and 'Intent'.
    2.  **Comparative Analysis:** Compare the Seed's Profile (Niche, Intent, Format) against the Candidate's inferred Profile.
    3.  **Tier Assignment:** Use the definitions and rules below to assign a tier.

    ---
    **TIERING RULES (NEW - READ CAREFULLY)**
    ---
    This is a test of **Niche & Intent**, not just topics.

    * **Tier 1 (Direct Competitor):**
        * **Niche MATCH:** Seed and Candidate have the *same* core niche (e.g., 'Societal Critique' vs 'Societal Critique').
        * **Intent MATCH:** Both channels have the *same* goal (e.g., 'To critique' vs 'To critique').
        * (Format must also be similar, e.g., 'Video Essay').

    * **Tier 2 (Niche Competitor):**
        * **Niche is RELATED:** The niches are in the same *family* but not identical (e.g., Seed is 'Societal Critique', Candidate is 'Business Case Studies').
        * **Intent MATCH:** Both have a similar goal (e.g., 'To explain').
        * **Format MATCH:** (e.g., 'Video Essay' vs 'Explainer Documentary').

    * **Tier 3 (Audience Overlap / "The News" Tier):**
        * **Niche MATCH:** The *topic* is the same (e.g., 'Politics').
        * **Format MISMATCH:** The *format* is completely different (e.g., Seed is 'Video Essay', Candidate is 'Daily News Clips' or 'Livestream').
        * This tier is for channels that cover the same topics but in a different, non-competitor format.

    * **Tier 4 (Irrelevant / "The Movie Cynic" Tier):**
        * **Niche MISMATCH:** The niches are fundamentally different.
        * **This is the most important filter.** If the niche is wrong, it's Tier 4, *even if the topic seems related*.
        * (e.g., Seed is 'Societal Critique using movies' vs. Candidate is 'Movie Reviews'. This is a NICHE MISMATCH.)
        * (e.g., Seed is 'Documentary' vs. Candidate is 'Gaming Walkthrough' or 'Vlog'.)

    ---
    **OUTPUT FORMAT**
    ---
    Return ONLY a single, valid JSON object with two keys: "tier" and "reason".
    In your "reason" text, you **MUST NOT** use double quotes ("). Use single quotes (') instead.

    ---
    **EXAMPLES (Follow this new Niche/Intent logic)**
    ---

    **Example (Tier 1 - Direct Competitor):**
    {{
    "tier": 1,
    "reason": "Niche & Intent Match. Seed's niche is 'Explanatory Journalism'. Candidate's inferred niche is 'In-depth Analysis', and their intents to 'explain' align. This is a direct competitor."
    }}

    **Example (Tier 2 - Niche Competitor):**
    {{
    "tier": 2,
    "reason": "Niche is Related. Seed's niche is 'Societal Critique', but Candidate's inferred niche is 'Business Case Studies'. While both are 'Video Essays' that 'explain', their core niches are adjacent, not identical."
    }}

    **Example (Tier 3 - Format Mismatch):**
    {{
    "tier": 3,
    "reason": "Format Mismatch. Seed's format is 'Video Essay'. Candidate's titles ('Biden Speaks', 'Market Update') infer a 'Daily News Clip' format. Although the *topic* is 'Politics', the format is not a competitor."
    }}

    **Example (Tier 4 - Irrelevant Niche):**
    {{
    "tier": 4,
    "reason": "Niche Mismatch. Seed's niche is 'Societal Critique'. Candidate's titles ('When Nepo-Babies Self-Destruct', 'Marvel - Death Of An Empire') infer a 'Movie Review' niche. The *intent* is to review, not to critique society. This is irrelevant."
    }}

    Provide ONLY the JSON output for the candidate.
    """

    # --- PROMPT END --- """
    for attempt in range(retries):
        try:
            # --- Call GPT ---
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},  # Force JSON output
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,  # 500 tokens is plenty for the reason
            )

            result_text = response.choices[0].message.content.strip()
            parsed_json = json.loads(result_text)

            if "tier" in parsed_json and "reason" in parsed_json:
                return parsed_json  # Success!
            else:
                print(f"  ⚠️  JSON missing 'tier' or 'reason' keys.")
                return {"tier": -1, "reason": "ERROR: Malformed JSON."}

        except Exception as e:
            error_str = str(e)
            print(f"  ❌ LLM Error: {error_str[:150]}")
            if "rate_limit_exceeded" in error_str:
                print(
                    f"  ...RATE LIMIT HIT. Sleeping for 20 seconds (Attempt {attempt + 1}/{retries})..."
                )
                time.sleep(20)
                continue  # Try again

            # For other errors, fail
            return {"tier": -1, "reason": f"ERROR: {error_str[:100]}"}

    # Fallback if loop finishes
    return {"tier": -1, "reason": "ERROR: All retries failed."}


def get_openai_embedding(text_list: list, model="text-embedding-3-small"):
    """
    Generates embeddings using OpenAI's model (High Token Limit: ~8k).
    Returns a single AVERAGED vector for the input list.
    """
    if not gpt_client or not text_list:
        return None

    # 1. Clean and Validate Inputs
    valid_texts = [
        str(t).replace("\n", " ") for t in text_list if t and len(str(t)) > 10
    ]

    if not valid_texts:
        return None

    try:
        # 2. API Call (Batched)
        # OpenAI can handle multiple inputs in one request
        response = gpt_client.embeddings.create(input=valid_texts, model=model)

        # 3. Extract Vectors
        vectors = [item.embedding for item in response.data]

        # 4. Average them to get the "Channel Vector"
        # This ensures we represent ALL 3 videos equally
        avg_vector = np.mean(vectors, axis=0)

        return avg_vector

    except Exception as e:
        print(f"  ❌ OpenAI Embedding Error: {e}")
        return None


# --- NEW: SMART SLICER (Phase 1 Logic) ---
def get_smart_slice(text, chunk_size=1500):
    head = text[:4500]
    
    # Middle section
    t_len = len(text)
    mid_start = t_len // 2
    mid = text[mid_start : mid_start + 2000]
    
    # Tail section
    tail = text[-2000:]
    
    return f"{head}\n...\n{mid}\n...\n{tail}"


def verify_format_llm(candidate_data, client_format):
    """
    Strictly checks if the content matches the Client Format using Deep Data.
    Feeds 3 Videos x (Smart Slice).
    """
    name = candidate_data.get("Discovered_Channel_Name")
    deep_json = candidate_data.get("Deep_Scan_Data", "[]")

    try:
        deep_data = json.loads(deep_json)
    except:
        deep_data = []

    if not deep_data:
        return {"is_format_match": False, "reason": "No scan data", "confidence": 0.0}

    # 1. Build Dossier
    dossier = ""
    for i, vid in enumerate(deep_data):
        title = vid.get("title", "Unknown")
        duration = vid.get("duration", 0)

        # Use existing smart slice function
        raw_transcript = vid.get("caption_tracks", "")
        # Ensure get_smart_slice is available
        transcript_sample = get_smart_slice(raw_transcript)

        dossier += f"""
        === VIDEO {i + 1} ===
        TITLE: {title}
        DURATION: {duration}s
        TRANSCRIPT SAMPLE:
        {transcript_sample}
        ===================
        """

    # 2. Get Validation Rules (Using the NEW Config)
    config = FORMAT_VALIDATION_CONFIG.get(
        client_format, FORMAT_VALIDATION_CONFIG["General"]
    )

    pos_signals = config["positive_signals"]
    # neg_constraints = config["negative_constraints"]
    structure = config["structural_cue"]

    # 3. Modular Prompt with False Positive Protection
    prompt = f"""You are a Strict Format Validator.

CLIENT REQUIREMENT: "{client_format}"

TASK: Analyze the video samples below. Does this channel STRICTLY produce "{client_format}" content?

--- VALIDATION RULES ---
1. POSITIVE SIGNALS (Must have): {pos_signals}
2. STRUCTURAL CUE (Look for this flow): {structure}


DOSSIER:
{dossier}

OUTPUT JSON:
{{
    "is_format_match": true/false,
    "confidence": 0.0 to 1.0,
    "reason": "Brief explanation citing specific video evidence and structure."
}}
"""

    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "is_format_match": False,
            "reason": f"LLM Error: {e}",
            "confidence": 0.0,
        }


def get_candidate_fingerprint_independent(
    dossier: str, model_provider: str = "gpt-4o-mini", retries: int = 3
) -> dict:
    """
    PHASE 3 SPECIAL: Independent Candidate Analysis.
    Generates Niche, Intent, and Keywords based *ONLY* on the dossier.
    NO client constraints are passed, ensuring an honest classification.
    """

    print("   🧠 Generating Independent/Honest Fingerprint...")

    prompt = f"""You are a YouTube Search Algorithm Expert.
    
    GOAL: Analyze the provided channel DOSSIER and reverse-engineer its **True Identity**.
    
    --- DOSSIER ---
    \"\"\"
    {dossier}
    \"\"\"

    --- TASK ---
    1. **Identity**: Determine the *actual* Niche and Intent of this channel based strictly on the content provided.
    2. **Keywords**: Generate 15-20 search keywords that a user would type to find *this specific channel*.
    
    CRITICAL RULES:
    - **BE HONEST:** If the content is a Vlog, use keywords like "vlog", "daily life".
    - **BE HONEST:** If the content is News, use keywords like "news clips", "updates".
    - **BE HONEST:** If the content is a Podcast, use keywords like "podcast", "interview".
    - Do NOT force any specific format. Describe exactly what you see.

    --- OUTPUT FORMAT ---
    Return a SINGLE JSON object:
    {{
      "generated_niche": "The specific topic (e.g. 'Tech News', 'Daily Vlogging')",
      "generated_intent": "The psychological goal (e.g. 'To inform quickly', 'To entertain')",
      "search_keywords": ["keyword 1", "keyword 2", ...]
    }}
    """

    for attempt in range(retries):
        try:
            if not gpt_client:
                return {}

            response = gpt_client.chat.completions.create(
                model=model_provider,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an honest content analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            data = json.loads(response.choices[0].message.content.strip())

            if "search_keywords" in data and "generated_niche" in data:
                return data

        except Exception as e:
            print(f"  ⚠️ LLM Error (Attempt {attempt + 1}): {e}")
            time.sleep(1)

    return {}
