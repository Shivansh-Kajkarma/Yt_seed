import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. SETUP: Load .env and define base directory ---
# This script is in Manual_testing/, so BASE_DIR is two levels up
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

# --- 2. CONFIGURE GEMINI ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ CRITICAL: GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # --- MODIFIED: Upped tokens slightly for niche/keyword prompts ---
    generation_config = genai.GenerationConfig(
        max_output_tokens=4000, # Increased for the full JSON
        temperature=0.1
    )
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    gemini_model = genai.GenerativeModel(
        'gemini-2.0-flash-exp', # Using 1.5-flash as it's the modern equivalent
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print("✅ Gemini model configured: gemini-1.5-flash")
except Exception as e:
    print(f"❌ Gemini configuration failed: {e}")
    sys.exit(1)


# --- 3. HELPER FUNCTION (to find your data) ---
def find_latest_file_path(directory: Path, prefix: str, suffix: str) -> Path | None:
    """Finds the most recent file in a directory matching a pattern."""
    try:
        latest_file = max(
            directory.glob(f"{prefix}*{suffix}"),
            key=os.path.getctime
        )
        return latest_file
    except ValueError:
        return None # No files found

# --- 4. DEFINE YOUR PROMPT TEMPLATES HERE ---

# PROMPT 1: Your old "Search Query" prompt that gave bad results
PROMPT_A_GEMINI_1 = """You are an expert YouTube channel analyst.
    Analyze the provided raw data (channel name, description, video titles, video descriptions) 
    and extract a complete channel profile and its content themes.

    RAW DATA TO ANALYZE:
    \"\"\"
    {truncated_content}
    \"\"\"

    TASK: Return a single, valid JSON object with two top-level keys: "profile" and "keywords".

    ---
    PART 1: "profile"
    ---
    This key must contain an object with these 6 sub-keys.
    **CRITICAL:** For each key, provide a **concise, descriptive 1-sentence summary**, not just 2-3 words.

    1.  "niche": What is the channel's primary subject matter and unique angle?
    2.  "format": What is the primary video style and production value? (e.g., "Video essay", "Explainer documentary")
    3.  "intent": What is the channel's main GOAL? (e.g., "To explain complex topics", "To persuade with a specific viewpoint")
    4.  "speaker": Who is presenting? (e.g., "A solo creator (anonymous)", "A media company with various hosts")
    5.  "ideology": What is the channel's clear political or social bias? (e.g., "Progressive", "Skeptical and Critical")
    6.  "target_audience": Who is the ideal viewer? (e.g., "Critical thinkers", "Curious learners")

    ---
    PART 2: "keywords"
    ---
    Generate 2-4 most dominant content **THEMES** (categories).
    Under each theme, list 5-7 **analytical keywords and search queries** that describe this topic.

    **CRITICAL (THE GOAL):**
    Your goal is to *CATEGORIZE* the content, not to copy its emotional language.
    The keywords should be the *academic topic* or *niche* of the videos.
    We are looking for the *literal search queries* a person would use to find *other channels in this same niche*.

    **CRITICAL (THE RULES):**
    1.  **Reflect the Profile:** The themes MUST be analytical summaries of the 'niche' and 'intent' from PART 1.
    2.  **Be Analytical, Not Sensational:** The keywords should describe the *topic*, not the *clickbait*.
    3.  **IGNORE SPONSORS:** Video descriptions are often just ads ("TryHackMe", "FÜM", etc.). IGNORE this sponsor text completely and focus only on the high-signal titles and channel description.

    **Examples of GOOD keywords (Analytical & Topical):**
    ✅ "political commentary"
    ✅ "celebrity scandal analysis"
    ✅ "tech industry critique"
    ✅ "internet culture drama"
    ✅ "corporate controversy explained"
    ✅ "video essay [topic]"
    ✅ "the problem with [company]"

    **Examples of BAD keywords (Too Sensational/Vague):**
    ❌ "society destroyed" (Too sensational, bad search results)
    ❌ "celebrity exposed" (Too generic, will return tabloids)
    ❌ "government conspiracy" (Too broad)
    ❌ "cultural critique" (Too academic, not a search query)
    ❌ "company destroyed" (Too emotional)

    ---
    EXAMPLE OUTPUT (This is the style you must follow):
    {{
    "profile": {{
        "niche": "Societal and Cultural Critique with a skeptical lens",
        "format": "Long-form narrative video essays with high-production visuals",
        "intent": "To expose and critique societal flaws and media narratives",
        "speaker": "Anonymous solo creator",
        "ideology": "Skeptical and Critical of mainstream narratives",
        "target_audience": "Critical thinkers and young adults (18-35)"
    }},
    "keywords": {{
        "Political Commentary & Analysis": [
            "political commentary",
            "political news analysis",
            "political scandal explained",
            "controversial political figures",
            "government failures explained"
        ],
        "Celebrity & Entertainment Critique": [
            "celebrity scandal analysis",
            "entertainment industry critique",
            "celebrity controversy explained",
            "hollywood industry analysis",
            "celebrity downfall analysis"
        ],
        "Internet Culture & Creator Commentary": [
            "youtube creator drama",
            "influencer controversy analysis",
            "internet personality critique",
            "social media culture critique",
            "creator community drama"
        ]
    }}
    }}
    --- (End of Example) ---

    OUTPUT:
    Return ONLY the valid JSON for the channel in the "RAW DATA" section.
    """


# PROMPT 2: Our new "Categories" prompt (the "Perplexity" style)
PROMPT_B_GEMINI_2 = """You are an expert YouTube channel analyst.
Analyze the provided raw data (channel name, description, and video titles).
The VIDEO TITLES are the most important data.

RAW DATA TO ANALYZE:
\"\"\"
{truncated_content}
\"\"\"

TASK: Return a single, valid JSON object with two top-level keys: "profile" and "keywords".

---
PART 1: "profile" (NEW UPGRADED VERSION)
---
This key must contain an object with a detailed analysis of the channel.
Provide rich, descriptive sentences, not just 2-3 word labels.

1.  "profile_summary": A 1-2 sentence summary of the channel's purpose and content, as if for a "what to watch" guide.
2.  "likely_niche": The channel's specific, high-level category (e.g., "Explanatory Journalism", "Cultural Critique", "Business Case Studies", "Scam Investigation").
3.  "video_style": A description of the production style (e.g., "High-production video essays with motion graphics", "Talking-head analysis", "Investigative documentary").
4.  "intent": The channel's main GOAL (e.g., "To inform and educate on complex topics", "To persuade with critical analysis", "To expose wrongdoing").
5.  "target_audience": A description of the primary demographic (e.g., "Curious learners", "Politically-engaged young adults", "Tech enthusiasts").
---
PART 2: "keywords" (SOTA "CATEGORIES" VERSION)
---
Generate 3-4 dominant **Content Pillars** or **Semantic Categories** for this channel.
Under each category, list 5-10 keywords.

**CRITICAL (THE GOAL):**
These keywords should be the **TOPICS** and **NICHES** that describe the content, not "search queries." The goal is to identify *what* the channel is about.

**CRITICAL (THE RULES):**
1.  **PRIORITIZE VIDEO TITLES:** Base your analysis **PRIMARILY on the VIDEO TITLES**. The titles are the *most accurate* signal.
2.  **Use Channel Description as Context:** Use the channel description *only* as secondary context.
3.  **Reflect the Profile:** The categories MUST reflect the 'likely_niche' and 'intent' from PART 1. (e.g., if 'intent' is 'To expose', categories should be 'Scam Investigation').

**Examples of GOOD categories & keywords (Descriptive Topics):**
✅ "Celebrity & Entertainment Critique"
✅ "Internet Culture & Creator Commentary"
✅ "Tech & Business Critique"
✅ "Business Documentaries"

**Examples of BAD keywords (Sensational / Too Vague):**
❌ "society destroyed"
❌ "celebrity exposed"
❌ "government conspiracy"

---
EXAMPLE OUTPUT (This is the new style you must follow):
{{
  "profile": {{
    "profile_summary": "Vox is a news and opinion channel that aims to explain complex issues and make sense of the world.",
    "likely_niche": "Explanatory journalism and in-depth analysis",
    "video_style": "Well-researched, visually engaging video essays with motion graphics",
    "intent": "To inform and educate viewers on complex topics, encouraging critical thinking",
    "target_audience": "Individuals interested in understanding the underlying causes of current events"
  }},
  "keywords": {{
    "US Politics & Policy Analysis": [
      "US political system explained",
      "American policy analysis",
      "US elections analysis",
      "political polarization in America"
    ],
    "Economics & Business Trends": [
      "economic inequality explained",
      "business model analysis",
      "future of work",
      "corporate power explained"
    ],
    "Social Issues & Cultural Trends": [
      "social justice issues explained",
      "cultural trends analysis",
      "internet culture critique",
      "media landscape analysis"
    ]
  }}
}}
--- (End of Example) ---

OUTPUT:
Return ONLY the valid JSON for the channel in the "RAW DATA" section.
"""

PROMPT_C_PERPLEX_3="""You are an expert YouTube channel analyst.
    Analyze the provided raw data (channel name, description, video titles, video descriptions) 
    and extract a complete channel profile and its content themes.

    RAW DATA TO ANALYZE:
    \"\"\"
    {truncated_content}
    \"\"\"

    TASK: Return a single, valid JSON object with two top-level keys: "profile" and "keywords".

    ---
    PART 1: "profile"
    ---
    Extract a comprehensive channel profile with the following fields:
    
    {{
        "channel_name": "Extracted channel name from data",
        "description": "2-3 sentence description of what the channel does, based on video titles and descriptions. Be specific about format and intent.",
        "likely_niche": "The primary niche/category (e.g., 'Educational Documentary', 'Political Commentary', 'Entertainment Analysis')",
        "target_audience": "Who watches this content? Be specific about demographics and interests (e.g., '18-35 year old intellectually curious individuals seeking evidence-based explanations')",
        "video_style": "Describe the visual/production style (e.g., 'Animated graphics with expert interviews', 'Talking-head commentary with B-roll', 'Documentary-style narrative')",
        "format": "Primary format (e.g., 'Educational Documentary', 'Commentary Analysis', 'Video Essay', 'Investigative Journalism')",
        "perspective": "Channel's perspective/tone (e.g., 'Balanced and evidence-based', 'Skeptical and critical', 'Comedic but informative', 'Sensationalist and contrarian')",
        "intent": "What is the channel trying to achieve? (e.g., 'To inform and educate', 'To entertain and critique', 'To expose corruption', 'To explain how things work')"
    }}

    **CRITICAL RULES FOR PROFILE:**
    1. **Be Specific, Not Generic:** Avoid one-word descriptions. Use full sentences.
    2. **Include Format:** Distinguish between "Documentary", "Commentary", "Analysis", "Podcast", etc.
    3. **Include Perspective/Tone:** This helps differentiate competitors (e.g., "balanced" vs "skeptical" vs "comedic").
    4. **Be Analytical:** Focus on structure and intent, not clickbait language.
    5. **Make It Distinct:** The profile should make this channel distinguishable from other channels in similar niches.

    ---
    PART 2: "keywords"
    ---
    Generate 2-4 most dominant content **THEMES** (categories).
    Under each theme, list 5-7 **analytical keywords and search queries** that describe this topic.

    **CRITICAL (THE GOAL):**
    Your goal is to *CATEGORIZE* the content, not to copy its emotional language.
    The keywords should be the *academic topic* or *niche* of the videos.
    We are looking for the *literal search queries* a person would use to find *other channels in this same niche*.

    **CRITICAL (THE RULES):**
    1.  **Reflect the Profile:** The themes MUST be analytical summaries of the 'niche' and 'intent' from PART 1.
    2.  **Be Analytical, Not Sensational:** The keywords should describe the *topic*, not the *clickbait*.
    3.  **IGNORE SPONSORS:** Your code already filters out video descriptions if they are ads. This is just a reminder to focus on the high-signal titles.

    **Examples of GOOD keywords (Analytical & Topical):**
    ✅ "political commentary"
    ✅ "celebrity scandal analysis"
    ✅ "tech industry critique"
    ✅ "internet culture drama"
    ✅ "corporate controversy explained"
    ✅ "video essay [topic]"
    ✅ "the problem with [company]"

    **Examples of BAD keywords (Too Sensational/Vague):**
    ❌ "society destroyed" (Too sensational, bad search results)
    ❌ "celebrity exposed" (Too generic, will return tabloids)
    ❌ "government conspiracy" (Too broad)
    ❌ "cultural critique" (Too academic, not a search query)
    ❌ "company destroyed" (Too emotional)

    ---
    EXAMPLE OUTPUT (This is the style you must follow):
    {{
        "profile": {{
            "channel_name": "Vox",
            "description": "Vox is an explanatory journalism channel that creates well-researched videos with animated graphics on politics, economics, science, and culture. They aim to explain complex issues by exploring their underlying causes and different perspectives.",
            "likely_niche": "Educational documentary-style explanatory journalism",
            "target_audience": "Intellectually curious individuals (18-35, college-educated or equivalent) seeking evidence-based, nuanced explanations of complex issues",
            "video_style": "Animated graphics-heavy, visually engaging educational format with expert interviews and motion design",
            "format": "Documentary-style educational explainer",
            "perspective": "Balanced, evidence-based, non-partisan analytical approach",
            "intent": "To inform and educate viewers through rigorous research and visual storytelling, encouraging critical thinking and deeper understanding"
        }},
        "keywords": {{
            "Political Commentary & Analysis": [
                "political commentary",
                "political news analysis",
                "political scandal explained",
                "controversial political figures",
                "government failures explained"
            ],
            "Celebrity & Entertainment Critique": [
                "celebrity scandal analysis",
                "entertainment industry critique",
                "celebrity controversy explained",
                "hollywood industry analysis",
                "celebrity downfall analysis"
            ],
            "Internet Culture & Creator Commentary": [
                "youtube creator drama",
                "influencer controversy analysis",
                "internet personality critique",
                "social media culture critique",
                "creator community drama"
            ]
        }}
    }}
    --- (End of Example) ---

    OUTPUT:
    Return ONLY the valid JSON for the channel in the "RAW DATA" section.
    """

PROMPT_D_GPT_4 = """You are an expert YouTube channel analyst.
    Analyze the provided raw data (channel name, description, video titles, video descriptions) 
    and extract a complete channel profile and its content themes.

    RAW DATA TO ANALYZE:
    \"\"\"
    {truncated_content}
    \"\"\"

    TASK: Return a single, valid JSON object with two top-level keys: "profile" and "keywords".

    ---
    PART 1: "profile"
    ---
    Analyze the provided data and produce a detailed, *holistic channel profile* in this format:

    {{
    "channel_name": "...",
    "description": "...",
    "likely_niche": "...",
    "target_audience": "...",
    "video_style": "...",
    "intent": "..."
    }}

    Guidelines:
    - Be objective and descriptive (avoid emotion or bias).
    - Reflect the channel's *core identity* and *niche specialization*.
    - The "intent" must express **why** this channel creates content, not what it posts.
    - Keep tone like a professional media analyst, not a YouTuber or marketer.
    - Use complete sentences and 2-3 lines per field.

    ---
    PART 2: "keywords"
    ---
    Generate 2-4 most dominant content **THEMES** (categories).
    Under each theme, list 5-7 **analytical keywords and search queries** that describe this topic.

    **CRITICAL (THE GOAL):**
    Your goal is to *CATEGORIZE* the content, not to copy its emotional language.
    The keywords should be the *academic topic* or *niche* of the videos.
    We are looking for the *literal search queries* a person would use to find *other channels in this same niche*.

    **CRITICAL (THE RULES):**
    1.  **Reflect the Profile:** The themes MUST be analytical summaries of the 'niche' and 'intent' from PART 1.
    2.  **Be Analytical, Not Sensational:** The keywords should describe the *topic*, not the *clickbait*.
    3.  **IGNORE SPONSORS:** Your code already filters out video descriptions if they are ads. This is just a reminder to focus on the high-signal titles.

    **Examples of GOOD keywords (Analytical & Topical):**
    ✅ "political commentary"
    ✅ "celebrity scandal analysis"
    ✅ "tech industry critique"
    ✅ "internet culture drama"
    ✅ "corporate controversy explained"
    ✅ "video essay [topic]"
    ✅ "the problem with [company]"

    **Examples of BAD keywords (Too Sensational/Vague):**
    ❌ "society destroyed" (Too sensational, bad search results)
    ❌ "celebrity exposed" (Too generic, will return tabloids)
    ❌ "government conspiracy" (Too broad)
    ❌ "cultural critique" (Too academic, not a search query)
    ❌ "company destroyed" (Too emotional)

    ---
    EXAMPLE OUTPUT (This is the style you must follow):
    {{
    "profile": {{ ... }},
    "keywords": {{
        "Political Commentary & Analysis": [
            "political commentary",
            "political news analysis",
            "political scandal explained",
            "controversial political figures",
            "government failures explained"
        ],
        "Celebrity & Entertainment Critique": [
            "celebrity scandal analysis",
            "entertainment industry critique",
            "celebrity controversy explained",
            "hollywood industry analysis",
            "celebrity downfall analysis"
        ],
        "Internet Culture & Creator Commentary": [
            "youtube creator drama",
            "influencer controversy analysis",
            "internet personality critique",
            "social media culture critique",
            "creator community drama"
        ]
    }}
    }}
    --- (End of Example) ---

    OUTPUT:
    Return ONLY the valid JSON for the channel in the "RAW DATA" section.
    """

PROMPT_E_GEMINI_5 = """You are an expert YouTube channel analyst.
    Analyze the provided raw data (channel name, description, video titles, video descriptions) 
    and extract a complete channel profile and its content themes.

    RAW DATA TO ANALYZE:
    \"\"\"
    {truncated_content}
    \"\"\"

    TASK: Return a single, valid JSON object with two top-level keys: "profile" and "keywords".

    ---
    PART 1: "profile"
    ---
    (Your profile part is perfect, leave it as-is)
    ...

    ---
    PART 2: "keywords"
    ---
    Generate 2-4 most dominant content **THEMES** (categories).
    Under each theme, list 5-7 **analytical keywords and search queries** that describe this topic.

    **CRITICAL (THE GOAL):**
    Your goal is to *CATEGORIZE* the content, not to copy its emotional language.
    The keywords should be the *academic topic* or *niche* of the videos.
    We are looking for the *literal search queries* a person would use to find *other channels in this same niche*.

    **CRITICAL (THE RULES):**
    1.  **Reflect the Profile:** The themes MUST be analytical summaries of the 'niche' and 'intent' from PART 1.
    2.  **Be Analytical, Not Sensational:** The keywords should describe the *topic*, not the *clickbait*.
    3.  **IGNORE SPONSORS:** Your code already filters out video descriptions if they are ads. This is just a reminder to focus on the high-signal titles.

    **Examples of GOOD keywords (Analytical & Topical):**
    ✅ "political commentary"
    ✅ "celebrity scandal analysis"
    ✅ "tech industry critique"
    ✅ "internet culture drama"
    ✅ "corporate controversy explained"
    ✅ "video essay [topic]"
    ✅ "the problem with [company]"

    **Examples of BAD keywords (Too Sensational/Vague):**
    ❌ "society destroyed" (Too sensational, bad search results)
    ❌ "celebrity exposed" (Too generic, will return tabloids)
    ❌ "government conspiracy" (Too broad)
    ❌ "cultural critique" (Too academic, not a search query)
    ❌ "company destroyed" (Too emotional)

    ---
    EXAMPLE OUTPUT (This is the style you must follow):
    {{
    "profile": {{ ... }},
    "keywords": {{
        "Political Commentary & Analysis": [
            "political commentary",
            "political news analysis",
            "political scandal explained",
            "controversial political figures",
            "government failures explained"
        ],
        "Celebrity & Entertainment Critique": [
            "celebrity scandal analysis",
            "entertainment industry critique",
            "celebrity controversy explained",
            "hollywood industry analysis",
            "celebrity downfall analysis"
        ],
        "Internet Culture & Creator Commentary": [
            "youtube creator drama",
            "influencer controversy analysis",
            "internet personality critique",
            "social media culture critique",
            "creator community drama"
        ]
    }}
    }}
    --- (End of Example) ---

    OUTPUT:
    Return ONLY the valid JSON for the channel in the "RAW DATA" section.
    """

# --- 5. THE GEMINI FINGERPRINT FUNCTION ---
def get_channel_fingerprint_oneshot_gemini(
    channel_name: str,
    channel_description: str,
    video_df: pd.DataFrame,
    prompt_template: str, # <-- NEW: Accepts a prompt template
    retries: int = 3
) -> dict:
    """
    Performs a single, "one-shot" LLM call to extract BOTH the
    detailed channel profile and the focused SEO keywords using Gemini.
    
    (Version 3.1: TITLES-ONLY + GEMINI + PROMPT TEMPLATE)
    """
    
    # --- 1. Combine all HIGH-SIGNAL text for context ---
    safe_channel_desc = str(channel_description) if pd.notna(channel_description) else "N/A - No description provided"
    
    video_titles = video_df['title'].tolist()
    safe_titles = [str(t) for t in video_titles if pd.notna(t)]
    
    # --- Build the CLEAN text blob (TITLES-ONLY) ---
    combined_text = f"CHANNEL NAME: {channel_name}\n\n"
    combined_text += f"CHANNEL DESCRIPTION:\n{safe_channel_desc}\n\n"
    combined_text += "--- RECENT VIDEO TITLES (CRITICAL DATA) ---\n"
    combined_text += "\n".join(safe_titles)
    
    truncated_content = combined_text 
    
    print(f"📤 Sending {len(truncated_content)} chars to Gemini...")
    
    # --- 2. Inject data into the prompt template ---
    try:
        prompt = prompt_template.format(truncated_content=truncated_content)
    except KeyError as e:
        print(f"❌ ERROR: Your prompt template is missing a placeholder: {e}")
        print("   Make sure your prompt includes '{truncated_content}'")
        return {}

    
    for attempt in range(retries):
        try:
            # ========== GEMINI MODE ==========
            response = gemini_model.generate_content(prompt)
            
            # --- Clean the response ---
            # Gemini often wraps JSON in ```json ... ```
            result_text = response.text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            # ========== Parse JSON Output ==========
            try:
                parsed_json = json.loads(result_text)
                
                if "profile" in parsed_json and "keywords" in parsed_json:
                    print(f"✅ One-shot analysis successful for {channel_name}.")
                    return parsed_json
                else:
                    print(f"⚠️ LLM returned invalid JSON structure: {result_text[:100]}... (Attempt {attempt+1})")

            except json.JSONDecodeError:
                print(f"⚠️ LLM output was not valid JSON: {result_text[:100]}... (Attempt {attempt+1})")
                
        except Exception as e:
            print(f"❌ LLM One-Shot Error (Attempt {attempt+1}/{retries}): {str(e)[:100]}")
            time.sleep(5 * (attempt + 1))
            
    print(f"❌ All retries failed for {channel_name}.")
    return {}

# --- 6. MAIN TESTING FUNCTION ---
def main():
    """
    Main function to run the A/B test.
    """
    print("--- 🚀 Starting Manual Gemini Prompt Test ---")
    
    RUN_TAG = "moon" # We're testing the "moon" channel
    run_tag = RUN_TAG
    # This script saves outputs in its *own* directory
    OUTPUT_DIR = Path(__file__).resolve().parent

    # --- A. Load Seed Channel Data (from Phase 1) ---
    print(f"\n[STEP 1/3] Loading Phase 1 data for '{RUN_TAG}'...")
    
    # Find the data files in the root project (BASE_DIR)
    FINGERPRINT_DIR = BASE_DIR / "FINGERPRINTS_ONESHOT"
    SEED_VIDEOS_DIR = BASE_DIR / "PHASE_1_OUTPUTS"
    
    SEED_FINGERPRINT_FILE = find_latest_file_path(FINGERPRINT_DIR, f"fingerprints_oneshot_{run_tag}", ".json")
    SEED_VIDEOS_FILE = find_latest_file_path(SEED_VIDEOS_DIR, f"sample_videos_{run_tag}", ".csv")

    if not SEED_FINGERPRINT_FILE or not SEED_VIDEOS_FILE:
        print(f"❌ ERROR: Missing Phase 1 files for tag '{run_tag}' in project root.")
        return
        
    print(f"  Using Seed Fingerprint: {SEED_FINGERPRINT_FILE.name}")
    print(f"  Using Seed Video Data: {SEED_VIDEOS_FILE.name}")
    
    try:
        with open(SEED_FINGERPRINT_FILE, 'r') as f:
            seed_data = json.load(f)
        first_channel_key = next(iter(seed_data["channels"]))
        channel_desc = (
            seed_data["channels"][first_channel_key]
            .get("fingerprint", {})
            .get("profile", {})
            .get("description", "")
        )
        channel_name = seed_data["channels"][first_channel_key].get("channel_name", run_tag)
        
        df_seed_videos = pd.read_csv(SEED_VIDEOS_FILE)
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load or parse seed files: {e}")
        return

    print(f"✅ Loaded data for channel: {channel_name}")

    # --- B. Define the A/B Test ---
    prompts_to_test = {
        "GEMINI_1_MOON": PROMPT_A_GEMINI_1 ,
        "GEMINI_2_MOON": PROMPT_B_GEMINI_2,
        "PERPLEXITY_1_MOON": PROMPT_C_PERPLEX_3,
        "GPT_MOON": PROMPT_D_GPT_4,
        "PREVIOUS_GEMINI_SOTA_MOON": PROMPT_E_GEMINI_5
    }

    # --- C. Run the Test Loop ---
    print(f"\n[STEP 2/3] Running {len(prompts_to_test)} prompt tests...")
    
    for test_name, prompt_template in prompts_to_test.items():
        print(f"\n--- Testing Prompt: {test_name} ---")
        
        fingerprint = get_channel_fingerprint_oneshot_gemini(
            channel_name=channel_name,
            channel_description=channel_desc,
            video_df=df_seed_videos,
            prompt_template=prompt_template
        )
        
        if fingerprint:
            # Save the output
            output_path = OUTPUT_DIR / f"output_{test_name}.json"
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(fingerprint, f, indent=2, ensure_ascii=False)
                print(f"✅ SUCCESS: Saved results to {output_path.name}")
                print(json.dumps(fingerprint.get("keywords", {}), indent=2))
            except Exception as e:
                print(f"❌ ERROR saving JSON: {e}")
        else:
            print(f"❌ FAILED: No output generated for {test_name}.")
            
    print("\n[STEP 3/3] --- Test Complete ---")



# # --- 6. MAIN TESTING FUNCTION for manual test_files ---
# def main():
#     """
#     Main function to run the A/B test.
#     """
#     print("--- 🚀 Starting Manual Gemini Prompt Test ---")
    
#     # This script saves outputs in its *own* directory
#     OUTPUT_DIR = Path(__file__).resolve().parent

#     # --- A. Load Data from SEEN_VIDEOS_OUTPUTS ---
#     print(f"\n[STEP 1/3] Loading data from SEEN_VIDEOS_OUTPUTS...")
    
#     # ✅ NEW: Point directly to your specific CSV
#     SEED_VIDEOS_FILE = BASE_DIR / "SEEN_VIDEOS_OUTPUTS" / "sample_videos.csv"
    
#     if not SEED_VIDEOS_FILE.exists():
#         print(f"❌ ERROR: File not found: {SEED_VIDEOS_FILE}")
#         return
        
#     print(f"  Using Video Data: {SEED_VIDEOS_FILE}")
    
#     try:
#         df_seed_videos = pd.read_csv(SEED_VIDEOS_FILE)
        
#         # ✅ Get channel info from the CSV itself
#         # Assuming the CSV has columns: 'Channel_Name', 'channel_description', 'title', etc.
#         if 'Channel_Name' in df_seed_videos.columns:
#             channel_name = df_seed_videos['Channel_Name'].iloc[0]
#         else:
#             channel_name = "Unknown Channel"  # Fallback
            
#         if 'channel_description' in df_seed_videos.columns:
#             channel_desc = df_seed_videos['channel_description'].iloc[0]
#         else:
#             channel_desc = "No description available"
            
#         print(f"✅ Loaded {len(df_seed_videos)} videos for channel: {channel_name}")
#         print(f"   Channel Description: {channel_desc[:100]}...")
        
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load CSV: {e}")
#         return

#     # --- B. Define the A/B Test ---
#     prompts_to_test = {
#         "GEMINI_1": PROMPT_A_GEMINI_1 ,
#         "GEMINI_2": PROMPT_B_GEMINI_2,
#         "PERPLEXITY_1": PROMPT_C_PERPLEX_3,
#         "GPT": PROMPT_D_GPT_4,
#         "PREVIOUS_GEMINI_SOTA": PROMPT_E_GEMINI_5
#     }

#     # --- C. Run the Test Loop ---
#     print(f"\n[STEP 2/3] Running {len(prompts_to_test)} prompt tests...")
    
#     for test_name, prompt_template in prompts_to_test.items():
#         print(f"\n--- Testing Prompt: {test_name} ---")
        
#         fingerprint = get_channel_fingerprint_oneshot_gemini(
#             channel_name=channel_name,
#             channel_description=channel_desc,
#             video_df=df_seed_videos,
#             prompt_template=prompt_template
#         )
        
#         if fingerprint:
#             # Save the output
#             output_path = OUTPUT_DIR / f"output_{test_name}.json"
#             try:
#                 with open(output_path, 'w', encoding='utf-8') as f:
#                     json.dump(fingerprint, f, indent=2, ensure_ascii=False)
#                 print(f"✅ SUCCESS: Saved results to {output_path.name}")
#                 print(json.dumps(fingerprint.get("keywords", {}), indent=2))
#             except Exception as e:
#                 print(f"❌ ERROR saving JSON: {e}")
#         else:
#             print(f"❌ FAILED: No output generated for {test_name}.")
            
#     print("\n[STEP 3/3] --- Test Complete ---")



# --- 7. RUN THE SCRIPT ---
if __name__ == "__main__":
    main()