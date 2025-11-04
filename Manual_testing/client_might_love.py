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
import re

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
    generation_config = genai.GenerationConfig(
        max_output_tokens=2000,
        temperature=0.1
    )
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    gemini_model = genai.GenerativeModel(
        'gemini-2.0-flash-exp', # Use the latest Flash model
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print("✅ Gemini model configured: gemini-2.0-flash-exp")
except Exception as e:
    print(f"❌ Gemini configuration failed: {e}")
    sys.exit(1)


# ==================================================
# 3. CONFIGURATION
# ==================================================
# --- Set your run tag to find the input file ---
RUN_TAG = "vox"

# --- PASTE THE CHANNELS YOU WANT TO TEST ---
# (To stay under your 200/day limit)
CANDIDATES_TO_PROCESS = [
    "Democracy At Work",
    "Freethink",
    "Johnny Harris",
    "Bernie Sanders",
    "Moon",
    "J.J. McCullough",
    "Mr. Beat",
    "Alex O'Connor",
    "Visual Venture",
    "Queen",
    "WWEMusic",
    "Spreadsheet Nation"
]

# --- VOX SOTA PROFILE & KEYWORDS (From our last chat) ---
SEED_PROFILE = """
{
  "profile": {
            "channel_name": "Vox",
            "description": "Making sense of it all.",
            "likely_niche": "Explanatory journalism and in-depth analysis of current events, societal issues, and complex topics across various domains.",
            "target_audience": "Individuals seeking informed perspectives and comprehensive explanations of complex issues, including current events, politics, economics, science, and culture. The audience likely values critical thinking and nuanced analysis.",
            "video_style": "Documentary-style videos featuring a mix of visuals, data, and expert interviews, often incorporating animation and graphics to enhance understanding. The channel is known for its in-depth reporting and analytical approach.",
            "intent": "To provide viewers with comprehensive and accessible explanations of complex issues, fostering a deeper understanding of the world and encouraging informed discussion and critical thinking."
    }
}
"""

SEED_CATEGORIES = """
- "Societal Issues & Current Events Analysis"
- "Political & Economic Commentary"
- "Science & Technology Explained"
- "Cultural Trends & Analysis"
"""
# ==================================================


# --- 4. HELPER FUNCTIONS ---
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

def get_channel_tier_gemini(
    candidate_name: str, 
    candidate_titles: list, 
    seed_profile_str: str, 
    seed_categories_str: str
) -> dict:
    """
    Calls Gemini to perform the "Client's Gut Check" and assign a tier.
    """
    
    # Create the text blob of candidate titles
    titles_blob = "\n- ".join(candidate_titles)
    
    # --- This is the new SOTA prompt ---
    prompt = f"""You are an expert YouTube analyst and content strategist. Your goal is to help a client find *exact* competitors for their 'Documentary-style Explainer' channel, 'Vox'.

**SEED CHANNEL (VOX) CONTEXT:**
Here is the Seed Channel's profile:
{seed_profile_str}

Here are the Seed Channel's main Content Pillars:
{seed_categories_str}

**CANDIDATE CHANNEL ANALYSIS:**
I will now provide the 20 most recent video titles from a candidate channel.
Candidate Name: "{candidate_name}"
Candidate Titles:
- {titles_blob}

**YOUR TASK (Must follow these 3 steps):**

**Step 1: Format Analysis**
Analyze the *candidate's* video titles. What is their likely format?
- Are they 'Documentary / Video Essay' (long-form, analytical, 'why/how' titles)?
- Or are they 'News / Clips / Other' (short, daily, reactive, or non-analytical titles)?

**Step 2: Topic Analysis**
Compare the *candidate's* titles against the *Seed's Content Pillars* listed above. How many of the 4 seed pillars does the candidate *also* cover in a meaningful way (0, 1, or 2+)?

**Step 3: Tier Assignment**
Based on your analysis from Steps 1 & 2, assign a Tier using these rules:
- **Tier 1:** Format matches (Documentary/Essay) AND 2 or more topics match.
- **Tier 2:** Format matches (Documentary/Essay) AND 1 topic matches.
- **Tier 3:** Format does NOT match (e.g., News/Clips) OR 0 topics match.

**OUTPUT FORMAT:**
Return ONLY a single, valid JSON object with two keys: "tier" (an integer 1, 2, or 3) and "reason" (your step-by-step analysis explaining *why* you assigned that tier, referencing both format and topic).

**Example (Tier 1):**
{{
  "tier": 1,
  "reason": "Format matches: The titles ('How a new law is changing...') are clearly analytical video essays. Topic matches: The titles cover 3 of the seed's pillars: 'US Politics', 'Economics', and 'Social Issues'."
}}

**Example (Tier 3 - Format Mismatch):**
{{
  "tier": 3,
  "reason": "Format does not match: The titles ('Biden Speaks...', 'Market Watch...') suggest daily news clips, not documentaries. Even though the topics match (Politics, Economics), the format is a mismatch for the client."
}}

**Example (Tier 3 - Topic Mismatch):**
{{
  "tier": 3,
  "reason": "Format matches: The titles look like video essays. However, Topic match: 0. The topics are about 'Gaming' and 'Anime', which do not align with the seed's pillars."
}}

Provide ONLY the JSON output.
"""

    # --- Call Gemini ---
    try:
        response = gemini_model.generate_content(prompt)
        
        # Clean the response
        result_text = response.text.strip()
        
        # Use regex to find the JSON block, in case Gemini adds extra text
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not match:
            print(f"  ⚠️  No JSON found in Gemini response: {result_text[:100]}")
            return {"tier": -1, "reason": "ERROR: No JSON in response."}
            
        json_str = match.group(0)
        parsed_json = json.loads(json_str)
        
        if "tier" in parsed_json and "reason" in parsed_json:
            return parsed_json
        else:
            print(f"  ⚠️  JSON missing 'tier' or 'reason' keys.")
            return {"tier": -1, "reason": "ERROR: Malformed JSON."}

    except json.JSONDecodeError:
        print(f"  ⚠️  Gemini output was not valid JSON: {json_str[:100]}")
        return {"tier": -1, "reason": "ERROR: Invalid JSON."}
    except Exception as e:
        print(f"  ❌ LLM Error: {str(e)[:150]}")
        return {"tier": -1, "reason": f"ERROR: {str(e)[:100]}"}


# --- 5. MAIN SCRIPT ---
def main():
    """
    Main function to run the "Client's Gut Check" analysis.
    """
    print(f"--- 🚀 Starting Client's Tier Analysis for '{RUN_TAG}' ---")
    print(f"   Will process {len(CANDIDATES_TO_PROCESS)} selected channels.")
    
    # This script saves outputs in its *own* directory
    OUTPUT_DIR = Path(__file__).resolve().parent
    OUTPUT_FILE = OUTPUT_DIR / f"client_tier_analysis_{RUN_TAG}.json"

    # --- A. Load the Phase 2.5 Triage Report ---
    print(f"\n[STEP 1/3] Loading Phase 2.5 Triage Report...")
    TRIAGE_REPORT_DIR = BASE_DIR / "PHASE_2_5_TRIAGE_REPORTS"
    TRIAGE_REPORT_FILE = find_latest_file_path(
        TRIAGE_REPORT_DIR, f"phase2_5_triage_report_{RUN_TAG}", ".csv"
    )

    if not TRIAGE_REPORT_FILE:
        print(f"❌ ERROR: No Phase 2.5 triage report file found for tag '{RUN_TAG}'.")
        print(f"   Looked in: {TRIAGE_REPORT_DIR}")
        return

    print(f"  Using Triage File: {TRIAGE_REPORT_FILE.name}")
    try:
        df_full = pd.read_csv(TRIAGE_REPORT_FILE)
    except Exception as e:
        print(f"❌ ERROR: Failed to load triage report: {e}")
        return

    # --- B. Filter for the channels you want to test ---
    print(f"\n[STEP 2/3] Filtering for {len(CANDIDATES_TO_PROCESS)} channels...")
    df_filtered = df_full[df_full['Discovered_Channel_Name'].isin(CANDIDATES_TO_PROCESS)].copy()
    
    if len(df_filtered) == 0:
        print(f"❌ ERROR: Found 0 of the channels you listed in the CSV.")
        print(f"   Check the names in CANDIDATES_TO_PROCESS.")
        return
    
    print(f"   Found {len(df_filtered)} channels to process.")

    # --- C. Run the LLM Tier Analysis Loop ---
    print(f"\n[STEP 3/3] Starting LLM Tier Analysis Loop (Limit: 200/day)...")
    
    results = {}
    
    for index, row in df_filtered.iterrows():
        channel_name = row['Discovered_Channel_Name']
        print(f"\n--- Processing: {channel_name} ({index+1}/{len(df_filtered)}) ---")
        
        try:
            # Load the cached video data
            videos_json = row['Discovered_Videos_JSON']
            videos_list = json.loads(videos_json)
            if not videos_list:
                print("  ⚠️  No videos in cache. Skipping.")
                results[channel_name] = {"tier": -1, "reason": "ERROR: No videos in cache."}
                continue
            
            # Extract just the titles
            video_titles = [vid.get('title', '') for vid in videos_list if vid.get('title')]
            
            # Call our new Gemini function
            tier_data = get_channel_tier_gemini(
                candidate_name=channel_name,
                candidate_titles=video_titles,
                seed_profile_str=SEED_PROFILE,
                seed_categories_str=SEED_CATEGORIES
            )
            
            print(f"  ✅ Tier: {tier_data.get('tier')}")
            print(f"  > Reason: {tier_data.get('reason')}")
            results[channel_name] = tier_data
            
            # --- Rate limit for free Gemini API ---
            time.sleep(20) 

        except Exception as e:
            print(f"  ❌❌ UNEXPECTED ERROR on {channel_name}: {e}")
            results[channel_name] = {"tier": -1, "reason": f"ERROR: {str(e)[:100]}"}
            
    # --- D. Save Final Output ---
    print("\n--- ANALYSIS COMPLETE ---")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅✅✅ Success! Tier analysis saved to:")
        print(f"   {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ ERROR saving final JSON: {e}")

# --- 6. RUN THE SCRIPT ---
if __name__ == "__main__":
    main()