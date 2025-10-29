import pandas as pd
import json
import re
import time
from pathlib import Path
import os
from dotenv import load_dotenv
from typing import List, Dict # Added Dict type hint

# --- Choose your LLM Provider ---
MODEL_PROVIDER = "gpt"  # Change to "gemini" to use Gemini
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# --- Load API Keys ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configure LLM Clients ---
gemini_model = None
gpt_client = None

# (Keep your LLM client configuration logic here - unchanged)
if MODEL_PROVIDER == "gemini":
    if GOOGLE_API_KEY:
        import google.generativeai as genai
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            # Use a model good at reasoning
            # NOTE: Gemini might struggle more with strict multi-line formats than GPT-4 family
            gemini_model = genai.GenerativeModel('gemini-pro')
            print("✅ Gemini model configured.")
        except Exception as e:
            print(f"❌ Gemini configuration failed: {e}")
            gemini_model = None
    else:
        print("⚠️ GOOGLE_API_KEY not found. Cannot use Gemini.")

elif MODEL_PROVIDER == "gpt":
    if OPENAI_API_KEY:
        from openai import OpenAI
        try:
            gpt_client = OpenAI(api_key=OPENAI_API_KEY)
            print("✅ OpenAI client configured.")
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            gpt_client = None
    else:
        print("⚠️ OPENAI_API_KEY not found. Cannot use GPT.")
else:
    print(f"❌ Invalid MODEL_PROVIDER: {MODEL_PROVIDER}. Choose 'gemini' or 'gpt'.")


# --- Utility: Text Cleaning ---
def preprocess_text_for_llm(text: str, max_len=500) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len]

# --- YOUR EXCELLENT NEW PROMPT Function ---
def create_competitor_prompt_v2_content(seed_name, seed_niche, seed_about, seed_keywords: List[str],
                                        discovered_name, discovered_niche, discovered_about, discovered_keywords: List[str]):
    """Content-based competitor detection (strict format + topic match)"""
    
    # Truncate and clean descriptions
    seed_about_clean = preprocess_text_for_llm(seed_about)
    discovered_about_clean = preprocess_text_for_llm(discovered_about)

    # Format keywords for prompt
    seed_kws_str = ", ".join(seed_keywords[:10]) if seed_keywords else 'N/A'
    discovered_kws_str = ", ".join(discovered_keywords[:10]) if discovered_keywords else 'N/A'

    prompt = f"""You are an expert YouTube analyst determining if two channels are DIRECT CONTENT COMPETITORS.

SEED CHANNEL: "{seed_name}"
Primary Niche: "{seed_niche or 'N/A'}"
Channel Description: "{seed_about_clean or 'N/A'}"
Keywords/Topics: {seed_kws_str}

CANDIDATE CHANNEL: "{discovered_name}"
Primary Niche: "{discovered_niche or 'N/A'}"
Channel Description: "{discovered_about_clean or 'N/A'}"
Keywords/Topics: {discovered_kws_str}

DEFINITION: Content competitors make videos on the SAME TOPICS in the SAME FORMAT.

They could make a video with the SAME TITLE and it would fit BOTH channels.

STRICT EVALUATION CRITERIA (ALL must match):

1. **Keyword/Topic Overlap** (CRITICAL):
   - Keywords must show 70%+ topic overlap
   - Example: "business documentaries, corporate scandals, company failures" vs "business history, corporate analysis, company downfall" = HIGH overlap ✅
   - Example: "productivity tips, notion systems, study hacks" vs "career interviews, success stories, entrepreneur advice" = LOW overlap ❌

2. **Format Match** (CRITICAL - Inferred from keywords):
   - Look for format indicators in keywords:
     - Tutorial keywords: "tips", "how to", "tutorial", "guide", "hacks", "systems"
     - Documentary keywords: "documentary", "history", "story", "analysis", "deep-dive"
     - Podcast keywords: "podcast", "interview", "conversation", "chat", "discussion"
     - Short-form keywords: "shorts", "clips", "quick", "60 seconds"
   - Tutorial vs Tutorial = Match ✅
   - Documentary vs Documentary = Match ✅
   - Tutorial vs Podcast = NO Match ❌
   - Long-form vs Short-form = NO Match ❌

3. **Niche Precision Match**:
   - Niches must be VERY similar (not just related)
   - Example: "Business documentaries" vs "Corporate history" = Match ✅
   - Example: "Productivity tutorials" vs "Study tips" = Match ✅
   - Example: "Business docs" vs "Business tutorials" = NO Match ❌

4. **Language/Region Match**:
   - Keywords must suggest same language/region
   - English terms only vs Indian/Hindi terms = NO Match ❌

STRICT RULES FOR NO:
- Different format indicators in keywords (tutorial vs podcast) = NOT competitors
- Different languages/regions = NOT competitors
- Low keyword overlap (<70%) = NOT competitors

EXAMPLES:

Example 1 - YES (Perfect match):
Seed: Niche:"Business docs"/KWs:"corporate scandals, company failures, business history, documentary storytelling"
Candidate: Niche:"Corporate documentaries"/KWs:"business downfall, company analysis, corporate history, narrative docs"
COMPETITOR: Yes
CONFIDENCE: High
REASON: Keywords show 85% topic overlap and both have documentary format indicators.

Example 2 - YES (Good match):
Seed: Niche:"Productivity tutorials"/KWs:"notion tips, study hacks, time management, productivity systems"
Candidate: Niche:"Study tips"/KWs:"study systems, productivity hacks, note-taking tips, exam preparation"
COMPETITOR: Yes
CONFIDENCE: High
REASON: Keywords show 75% topic overlap and both have tutorial/tips format indicators.

Example 3 - NO (Format mismatch):
Seed: Niche:"Productivity tips"/KWs:"productivity systems, notion tutorial, time management tips"
Candidate: Niche:"Success podcast"/KWs:"entrepreneur interviews, podcast discussions, career conversations"
COMPETITOR: No
CONFIDENCE: High
REASON: Keywords show different formats; tutorial/tips indicators vs podcast/interview indicators despite related topics.

Example 4 - NO (Topic mismatch):
Seed: Niche:"Business docs"/KWs:"corporate analysis, business strategy, company history"
Candidate: Niche:"Business skills"/KWs:"excel tips, presentation skills, business communication tutorial"
COMPETITOR: No
CONFIDENCE: High
REASON: Keywords show different content types; documentary/analysis vs practical skills/tutorial despite same niche.

Example 5 - NO (Language/region mismatch):
Seed: Niche:"Productivity (English)"/KWs:"productivity systems, notion tips, workflow optimization"
Candidate: Niche:"Career guidance (Hindi)"/KWs:"UPSC preparation, sarkari exam, SSC strategy"
COMPETITOR: No
CONFIDENCE: High
REASON: Keywords indicate different languages and regional focus; Western productivity vs Indian exam system.

Example 6 - NO (Low topic overlap):
Seed: Niche:"Tech tutorials"/KWs:"coding tutorials, programming tips, software development"
Candidate: Niche:"Tech news"/KWs:"gadget reviews, tech updates, smartphone analysis"
COMPETITOR: No
CONFIDENCE: High
REASON: Keywords show low topic overlap; programming education vs consumer tech reviews.

**CRITICAL**: Both topic overlap (70%+) AND format match are REQUIRED.

Format indicators in keywords are KEY signals:
- "tips/tutorial/how-to" = Tutorial format
- "documentary/history/story" = Documentary format
- "podcast/interview/conversation" = Podcast format

Ask: "Do keywords show BOTH same topics AND same format?"

**Output Format** (exactly 4 lines):
COMPETITOR: [Yes/No]
CONFIDENCE: [High/Medium/Low]
KEYWORD_OVERLAP: [High (>70%)/Medium (50-70%)/Low (<50%)]
REASON: [One sentence explaining keyword overlap and format match/mismatch]

Example Output:
COMPETITOR: Yes
CONFIDENCE: High
KEYWORD_OVERLAP: High (80%)
REASON: Keywords show high topic overlap in business analysis with documentary format indicators in both.
"""
    return prompt


# --- ***CORRECTED*** Function to Call LLM and Parse Output ---
def classify_competitor(seed_name, seed_niche, seed_about, seed_keywords: List[str],
                         discovered_name, discovered_niche, discovered_about, discovered_keywords: List[str]) -> Dict:
    """
    Calls the configured LLM with full context and parses the NEW 4-line output.
    Returns a dictionary with 'status', 'confidence', 'overlap', 'reason'.
    """
    # Default result dictionary
    result = {
        "status": "Error",
        "confidence": "N/A",
        "overlap": "N/A",
        "reason": "LLM client not configured or call failed"
    }

    if (MODEL_PROVIDER == "gemini" and not gemini_model) or \
       (MODEL_PROVIDER == "gpt" and not gpt_client):
        return result # Return default error dict

    prompt = create_competitor_prompt_v2_content(seed_name, seed_niche, seed_about, seed_keywords,
                                      discovered_name, discovered_niche, discovered_about, discovered_keywords)

    for attempt in range(MAX_RETRIES):
        try:
            raw_response = ""
            if MODEL_PROVIDER == "gemini":
                # Increase tokens slightly for the more complex 4-line output
                generation_config = genai.GenerationConfig(max_output_tokens=100, temperature=0.1)
                response = gemini_model.generate_content(prompt, generation_config=generation_config)
                raw_response = response.text.strip()
            elif MODEL_PROVIDER == "gpt":
                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini", # Sufficient for this structured output
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100, # Increased tokens
                )
                raw_response = response.choices[0].message.content.strip()

            # --- NEW Robust Parsing for 4 Lines ---
            parsed_successfully = False
            # Use re.IGNORECASE and re.MULTILINE
            comp_match = re.search(r"^COMPETITOR:\s*(Yes|No)\s*$", raw_response, re.IGNORECASE | re.MULTILINE)
            conf_match = re.search(r"^CONFIDENCE:\s*(High|Medium|Low)\s*$", raw_response, re.IGNORECASE | re.MULTILINE)
            over_match = re.search(r"^KEYWORD_OVERLAP:\s*(High(?: \(\>\d+%\))?|Medium(?: \(\d+-\d+%\))?|Low(?: \(<\d+%\))?)\s*$", raw_response, re.IGNORECASE | re.MULTILINE)
            # Use DOTALL to match reason across potential newlines, make non-greedy
            reason_match = re.search(r"^REASON:\s*(.*?)\s*$", raw_response, re.IGNORECASE | re.MULTILINE | re.DOTALL)

            if comp_match and conf_match and over_match and reason_match:
                result["status"] = comp_match.group(1).capitalize()
                result["confidence"] = conf_match.group(1).capitalize()
                # Extract just the High/Medium/Low part of overlap
                overlap_text = over_match.group(1).split()[0].capitalize()
                result["overlap"] = overlap_text
                result["reason"] = reason_match.group(1).strip()
                parsed_successfully = True

                print(f"      LLM Result: {result['status']} | Conf: {result['confidence']} | Overlap: {result['overlap']} | Reason: {result['reason'][:60]}...")
                return result # Success!
            else:
                # Log which parts failed for debugging
                missing = []
                if not comp_match: missing.append("COMPETITOR")
                if not conf_match: missing.append("CONFIDENCE")
                if not over_match: missing.append("KEYWORD_OVERLAP")
                if not reason_match: missing.append("REASON")
                print(f"      ⚠️ Failed to parse LLM Output (Missing: {', '.join(missing)}). Raw:\n{raw_response}")
                result["reason"] = f"Parsing Failed (Missing: {', '.join(missing)}): {raw_response[:100]}"
                # Continue to retry if parsing failed

        except Exception as e:
            print(f"      ❌ LLM API Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            result["reason"] = f"API Error: {str(e)[:100]}" # Store error in reason
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                 wait_time = RETRY_DELAY * (attempt + 1) * 2
                 print(f"      Rate limit likely hit. Waiting {wait_time}s...")
                 time.sleep(wait_time)
            elif attempt < MAX_RETRIES - 1:
                print(f"      Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                 result["reason"] = f"API Error after {MAX_RETRIES} attempts: {e}"
                 return result # Return error dict after last retry

        # If parsing failed on last attempt
        if attempt == MAX_RETRIES - 1 and not parsed_successfully:
             return result # Return dict with parsing failed reason

    return result # Should not be reached if MAX_RETRIES > 0


# --- ***CORRECTED*** Main Script Logic ---
def main():
    start_time = time.time()
    print(f"--- Starting Competitor Classification ({MODEL_PROVIDER.upper()}) ---")

# /home/rareboy/Internship/Kajkarma/
    input_csv_path = Path("./FINAL_SEED_SUBSCRIBE_MERGED.csv")
    output_csv_path = Path(f"./check_competitors_{MODEL_PROVIDER}_v4_strict.csv") # v4_strict
    seen_channels_csv_path = Path("./seen_channels.csv")
    fingerprints_json_path = Path(f"./channel_fingerprints_{MODEL_PROVIDER}.json")

    if not input_csv_path.exists(): print(f"❌ ERROR: Input file not found: {input_csv_path}"); return

    # --- Load descriptions ---
    channel_descriptions = {}
    if seen_channels_csv_path.exists():
         try:
              df_seen = pd.read_csv(seen_channels_csv_path)
              if 'Channel_ID' in df_seen.columns and 'description' in df_seen.columns:
                   df_seen['description'] = df_seen['description'].fillna('')
                   channel_descriptions = df_seen.set_index('Channel_ID')['description'].to_dict()
                   print(f"✅ Loaded descriptions for {len(channel_descriptions)} channels.")
              else: print(f"⚠️ {seen_channels_csv_path.name} missing required columns.")
         except Exception as e: print(f"⚠️ Error loading descriptions: {e}")
    else: print(f"⚠️ {seen_channels_csv_path.name} not found.")

    # --- Load keywords ---
    channel_keywords = {}
    if fingerprints_json_path.exists():
        try:
            with open(fingerprints_json_path, 'r', encoding='utf-8') as f:
                fingerprint_data = json.load(f)
                for name, data in fingerprint_data.get("channels", {}).items():
                    if "keywords" in data: channel_keywords[name] = data["keywords"]
                print(f"✅ Loaded keywords for {len(channel_keywords)} channels.")
        except Exception as e: print(f"⚠️ Error loading keywords: {e}")
    else: print(f"⚠️ {fingerprints_json_path.name} not found.")

    if (MODEL_PROVIDER == "gemini" and not gemini_model) or \
       (MODEL_PROVIDER == "gpt" and not gpt_client):
         print(f"❌ ERROR: LLM client failed to configure."); return

    try:
        df = pd.read_csv(input_csv_path)
        print(f"✅ Loaded {len(df)} rows from {input_csv_path.name}")
    except Exception as e: print(f"❌ ERROR reading CSV: {e}"); return

    # --- ADDED: Columns for new output format ---
    if 'Is_Competitor' not in df.columns: df['Is_Competitor'] = "Pending"
    if 'Confidence' not in df.columns: df['Confidence'] = ""
    if 'Keyword_Overlap' not in df.columns: df['Keyword_Overlap'] = ""
    if 'Reason' not in df.columns: df['Reason'] = ""
    # --- END ADDED ---

    if 'Seed_Channel_ID' not in df.columns or 'Discovered_Channel_ID' not in df.columns:
         print(f"❌ ERROR: Input CSV needs 'Seed_Channel_ID' and 'Discovered_Channel_ID'."); return

    rows_to_process = df[(df['Is_Competitor'] == "Pending") | (df['Is_Competitor'] == "Error")].index
    print(f"Found {len(rows_to_process)} rows to classify.")

    processed_count = 0
    for index in rows_to_process:
        row = df.loc[index]
        print(f"\nProcessing row {index + 1}/{len(df)}: {row['Seed_Channel_Name']} vs {row['Discovered_Channel_Name']}")

        seed_channel_id = row['Seed_Channel_ID']
        discovered_channel_id = row['Discovered_Channel_ID']
        seed_channel_name = row['Seed_Channel_Name']
        discovered_channel_name = row['Discovered_Channel_Name']

        seed_about = channel_descriptions.get(seed_channel_id, "")
        discovered_about = channel_descriptions.get(discovered_channel_id, "")
        seed_kws = channel_keywords.get(seed_channel_name, [])
        discovered_kws = channel_keywords.get(discovered_channel_name, [])

        # Add warnings if data is missing, but still proceed
        if not seed_about: print("     ⚠️ Seed description missing.")
        if not discovered_about: print("     ⚠️ Discovered description missing.")
        if not seed_kws: print("     ⚠️ Seed keywords missing.")
        if not discovered_kws: print("     ⚠️ Discovered keywords missing.")

        # --- MODIFIED: Call updated function and handle dictionary result ---
        llm_result = classify_competitor(
            seed_channel_name,
            row.get('Seed_Niche', ''),
            seed_about,
            seed_kws,
            discovered_channel_name,
            row.get('Discovered_Niche', ''),
            discovered_about,
            discovered_kws
        )

        # Update DataFrame with the new fields from the dictionary
        df.loc[index, 'Is_Competitor'] = llm_result["status"]
        df.loc[index, 'Confidence'] = llm_result["confidence"]
        df.loc[index, 'Keyword_Overlap'] = llm_result["overlap"]
        df.loc[index, 'Reason'] = llm_result["reason"]
        # --- END MODIFIED ---

        processed_count += 1

        if processed_count % 10 == 0:
             try:
                  df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                  print(f"--- Saved progress ({processed_count} processed) ---")
             except Exception as e: print(f"  ⚠️ Error saving intermediate progress: {e}")

        time.sleep(1.5) # Keep delay

    # Final save
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ Successfully processed {processed_count} rows.")
        print(f"💾 Final results saved to: {output_csv_path.name}")
    except Exception as e: print(f"❌ Error saving final CSV: {e}")

    end_time = time.time()
    print(f"⏱️ Total time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()


# def create_competitor_prompt_v1_audience(seed_name, seed_niche, seed_about, seed_keywords: List[str],
#                                          discovered_name, discovered_niche, discovered_about, discovered_keywords: List[str]):
#     """Audience-based competitor detection (format differences OK)"""
    
#     # Truncate and clean descriptions
#     seed_about_clean = preprocess_text_for_llm(seed_about)
#     discovered_about_clean = preprocess_text_for_llm(discovered_about)

#     # Format keywords for prompt
#     seed_kws_str = ", ".join(seed_keywords[:10]) if seed_keywords else 'N/A'
#     discovered_kws_str = ", ".join(discovered_keywords[:10]) if discovered_keywords else 'N/A'

#     prompt = f"""You are an expert YouTube analyst determining if two channels compete for the SAME AUDIENCE.

# SEED CHANNEL: "{seed_name}"
# Primary Niche: "{seed_niche or 'N/A'}"
# Channel Description: "{seed_about_clean or 'N/A'}"
# Keywords/Topics: {seed_kws_str}

# CANDIDATE CHANNEL: "{discovered_name}"
# Primary Niche: "{discovered_niche or 'N/A'}"
# Channel Description: "{discovered_about_clean or 'N/A'}"
# Keywords/Topics: {discovered_kws_str}

# DEFINITION: Audience competitors target the SAME VIEWER DEMOGRAPHIC, even if content format differs.

# The SAME VIEWER would subscribe to BOTH channels because they serve related interests.

# EVALUATION CRITERIA:

# 1. **Keyword/Topic Overlap** (MOST IMPORTANT):
#    - Do keywords show 60%+ topic overlap?
#    - Example: "productivity tips, time management, study hacks" vs "productivity systems, workflow optimization, student tips" = HIGH overlap ✅
   
# 2. **Target Audience Match**:
#    - Same profession/life stage? (students, professionals, entrepreneurs)
#    - Same interest domain? (business, self-improvement, tech)
   
# 3. **Niche Relatedness**:
#    - Are niches in the same BROADER domain?
#    - Example: "Productivity" and "Self-improvement" = Related ✅
#    - Example: "Business" and "Finance" = Related ✅
#    - Example: "Business" and "True Crime" = Not Related ❌

# 4. **Language/Region Match**:
#    - Keywords suggest same language? (English terms vs Hindi terms)
#    - Descriptions suggest same region? (Western vs India-focused)

# FORMAT DIFFERENCES ARE OK if audience matches:
# - Tutorial vs Podcast = OK (same audience, different format)
# - Documentary vs Education = OK (same audience, different style)
# - Keywords like "tutorial" vs "podcast" = Don't disqualify!

# STRICT RULES FOR NO:
# - Different language/region (English vs Hindi/India) = NOT competitors
# - Different core domain (Business vs Gaming) = NOT competitors
# - Different life stage (Students vs Retirees) = NOT competitors

# EXAMPLES:

# Example 1 - YES (Same audience, different format):
# Seed: Niche:"Productivity tutorials"/KWs:"notion tips, time management, study hacks"
# Candidate: Niche:"Career podcasts"/KWs:"career growth, professional development, success habits"
# COMPETITOR: Yes
# REASON: Both target ambitious young professionals seeking growth despite different formats shown in keywords.

# Example 2 - YES (High keyword overlap):
# Seed: Niche:"Business documentaries"/KWs:"corporate scandals, company failures, business history"
# Candidate: Niche:"Entrepreneurship stories"/KWs:"startup failures, business mistakes, entrepreneur journeys"
# COMPETITOR: Yes
# REASON: Both target business enthusiasts with high keyword overlap in business/failure themes.

# Example 3 - NO (Language/region mismatch):
# Seed: Niche:"Productivity tips"/KWs:"productivity systems, workflow optimization, notion templates"
# Candidate: Niche:"Exam preparation"/KWs:"UPSC strategy, SSC preparation, sarkari exam tips"
# COMPETITOR: No
# REASON: Keywords show different regions and audiences; Western productivity vs Indian exam prep.

# Example 4 - NO (Different domain):
# Seed: Niche:"Business analysis"/KWs:"corporate analysis, business strategy, market trends"
# Candidate: Niche:"True crime"/KWs:"unsolved mysteries, crime investigation, murder cases"
# COMPETITOR: No
# REASON: Keywords show completely different domains; business vs crime despite similar documentary format.

# Example 5 - NO (Different life stage):
# Seed: Niche:"Student productivity"/KWs:"study tips, exam preparation, note-taking systems"
# Candidate: Niche:"Retirement planning"/KWs:"retirement savings, senior living, pension advice"
# COMPETITOR: No
# REASON: Keywords indicate different life stages and completely different viewer needs.

# **CRITICAL**: Focus on keyword overlap and target audience. Format keywords (tutorial/podcast/documentary) should NOT disqualify if topics overlap.

# Ask: "Would the same person be interested in BOTH sets of keywords/topics?"

# **Output Format** (exactly 3 lines):
# COMPETITOR: [Yes/No]
# CONFIDENCE: [High/Medium/Low]
# REASON: [One sentence explaining keyword/topic overlap or audience mismatch]

# Example Output:
# COMPETITOR: Yes
# CONFIDENCE: High
# REASON: Keywords show 70% topic overlap in business/entrepreneurship domain targeting same audience.
# """
#     return prompt
