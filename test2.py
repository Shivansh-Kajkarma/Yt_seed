import pandas as pd
import json
import re
import time
from pathlib import Path
import os
from dotenv import load_dotenv

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

if MODEL_PROVIDER == "gemini":
    if GOOGLE_API_KEY:
        import google.generativeai as genai
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            # Use a model good at reasoning like gemini-pro
            gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini model configured.")
        except Exception as e:
            print(f"❌ Gemini configuration failed: {e}")
            gemini_model = None # Ensure it's None if setup fails
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
            gpt_client = None # Ensure it's None if setup fails
    else:
        print("⚠️ OPENAI_API_KEY not found. Cannot use GPT.")
else:
    print(f"❌ Invalid MODEL_PROVIDER: {MODEL_PROVIDER}. Choose 'gemini' or 'gpt'.")


# --- LLM Prompt Function (Copied from above) ---
def create_competitor_prompt(seed_name, seed_niche, discovered_name, discovered_niche):
    prompt = f"""You are an expert YouTube analyst classifying channel relationships.

SEED CHANNEL: "{seed_name}"
Primary Niche: "{seed_niche or 'N/A'}"

CANDIDATE CHANNEL: "{discovered_name}"
Primary Niche: "{discovered_niche or 'N/A'}"

TASK: Determine if the CANDIDATE channel is a **direct competitor** to the SEED channel.

A **direct competitor** has:
1.  **Very Similar Niche:** Covers almost the same core topics.
2.  **Similar Intent/Format:** Serves the same audience need (e.g., both Education, both Storytelling, both News Analysis).
3.  **Likely Same Language/Region:** Niches often imply this (e.g., "UPSC exam prep" implies Hindi/India).

Consider these examples:
- "Business Documentaries" vs. "Corporate Storytelling" → Yes (Similar niche & intent)
- "Product Management Education" vs. "Tech Career Prep" → Yes (Similar niche & intent)
- "Business Documentaries" vs. "Business Skills Tutorials" → No (Different intent: Storytelling vs. Education)
- "Productivity Strategies (English)" vs. "Exam Motivation (Hindi)" → No (Different language/audience implied by niche)
- "Geopolitical Analysis" vs. "True Crime Documentaries" → No (Different core topic)

**Output Format:**
Return *only* two lines:
COMPETITOR: [Yes/No]
REASON: [Provide a brief (10-15 word) justification based on niche and intent comparison]

Example 1:
COMPETITOR: Yes
REASON: Both channels focus on product management careers with an educational intent.

Example 2:
COMPETITOR: No
REASON: Seed is storytelling/documentary, Candidate is educational/career prep, serving different audience needs.

Example 3:
COMPETITOR: No
REASON: Niches indicate different primary languages and target audiences (English productivity vs. Hindi exam prep).
"""
    return prompt

# --- Function to Call LLM and Parse Output ---
def classify_competitor(seed_name, seed_niche, discovered_name, discovered_niche):
    """Calls the configured LLM and parses the Yes/No competitor status and reason."""
    if (MODEL_PROVIDER == "gemini" and not gemini_model) or \
       (MODEL_PROVIDER == "gpt" and not gpt_client):
        return "Error", "LLM client not configured"

    prompt = create_competitor_prompt(seed_name, seed_niche, discovered_name, discovered_niche)
    
    for attempt in range(MAX_RETRIES):
        try:
            raw_response = ""
            if MODEL_PROVIDER == "gemini":
                response = gemini_model.generate_content(prompt)
                raw_response = response.text.strip()
            elif MODEL_PROVIDER == "gpt":
                response = gpt_client.chat.completions.create(
                    model="gpt-4o", # Using mini for cost/speed, can upgrade if needed
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=60, # Enough for "COMPETITOR: Yes/No \n REASON: ..."
                )
                raw_response = response.choices[0].message.content.strip()

            # --- Robust Parsing ---
            is_competitor = "Unknown"
            reason = "Parsing failed"

            comp_match = re.search(r"COMPETITOR:\s*(Yes|No)", raw_response, re.IGNORECASE)
            reason_match = re.search(r"REASON:\s*(.*)", raw_response, re.IGNORECASE)

            if comp_match:
                is_competitor = comp_match.group(1).capitalize() # Standardize Yes/No

            if reason_match:
                reason = reason_match.group(1).strip()
            elif not comp_match and raw_response: # Fallback if parsing fails but got text
                 reason = f"LLM Output: {raw_response[:100]}" # Store raw output snippet

            if is_competitor != "Unknown": # If we got at least Yes/No, return
                 print(f"      LLM Result: {is_competitor} | Reason: {reason[:50]}...")
                 return is_competitor, reason
            else:
                 print(f"      ⚠️ Failed to parse LLM Output: {raw_response}")
                 reason = f"Parsing Failed: {raw_response[:100]}"
                 # Continue to retry if parsing failed

        except Exception as e:
            print(f"      ❌ LLM API Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"      Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                return "Error", f"API Error after {MAX_RETRIES} attempts: {e}"
        
        # If parsing failed on last attempt
        if attempt == MAX_RETRIES -1 and is_competitor == "Unknown":
             return "Error", reason


    return "Error", "Max retries reached without success" # Should not be reached


# --- Main Script Logic ---
def main():
    start_time = time.time()
    print(f"--- Starting Competitor Classification ({MODEL_PROVIDER.upper()}) ---")
    
    input_csv_path = Path("./check.csv") # Assuming check.csv is in the same folder
    output_csv_path = Path(f"./check_competitors_{MODEL_PROVIDER}.csv")

    if not input_csv_path.exists():
        print(f"❌ ERROR: Input file not found at {input_csv_path}")
        return

    if (MODEL_PROVIDER == "gemini" and not gemini_model) or \
       (MODEL_PROVIDER == "gpt" and not gpt_client):
         print(f"❌ ERROR: {MODEL_PROVIDER.upper()} client failed to configure. Check API keys.")
         return

    try:
        df = pd.read_csv(input_csv_path)
        print(f"✅ Loaded {len(df)} rows from {input_csv_path.name}")
    except Exception as e:
        print(f"❌ ERROR reading CSV: {e}")
        return

    # Add new columns if they don't exist
    if 'Is_Competitor' not in df.columns:
        df['Is_Competitor'] = "Pending"
    if 'Reason' not in df.columns:
        df['Reason'] = ""

    # Process rows that haven't been classified yet
    rows_to_process = df[df['Is_Competitor'] == "Pending"].index
    print(f"Found {len(rows_to_process)} rows to classify.")

    processed_count = 0
    for index in rows_to_process:
        row = df.loc[index]
        print(f"\nProcessing row {index + 1}/{len(df)}: {row['Seed_Channel_Name']} vs {row['Discovered_Channel_Name']}")

        competitor_status, reason_text = classify_competitor(
            row['Seed_Channel_Name'],
            row.get('Seed_Niche', ''), # Use .get for safety if column missing
            row['Discovered_Channel_Name'],
            row.get('Discovered_Niche', '') # Use .get for safety
        )

        df.loc[index, 'Is_Competitor'] = competitor_status
        df.loc[index, 'Reason'] = reason_text
        processed_count += 1

        # --- Save progress periodically and sleep ---
        if processed_count % 10 == 0: # Save every 10 rows
             try:
                  df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                  print(f"--- Saved progress ({processed_count} processed) ---")
             except Exception as e:
                  print(f"  ⚠️ Error saving intermediate progress: {e}")

        time.sleep(1) # Add a small delay between API calls

    # Final save
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ Successfully classified {processed_count} rows.")
        print(f"💾 Final results saved to: {output_csv_path.name}")
    except Exception as e:
        print(f"❌ Error saving final CSV: {e}")

    end_time = time.time()
    print(f"⏱️ Total time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()