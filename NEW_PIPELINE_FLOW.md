# 🚀 NEW PIPELINE FLOW - YTSEED Project

## 📋 Overview
The pipeline has been restructured with **yt-dlp deep scanning** and **2 new client inputs**.

---

## 🔄 New Flow Architecture

```
INPUT: Google Sheet + input_format + clients_intent
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: ytdlp_scripts/phase1_seed_processing.py           │
│ - Fetch videos via YouTube API                             │
│ - Deep scan with yt-dlp (transcripts, tags, chapters)      │
│ - Generate fingerprint WITH format/intent constraints      │
│ - Save: {run_tag}_phase1 + {run_tag}_phase1_fingerprints  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: SCRIPTS/phase2_get_discovered_channels.py         │
│ - Search YouTube for candidates using keywords             │
│ - Apply static filters (country, subscribers, etc.)        │
│ - Fetch recent videos (API metadata only)                  │
│ - Save: {run_tag}_phase2                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3 STEP 1: ytdlp_scripts/phase3_step1_api_filter.py  │
│ - Quick LLM filter using API metadata (titles/desc)        │
│ - Filters by format & intent (GPT-4o-mini)                 │
│ - Drop obvious mismatches (Gaming/Vlog when need Podcast)  │
│ - Save: {run_tag}_phase3_step1                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3 STEP 2: ytdlp_scripts/phase3_step2_deep_scan.py   │
│ - Deep scan 5 videos per candidate with yt-dlp             │
│ - Fetch transcripts, chapters, tags, duration              │
│ - Calculate metrics (no filtering, just data collection)   │
│ - Save: {run_tag}_phase3_step2                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3 STEP 3A: ytdlp_scripts/phase3_step3a_scoring_llm.py│
│ - Strict format validation using deep data                 │
│ - Analyzes Start/Middle/End of transcripts                 │
│ - Verifies format match (Podcast vs Tutorial vs Doc)       │
│ - Assigns confidence score                                 │
│ - Save: {run_tag}_phase3_step3a                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3 STEP 3B: ytdlp_scripts/phase3_step3b_scoring_emb.py│
│ - Calculate embedding similarity                           │
│ - Compare seed transcripts vs candidate transcripts        │
│ - Uses OpenAI embeddings                                   │
│ - Assigns similarity score (0-1)                           │
│ - Save: {run_tag}_phase3_step3b                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: ytdlp_scripts/phase4_final_ranking.py            │
│ - Combine format match + similarity scores                 │
│ - Apply thresholds for Tier 1/2                            │
│ - Run tie-breaker LLM for edge cases                       │
│ - Generate final CSV report                                │
│ - Save: {run_tag}_phase4 (final results)                   │
└─────────────────────────────────────────────────────────────┘
    ↓
OUTPUT: Tier 1/2 channels with reasoning
```

---

## 🆕 New Input Parameters

### 1. **`input_format`** (Required)
**Description:** The type of content the client is looking for.

**Examples:**
- `"Podcast"`
- `"Tutorial"`
- `"Documentary"`
- `"Video Essay"`
- `"Product Review"`
- `"Vlog"`
- `"Interview"`

**Used In:**
- Phase 1: Passed to `get_channel_fingerprint_oneshot()` for targeted fingerprint
- Phase 3 Step 1: LLM filters candidates by format
- Phase 3 Step 3A: Strict format validation
- Phase 4: Final scoring and tier assignment

---

### 2. **`clients_intent`** (Required)
**Description:** The specific topic/niche the client is targeting.

**Examples:**
- `"Interviews with startup founders"`
- `"Python programming tutorials"`
- `"True crime documentaries"`
- `"Tech product reviews"`
- `"Fitness and nutrition"`

**Used In:**
- Phase 1: Passed to `get_channel_fingerprint_oneshot()` for keyword extraction
- Phase 3 Step 1: LLM filters by intent relevance
- Phase 4: Final ranking considers intent match

---

## 📊 Data Flow Between Phases

### Phase 1 Output → Phase 2 Input
```json
{
  "metadata": {
    "run_tag": "moon",
    "client_constraints": {
      "format": "Podcast",
      "intent": "Interviews with founders"
    }
  },
  "channels": {
    "UC123": {
      "channel_name": "Lenny's Podcast",
      "fingerprint": {
        "profile": "...",
        "keywords": {...}
      }
    }
  }
}
```

### Phase 2 Output → Phase 3 Step 1 Input
```json
{
  "Discovered_Channel_Name": "Dave2D",
  "Discovered_Channel_ID": "UCVYamHliCI9rw1tHR1xbkfw",
  "Discovered_Channel_Description": "Tech reviews...",
  "Discovered_Videos_JSON": "[{video_id, title, description}...]",
  "Subscriber_Count": 5000000
}
```

### Phase 3 Step 2 Output → Phase 3 Step 3A/B Input
```json
{
  "Discovered_Channel_Name": "Dave2D",
  "Deep_Scan_Data": "[{title, duration, caption_tracks, has_chapters, tags}...]",
  "Avg_Duration": 720,
  "Videos_With_Chapters": 3
}
```

### Phase 4 Final Output
```json
{
  "tier": 1,
  "Discovered_Channel_Name": "Dave2D",
  "Discovered_Channel_URL": "...",
  "score_format_match": 0.95,
  "score_similarity": 0.82,
  "reason": "Perfect format match: Professional reviews with structured chapters",
  "format_confidence": "HIGH",
  "Discovered_From_Run": "MKBHD"
}
```

---

## 🔧 Required Updates

### 1. **`main.py` (FastAPI Endpoints)**
**Changes Needed:**
- Update `/start_pipeline` to accept 3 parameters:
  - `sheet_url` (existing)
  - `input_format` (new)
  - `clients_intent` (new)
- Pass these to Celery task

**Example:**
```python
@app.post("/start_pipeline")
def start_pipeline(sheet_url: str, input_format: str, clients_intent: str):
    task = run_phase_pipeline.delay(sheet_url, None, input_format, clients_intent)
    return {"task_id": task.id, "message": "Pipeline started"}
```

---

### 2. **`celery_worker.py`**
**Changes Needed:**
- Update task signature to include new parameters
- Pass them to `full_pipeline_from_sheet()`

**Example:**
```python
@celery_app.task(bind=True)
def run_phase_pipeline(self, sheet_url: str, seed_dict: dict = None, 
                       input_format: str = "General", clients_intent: str = "General"):
    result = full_pipeline_from_sheet(self, sheet_url, seed_dict, input_format, clients_intent)
    return {"status": "success", "details": result}
```

---

### 3. **`utils/pipeline_wrapper.py`**
**Changes Needed:**
- Update function signature to accept new parameters
- Pass to Phase 1 script (ytdlp version)
- **Remove old SCRIPTS/phase1 call**
- **Replace with ytdlp_scripts/phase1 call**
- Update Phase 3 to call all 4 steps (step1, step2, step3a, step3b)
- Add Phase 4 call

**Example:**
```python
def full_pipeline_from_sheet(celery_task, sheet_url, seed_dict, input_format, clients_intent):
    if seed_dict:
        run_tag = normalize_run_tag(seed_dict["Channel_Name"])
        
        # Phase 1: NEW ytdlp version
        subprocess.run([
            "python", "ytdlp_scripts/phase1_seed_processing.py",
            run_tag, seed_dict["Channel_URL"], input_format, clients_intent
        ])
        
        # Phase 2: Existing
        phase2_main(run_tag)
        
        # Phase 3: NEW multi-step
        subprocess.run(["python", "ytdlp_scripts/phase3_step1_api_filter.py", run_tag])
        subprocess.run(["python", "ytdlp_scripts/phase3_step2_deep_scan.py", run_tag])
        subprocess.run(["python", "ytdlp_scripts/phase3_step3a_scoring_llm.py", run_tag])
        subprocess.run(["python", "ytdlp_scripts/phase3_step3b_scoring_emb.py", run_tag])
        
        # Phase 4: NEW final ranking
        subprocess.run(["python", "ytdlp_scripts/phase4_final_ranking.py", run_tag])
```

---

### 4. **Feedback Loop**
**Changes Needed:**
- Update to load from `{run_tag}_phase4` (not `phase3`)
- Still queue Tier 1/2 channels as new seeds
- Pass same `input_format` and `clients_intent` to child seeds

---

## 📝 Key Differences from Old Pipeline

| Aspect | Old Pipeline | New Pipeline |
|--------|-------------|--------------|
| **Phase 1** | Basic API fetch | Deep yt-dlp scan with transcripts |
| **Fingerprint** | Generic keywords | Format + Intent aware |
| **Phase 3** | Single tier scoring | 4-step funnel (filter → scan → verify → score) |
| **Format Check** | Not strict | Strict validation via transcripts |
| **Embedding** | Early filter (Phase 2.5) | Late scoring (Phase 3 Step 3B) |
| **Final Output** | Phase 3 results | Phase 4 combined ranking |

---

## 🎯 Example Request

### Old API Call:
```bash
POST /start_pipeline?sheet_url=https://docs.google.com/spreadsheets/d/...
```

### New API Call:
```bash
POST /start_pipeline
Body: {
  "sheet_url": "https://docs.google.com/spreadsheets/d/...",
  "input_format": "Podcast",
  "clients_intent": "Interviews with tech founders about scaling startups"
}
```

---

## ✅ Current Status

### ✅ Completed
- All ytdlp_scripts phases implemented
- Phase 1: Deep scanning working
- Phase 3 Steps 1-3: Format validation logic ready
- Client constraints saved to fingerprints

### ⚠️ Needs Update
- `main.py`: Add input_format/clients_intent parameters
- `celery_worker.py`: Update task signature
- `pipeline_wrapper.py`: Switch to new phase flow
- Feedback loop: Use phase4 results
- Frontend: Update form to collect 2 new inputs

---

## 🔗 File Dependencies

```
main.py
  ↓ calls
celery_worker.py
  ↓ calls
utils/pipeline_wrapper.py
  ↓ orchestrates
  ├─ ytdlp_scripts/phase1_seed_processing.py
  ├─ SCRIPTS/phase2_get_discovered_channels.py
  ├─ ytdlp_scripts/phase3_step1_api_filter.py
  ├─ ytdlp_scripts/phase3_step2_deep_scan.py
  ├─ ytdlp_scripts/phase3_step3a_scoring_llm.py
  ├─ ytdlp_scripts/phase3_step3b_scoring_emb.py
  └─ ytdlp_scripts/phase4_final_ranking.py
```

---

## 🚦 Next Steps

1. ✅ Review this document
2. 🔧 Update `main.py` endpoints
3. 🔧 Update `celery_worker.py` task
4. 🔧 Update `pipeline_wrapper.py` orchestration
5. 🧪 Test with sample seed + format + intent
6. 📱 Update frontend form

---

**Questions?**
- How should we handle missing format/intent? (Default to "General"?)
- Should format be dropdown or free text?
- Any validation needed for intent length?
