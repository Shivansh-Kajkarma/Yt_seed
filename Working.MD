## 1) What the client wants (plain & short)

The client — a single-user video editing agency owner — wants a small web app where he can upload a list of seed YouTube channels (CSV, Google Sheet, or a form). The system will:

* Read the seed channels he uploads.
* Analyze the recent videos of each seed channel to determine what topics that channel focuses on.
* Find other YouTube channels that publish on the same topics.
* Score how similar those discovered channels are to each seed.
* Optionally auto-subscribe (using his YouTube account) to sufficiently similar channels.
* Save everything in a single Google Sheet for his review and reporting.

**Important constraints:**

* Single user only (his account).
* Minimal frontend: upload & show status/results.
* Host the app on the company’s AWS EC2.
* No initial seed provided by client → we must accept CSV/Sheet/form uploads and also support creating dummy input locally for testing.

## 2) High-level product behavior (user story)

* He visits the web app and uploads a CSV or pastes a Google Sheet link (or fills a simple form later).
* He clicks “Run discovery.”
* Backend runs the discovery process (may take a few minutes); frontend can show progress.
* When done, the app writes a neat table of discovered channels into a Google Sheet and offers a CSV download.
* If auto-subscribe is enabled and a discovered channel’s similarity >= threshold (e.g., 0.7), the agent subscribes the client’s YouTube account to those channels (with prior OAuth consent).
* The Sheet contains: seed, discovered channel, URL, subs, similarity, level (1 or 2), date.

## 3) Concrete technical approach — end-to-end (what we will build)

### A. Inputs accepted

* CSV file upload (columns: `Seed_Channel_Name`, `Seed_Channel_URL`).
* Google Sheet link (sheet id → exported as CSV).
* (Later) Web form to paste a channel URL or name.

### B. Core processing pipeline (the exact steps)

1.  Load seed list (CSV or Sheet → normalized DataFrame).
2.  For each seed channel:
    * Resolve channel ID (from URL).
    * Get the channel’s uploads playlist id via `channels.list(part=contentDetails)`.
    * Pull N recent videos from uploads playlist (`playlistItems.list`, N = 20–30).
    * Extract fields: `video_title`, `description` (optional), `publish_date`, `video_id`.
3.  Combine video titles (and optional descriptions) into one text blob per channel.
4.  Preprocess text: lowercase, remove punctuation, remove stopwords, lemmatize.
5.  Compute a channel fingerprint (topic vector) using TF-IDF on that blob (or TF-IDF + embeddings hybrid).
6.  Search for candidate channels:
    * Use the top K keywords from fingerprint to run `search.list(q=..., type=video, maxResults=50)` to find videos.
    * Extract `channelId` for each result → aggregate channel hits into a candidate list.
7.  For each candidate channel (skip if in seen set):
    * Fetch its recent videos (like seed), build fingerprint, compute similarity (cosine) between seed and candidate fingerprints.
    * If similarity >= threshold (configurable, default 0.7) → mark as candidate match for Level 1.
    * Add candidate to `seen_channels`.
8.  Level 2 expansion:
    * For a controlled subset of Level 1 matches (e.g., top 5 per seed), repeat step 3–4 to produce Level 2 matches. Limit depth to 2 to avoid drift and quota explosions.
9.  Activity filter:
    * For each discovered channel, check the last upload date. If no uploads within X days (configurable, e.g., 180 days), mark as inactive and skip auto-subscribe (but can still log).
10. Dedup & results:
    * Maintain `seen_channels` persistent cache (JSON file) to avoid reprocessing across runs.
    * Accumulate results in memory/CSV `discovered_channels.csv`.
11. One-shot write:
    * At the end (or at safe intervals), push the entire result table to the Google Sheet (single batched write) and also keep local CSV saved.
12. Auto-subscribe (optional):
    * If `auto_subscribe=True`, use the stored OAuth credentials for the client’s account to call `youtube.subscriptions.insert` for channels whose similarity >= threshold and pass activity checks. Use rate limiting (`max_subs_per_day`).
13. Final output:
    * Google Sheet rows, downloadable CSV, and a short summary (counts, top matches).

### C. Key configuration (config.yaml)

`top_videos_per_channel: 25top_keywords_per_channel: 10similarity_threshold: 0.7max_channels_per_seed: 50max_level1_expansion_per_seed: 10max_level2_per_level1: 5max_subs_per_day: 50activity_days_threshold: 180auto_subscribe: false`

## 4) Implementation details & best practices (critical points)

* **OAuth & Permissions**
    * For reading public channel data, an API key is sufficient.
    * To subscribe on his behalf, we need OAuth with scope `youtube.force-ssl`. He must perform one consent step in the web app. The app will store the refresh token (encrypted / in a secure place on EC2).
    * Use `InstalledAppFlow` for development and a web OAuth flow for production if needed.
* **Deduplication & persistence**
    * Use `seen_channels.json` to persist processed channel IDs. Load at start, save periodically and at finish.
    * Also keep `discovered_channels.csv` as local durable output.
* **Quota & rate control**
    * Use batching calls where possible: `videos.list(id=comma_separated_ids)` up to 50 ids per call.
    * Limit search/list operations and cap Level 1/2 expansion to avoid quota exhaustion.
    * Implement a `quota_manager` that estimates units consumed; on `quotaExceeded` errors, persist state and abort/ retry next day.
* **Fingerprint & similarity**
    * Start with TF-IDF + cosine similarity (fast, interpretable).
    * If TF-IDF shows poor results (false positives or low separability), upgrade to sentence embeddings (`sentence-transformers`) or call a single LLM prompt to extract human-like topics.
    * Hybrid approach: TF-IDF for speed; embeddings for ambiguous cases.
* **Activity filter**
    * Compute `last_video_date`. If `now - last_video_date > activity_threshold`, mark inactive and skip auto-subscription.
* **Logging & monitoring**
    * Save logs locally and to a log file: `process_{timestamp}.log`.
    * Track metrics: `num_seeds_processed`, `num_candidates_found`, `num_subscriptions_attempted`, `quota_used_today`.
* **Security**
    * Put `client_secret.json` and `token.json` in a secure folder on EC2 only accessible to the app user.
    * Don’t commit secrets to Git.
    * Have a simple token revocation instruction for the client.

## 5) Inputs / Outputs — exact schema

**Input (seed CSV / Sheet)**
Columns:

* Seed_Channel_Name
* Seed_Channel_URL

(optional) Category, Notes

**Output (discovered CSV / Sheet)**
Columns:

* Seed Channel
* Discovered Channel
* Channel ID
* URL
* Subscriber Count
* Similarity Score (0–1)
* Level (1 or 2)
* Last Upload Date
* Active (Yes/No)
* Date Added

## 6) End-to-end flow (short checklist to follow now)

1.  Create dummy `seed_channels.csv` (3–5 seeds) and a test Google Sheet copy.
2.  Implement `load_input()` to accept CSV or Sheet link and return DataFrame.
3.  Implement `get_channel_id_from_url()`, `get_uploads_playlist_id()`, `get_recent_videos(playlist_id)`.
4.  Implement `extract_keywords_and_vector()` using TF-IDF.
5.  Implement `search_candidates_by_keywords()` and candidate aggregation.
6.  Implement `compute_similarity()` and scoring logic.
7.  Implement `seen_channels` persistence and dedup logic.
8.  Implement batch-write to local CSV and single batched write to Google Sheets.
9.  Implement activity filter and optional `auto_subscribe()` (with safe default off).
10. Test locally with dummy data. Save logs and results.
11. Hand over the function(s) or module with `requirements.txt` and usage README for Uday to call as Celery task.

## 7) Testing plan (what you must verify)

* **Unit test:** `extract_keywords_and_vector()` on known seed data → check top keywords look sensible.
* **Integration test:** run `process_channels()` for one seed → output CSV has expected columns and plausible discovered channels.
* **Edge case test:** seed channel with <5 videos, channel with no description, or private videos → ensure graceful behavior.
* **Quota test:** simulate many channels and ensure `quotaExceeded` handling saves state and exits cleanly.

## 8) Deliverables you should produce

* `ml_engine.py` (or module) containing:
    * `load_input()`, `process_channels()`, `extract_keywords()`, `compute_similarity()`, `log_results()`, `auto_subscribe()` (config-driven).
* `requirements.txt`
* `config.yaml` with sensible defaults
* Dummy `seed_channels.csv` and a sample `discovered_channels.csv`
* `README.md` explaining how to run locally and how Uday should call the main function (signature and expected inputs/outputs)
* Sample `token_placeholder_info.txt` describing how to add `client_secret.json` and perform OAuth for his account.