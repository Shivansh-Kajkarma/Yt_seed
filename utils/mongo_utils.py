import os
import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, BulkWriteError
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime
from typing import Tuple

# --- 1. Load Config ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "youtube_competitor_db" 

# --- 2. Connection Cache ---
_client = None
_db = None

def get_mongo_db():
    """
    Establishes connection to MongoDB and returns the database object.
    Caches the connection for efficiency.
    """
    global _client, _db
    
    # --- THIS IS THE FIX ---
    # We MUST check with 'is not None', not 'if _db:'
    if _db is not None:
        return _db
    # --- END OF FIX ---
        
    if not MONGO_URI:
        print("❌ ERROR: MONGO_URI not found in .env file.")
        raise ValueError("MONGO_URI not set")

    try:
        if _client is None:
            print("  Connecting to MongoDB Atlas...")
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            _client.server_info()
            print("  ✅ MongoDB connection successful.")
        
        _db = _client[DB_NAME]
        return _db
        
    except ConnectionFailure as e:
        print(f"❌ CRITICAL: MongoDB connection failed: {e}")
        _client = None 
        raise # <-- ADDED THIS: Re-raise the connection error

# --- 3. Save/Load Functions (Now with 'raise' on error) ---

def save_dataframe_to_mongo(df: pd.DataFrame, collection_name: str, unique_key_column: str):
    """
    Saves a DataFrame to a MongoDB collection using bulk 'upsert'.
    This PREVENTS duplicates by updating existing records.
    """
    if df.empty:
        print(f"  ⚠️  DataFrame is empty. Skipping save to '{collection_name}'.")
        return
        
    try:
        db = get_mongo_db()
        collection = db[collection_name]
        
        records = df.to_dict('records')
        
        operations = [
            UpdateOne(
                {unique_key_column: rec[unique_key_column]},
                {"$set": rec},
                upsert=True
            )
            for rec in records
        ]
        
        print(f"  Saving {len(records)} records to MongoDB collection: '{collection_name}'...")
        result = collection.bulk_write(operations)
        print(f"  ✅ Success: {result.upserted_count} inserted, {result.modified_count} updated.")
        
    except BulkWriteError as bwe:
        print(f"  ❌ ERROR during bulk save to '{collection_name}': {bwe.details}")
        raise # <-- ADDED
    except Exception as e:
        print(f"  ❌ UNEXPECTED ERROR saving to '{collection_name}': {e}")
        raise # <-- ADDED


def save_record_to_mongo(record: Dict[str, Any], collection_name: str, unique_key_column: str):
    """
    Saves a SINGLE dictionary record to a MongoDB collection.
    Used by Phase 2 for saving one channel at a time.
    """
    try:
        db = get_mongo_db()
        collection = db[collection_name]
        
        key = {unique_key_column: record[unique_key_column]}
        
        collection.update_one(key, {"$set": record}, upsert=True)
        # print(f"  ✅ Saved record for {record[unique_key_column]} to '{collection_name}'")

    except Exception as e:
        print(f"  ❌ UNEXPECTED ERROR saving record to '{collection_name}': {e}")
        raise # <-- ADDED


def save_json_blob(data: Dict[str, Any], collection_name: str, unique_key: str, key_value: str):
    """
    Saves a large, complex JSON blob (like the fingerprint file) as a single document.
    """
    try:
        db = get_mongo_db()
        collection = db[collection_name]
        
        print(f"  Saving JSON blob to '{collection_name}' with key {key_value}...")
        collection.replace_one({unique_key: key_value}, data, upsert=True)
        print(f"  ✅ JSON blob saved.")
        
    except Exception as e:
        print(f"  ❌ UNEXPECTED ERROR saving JSON blob to '{collection_name}': {e}")
        raise # <-- ADDED


def load_collection_as_df(collection_name: str, query_filter: Dict = None) -> pd.DataFrame:
    """
    Loads an entire MongoDB collection into a pandas DataFrame.
    """
    if query_filter is None:
        query_filter = {}
        
    print(f"  Loading collection '{collection_name}' with filter {query_filter}...")
    try:
        db = get_mongo_db()
        collection = db[collection_name]
        
        cursor = collection.find(query_filter)
        df = pd.DataFrame(list(cursor))
        
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
            
        print(f"  ✅ Loaded {len(df)} records.")
        return df
        
    except Exception as e:
        print(f"  ❌ UNEXPECTED ERROR loading collection '{collection_name}': {e}")
        raise # <-- ADDED


def check_quota_and_pause(e, run_tag: str, seed_name: str | None = None):
    """
    Centralized YouTube quotaExceeded handling.
    This function now lives in mongo_utils.
    """
    # Check for quota error in the exception message
    if "quotaExceeded" in str(e):
        print("--- 🛑 YOUTUBE QUOTA EXCEEDED ---")
        try:
            # We are already in mongo_utils, so we can call save_json_blob
            save_json_blob(
                {
                    "run_tag": run_tag,
                    "seed": seed_name,
                    "status": "paused_due_to_quota",
                    "reason": "quotaExceeded",
                    "timestamp": datetime.now().isoformat()
                },
                "run_progress",
                "run_tag",
                run_tag
            )
            print("--- ✅ Paused status saved to MongoDB ---")
        except Exception as mongo_e:
            print(f"--- ❌ FAILED to save paused status to MongoDB: {mongo_e} ---")
            pass # We still want to exit even if Mongo save fails
            
        # This raises a SystemExit, which will stop the subprocess
        raise SystemExit("YouTube quota exceeded. Safe exit for now.")
    
# ==================================================
# 2. HELPER FUNCTIONS (REFACTORED FOR MONGO)
# ==================================================

def load_seen_channels_from_mongo(run_tag: str, MONGO_SEEN_CHANNELS_LOG: str) -> Tuple[dict, set]:
    """Loads the high-level 'seen' log from Mongo for this run_tag."""
    print(f"  📂 Loading previously seen channels for '{run_tag}' from Mongo...")
    seen_data_dict = {}
    seen_ids = set()
    try:
        df = load_collection_as_df(
            MONGO_SEEN_CHANNELS_LOG, 
            {"run_tag": run_tag}
        )
        if not df.empty:
            seen_ids = set(df["Channel_ID"].astype(str).tolist())
            # Convert to dict for fast lookups/updates
            seen_data_dict = df.set_index("Channel_ID").to_dict('index')
            print(f"  ✅ Loaded {len(seen_ids)} previously seen channels.")
        else:
            print("  📂 No existing seen channels log found. Starting fresh.")
    except Exception as e:
        print(f"  ⚠️  Could not read seen channels log: {e}. Starting fresh.")
    
    # Return dict for fast in-memory updates, set for fast lookups
    return seen_data_dict, seen_ids

def save_seen_channels_to_mongo(seen_data_dict: dict, MONGO_SEEN_CHANNELS_LOG: str):
    """Saves the 'seen' log back to Mongo using upsert."""
    if not seen_data_dict:
        return
    try:
        # Convert dict values back to a list of records
        records_list = list(seen_data_dict.values())
        df_to_save = pd.DataFrame(records_list)
        
        # Ensure key columns exist
        if "Channel_ID" not in df_to_save.columns or "run_tag" not in df_to_save.columns:
            print("  ❌ ERROR: Seen channels data is missing Channel_ID or run_tag.")
            return

        # Create a composite key for upserting
        df_to_save["_seen_key"] = df_to_save["run_tag"] + "::" + df_to_save["Channel_ID"]
        
        save_dataframe_to_mongo(
            df_to_save,
            MONGO_SEEN_CHANNELS_LOG,
            unique_key_column="_seen_key"
        )
        # print(f"  ...seen log updated in Mongo.") # Too noisy
    except Exception as e:
        print(f"  ❌ Error saving seen channels log to Mongo: {e}")

def load_cached_ids_from_mongo(collection_name: str) -> set:
    """Loads ONLY the IDs from the main Phase 2 data collection."""
    print(f"  🔄 Loading already cached channel IDs from '{collection_name}'...")
    try:
        df = load_collection_as_df(collection_name)
        if not df.empty and "Discovered_Channel_ID" in df.columns:
            cached_ids = set(df["Discovered_Channel_ID"].astype(str).tolist())
            print(f"  ✅ Found {len(cached_ids)} channels in cache to skip.")
            return cached_ids
    except Exception as e:
        print(f"  ⚠️  Could not read cached channels: {e}")
    
    print("  📂 No previously cached channels found.")
    return set()

def load_search_cache_from_mongo(cache_key: str, MONGO_SEARCH_CACHE: str) -> (set | None):
    """Tries to load a cached search result from Mongo."""
    try:
        db = load_collection_as_df(MONGO_SEARCH_CACHE, {"_cache_key": cache_key})
        if not db.empty:
            # Load the list of IDs from the 'result_ids' field
            candidate_ids = set(db.iloc[0].get("result_ids", []))
            if candidate_ids:
                print(f"  ✅ Found existing search cache in Mongo: {cache_key}")
                print(f"  Loaded {len(candidate_ids)} candidates from cache. (0 quota units used)")
                return candidate_ids
    except Exception as e:
        print(f"  ⚠️ Error loading Mongo search cache: {e}")
    
    print(f"  ℹ️ No search cache found in Mongo for key: {cache_key}")
    return None

def save_search_cache_to_mongo(cache_key: str, candidate_ids: set, run_tag: str, MONGO_SEARCH_CACHE: str):
    """Saves a search result to the Mongo cache."""
    try:
        payload = {
            "_cache_key": cache_key,
            "result_ids": list(candidate_ids), # Convert set to list for JSON
            "run_tag": run_tag,
            "created_at": datetime.now().isoformat()
        }
        # Use save_json_blob to upsert this single document
        save_json_blob(
            payload,
            MONGO_SEARCH_CACHE,
            unique_key="_cache_key",
            key_value=cache_key
        )
        print(f"  ✅ Saved {len(candidate_ids)} found candidates to Mongo cache.")
    except Exception as e:
        print(f"  ⚠️ Error saving to Mongo search cache: {e}")
