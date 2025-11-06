import os
import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, BulkWriteError
from dotenv import load_dotenv
from typing import List, Dict, Any

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