from pymongo import MongoClient
import pandas as pd

# 1. Connect to MongoDB
client = MongoClient("mongodb+srv://uishivansh2503_db_user:3svnnxBJ04eu4iFb@kajkarma.wapdfls.mongodb.net/?appName=Kajkarma")
db = client["yt_3_test1"]
col = db["ALI_ABDAAL_final_ranked"]
query = {"Final_Tier": {"$in": [1, 2]}}
projection = {
    "_id": 0,
    "Discovered_Channel_ID": 1,
    "Discovered_Channel_Name": 1,
    "run_tag": 1,
    "Final_Tier": 1,
    "Final_Status": 1,
    "Seed_Channel_Name": 1,
    "Discovered_Channel_URL": 1,   # we'll move this to last
}

docs = list(col.find(query, projection))

# 2. Convert to DataFrame
df = pd.DataFrame(docs)

# 3. Reorder columns so URL is last
ordered_cols = [
    "Discovered_Channel_ID",
    "Discovered_Channel_Name",
    "run_tag",
    "Final_Tier",
    "Final_Status",
    "Seed_Channel_Name",
    "Discovered_Channel_URL",   # last
]

df = df[ordered_cols]

# 4. Export CSV
df.to_csv("filtered_channels.csv", index=False)

# 5. Export JSON (preserving order)
df.to_json("filtered_channels.json", orient="records", indent=2)

print("✅ Exported CSV + JSON with URL as last column!")