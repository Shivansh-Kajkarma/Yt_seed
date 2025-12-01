import pandas as pd

df = pd.read_csv("/home/rareboy/Internship/Kajkarma/YTSEED/testing_new/output/lennypodcat_final_ranked_FULL_20251201_192233.csv")

df_filtered = df[["Discovered_Channel_Name", "Final_Tier","Final_Status","LLM_Recheck_Reason"]]

df_filtered.to_csv("/home/rareboy/Internship/Kajkarma/YTSEED/testing_new/output/lennypodcat_final_ranked_DEBUG_20251201_192233.csv", index=False)