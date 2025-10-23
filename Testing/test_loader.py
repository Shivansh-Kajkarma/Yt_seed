from channel_loader import load_seed_channels

def run_tests():
    print("--- Test 1: Loading from local CSV ---")
    # This assumes 'seed_channels.csv' is in the same directory
    local_df = load_seed_channels(source="seed_channels.csv", source_type='csv')
    
    if local_df is not None:
        print("Local CSV load SUCCESS. Data:")
        print(local_df.head())
    else:
        print("Local CSV load FAILED.")
        
    print("\n" + "="*30 + "\n")
    
    # print("--- Test 2: Loading from Google Sheet URL ---")
    # # --- DUMMY GOOGLE SHEET LINK ---
    # # To create your own:
    # # 1. Create a Google Sheet with 'Channel_Name' and 'Channel_URL' columns
    # # 2. Add some data (you can copy/paste from your CSV)
    # # 3. Go to File -> Share -> Publish to web
    # # 4. Select the correct sheet, and choose "Comma-separated values (.csv)"
    # # 5. Click "Publish" and copy the generated link.
    
    
    # # This is a real, public test link with the same structure
    # dummy_sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSr6AmKcMa9bIEnxerj5KgPvPQ1-K3aGk3Qkqk1BCuvhK0Blj62s0giGjxjjP7hqUhVLPcMNat2fJCf/pub?gid=0&single=true&output=csv"
    # gsheet_df = load_seed_channels(source=dummy_sheet_url, source_type='google_sheet')
    
    # if gsheet_df is not None:
    #     print("Google Sheet load SUCCESS. Data:")
    #     print(gsheet_df.head())
    # else:
    #     print("Google Sheet load FAILED.")

if __name__ == "__main__":
    run_tests()