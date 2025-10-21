# YouTube Channel Discovery Tool

This project discovers new YouTube channels based on a set of seed channels.

## Phase 1: Channel Loader

This module (`channel_loader.py`) is responsible for loading the initial list of seed channels.

### Features
- Loads channels from a local `.csv` file.
- Loads channels from a public Google Sheet URL.

### Setup
1.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
2.  Ensure you have a `seed_channels.csv` file in the root directory with `Channel_Name` and `Channel_URL` columns.

### How to Use the Module

You can import and use the `load_seed_channels` function in other parts of your project.

```python
from channel_loader import load_seed_channels

# Load from CSV (default)
df_csv = load_seed_channels(source="seed_channels.csv")

# Load from Google Sheet
# Note: The sheet must be "Published to the web" as a CSV.
sheet_url = "YOUR_PUBLIC_GOOGLE_SHEET_URL_HERE"
df_gsheet = load_seed_channels(source=sheet_url, source_type='google_sheet')

if df_csv is not None:
    print(df_csv)