import pandas as pd
import re
from typing import Optional, Literal

def _load_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Internal function to load seed channels from a local CSV file.
    
    Args:
        file_path: The path to the .csv file.
        
    Returns:
        A DataFrame with 'Channel_Name' and 'Channel_URL', or None if loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        # Validate required columns
        if 'Channel_Name' not in df.columns or 'Channel_URL' not in df.columns:
            print(f"Error: CSV file '{file_path}' must contain 'Channel_Name' and 'Channel_URL' columns.")
            return None
        
        print(f"Successfully loaded {len(df)} channels from {file_path}")
        return df[['Channel_Name', 'Channel_URL']]
    
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error loading CSV from '{file_path}': {e}")
        return None

def _load_from_google_sheet(sheet_url: str) -> Optional[pd.DataFrame]:
    """
    Internal function to load seed channels from a PUBLIC Google Sheet URL.
    
    Note: The Google Sheet must be "Published to the web" as a CSV.
    (File -> Share -> Publish to web -> Select sheet -> Select CSV)
    
    Args:
        sheet_url: The public URL of the Google Sheet (must be a CSV export link).
        
    Returns:
        A DataFrame with 'Channel_Name' and 'Channel_URL', or None if loading fails.
    """
    try:
        # Check if it's a direct CSV export link (from "publish" or "export")
        if 'output=csv' in sheet_url or 'export?format=csv' in sheet_url:
            csv_export_url = sheet_url
            print(f"Using direct CSV URL: {csv_export_url}")
        else:
            # Try to construct the export link from a standard /edit URL
            match = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', sheet_url)
            if not match:
                print("Error: Invalid Google Sheet URL. Must be a standard /edit link or a public 'output=csv' link.")
                return None
            
            sheet_id = match.group(1)
            gid_match = re.search(r'gid=([0-9]+)', sheet_url)
            gid = gid_match.group(1) if gid_match else '0'
            
            csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            print(f"Constructed CSV export URL: {csv_export_url}")

        df = pd.read_csv(csv_export_url, encoding='utf-8-sig')
        
        # print(f"[Debug] Columns found by pandas: {list(df.columns)}")
        
        df.columns = df.columns.str.strip()

        if 'Channel_Name' not in df.columns or 'Channel_URL' not in df.columns:
            print("Error: Google Sheet must contain 'Channel_Name' and 'Channel_URL' columns.")
            print(f"[Debug] Sanitized columns: {list(df.columns)}") # More debug
            return None
            
        print(f"Successfully loaded {len(df)} channels from Google Sheet.")
        return df[['Channel_Name', 'Channel_URL']]
        
    except Exception as e:
        print(f"Error loading Google Sheet from '{sheet_url}': {e}")
        print("Please ensure the URL is correct and the sheet is 'Published to the web' as a CSV.")
        return None

def load_seed_channels(
    source: str, 
    source_type: Literal['csv', 'google_sheet'] = 'csv'
) -> Optional[pd.DataFrame]:
    """
    Loads seed channels from a specified source.
    
    This is the main function Uday will call.

    Args:
        source: The file path (for CSV) or URL (for Google Sheet).
        source_type: The type of source, either 'csv' or 'google_sheet'. 
                     Defaults to 'csv'.

    Returns:
        A pandas DataFrame containing 'Channel_Name' and 'Channel_URL' columns,
        or None if loading fails.
    """
    print(f"Attempting to load seed channels from source: {source} (type: {source_type})")
    
    if source_type == 'csv':
        return _load_from_csv(source)
    elif source_type == 'google_sheet':
        return _load_from_google_sheet(source)
    else:
        print(f"Error: Invalid source_type '{source_type}'. Must be 'csv' or 'google_sheet'.")
        return None