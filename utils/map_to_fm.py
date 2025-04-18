#!/usr/bin/env python3
"""
map_to_fm.py

Maps HCFA JSON files to FileMaker records using fuzzy matching.
Sources from S3 valid folder and moves files to mapped or unmapped folders.
Uses Parquet files in S3 for matching instead of local SQLite database.
"""
import os
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Import S3 helper functions
from utils.s3_utils import list_objects, download, upload, move

# Load environment variables
load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET')

# S3 paths
VALID_PREFIX = 'data/hcfa_json/valid/'
MAPPED_PREFIX = 'data/hcfa_json/valid/mapped/'
UNMAPPED_PREFIX = 'data/hcfa_json/valid/unmapped/'
STAGING_PREFIX = 'data/hcfa_json/valid/mapped/staging/'
PARQUET_PREFIX = 'data/filemaker/'

def normalize_text(text):
    """
    Normalize text for comparison. Handles both formats:
    - "Last, First Middle" (HCFA format)
    - "First Middle Last" (FileMaker format)
    """
    if not text:
        return ""
    
    # Remove any non-alphanumeric characters except spaces and commas
    text = text.strip().upper()
    
    # Handle "Last, First Middle" format
    if "," in text:
        last, first_middle = text.split(",", 1)
        parts = [last.strip()] + first_middle.strip().split()
    else:
        # Handle "First Middle Last" format
        parts = text.split()
        if len(parts) > 1:  # Move last name to front if we have multiple parts
            parts = [parts[-1]] + parts[:-1]
    
    # Join all parts and remove any remaining non-alphanumeric chars
    text = "".join(parts)
    chars = [char for char in text if char.isalnum()]
    return ''.join(sorted(chars))  # Sort to handle middle name variations

def parse_date(date_str):
    """
    Parse date string into datetime object.
    Handles formats:
    - YYYY-MM-DD
    - MM/DD/YYYY
    - MM/DD/YY
    - MM/DD/YY - MM/DD/YY (takes first date)
    """
    if not date_str:
        return None
        
    # Handle date ranges by taking the first date
    if ' - ' in date_str:
        date_str = date_str.split(' - ')[0]
    
    date_formats = [
        "%Y-%m-%d",    # YYYY-MM-DD
        "%m/%d/%Y",    # MM/DD/YYYY
        "%m/%d/%y"     # MM/DD/YY - converts to 20YY for years 00-69, 19YY for 70-99
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def date_diff_days(date1, date2):
    """Calculate difference in days between two dates, handling None values."""
    if date1 is None or date2 is None:
        return float('inf')  # Return infinity if either date is None
    return abs((date1 - date2).days)

def load_orders_to_dataframe():
    """Load orders from S3 Parquet files into DataFrame."""
    print("Loading data from S3 Parquet files...")
    
    # Initialize S3 filesystem
    s3 = s3fs.S3FileSystem()
    
    # Read orders Parquet file
    orders_path = f"s3://{S3_BUCKET}/{PARQUET_PREFIX}orders.parquet"
    line_items_path = f"s3://{S3_BUCKET}/{PARQUET_PREFIX}line_items.parquet"
    
    print("Loading orders...")
    orders_df = pd.read_parquet(orders_path)
    print("Loading line items...")
    line_items_df = pd.read_parquet(line_items_path)
    
    # Clean and convert DOS dates in line items
    print("Processing DOS dates...")
    line_items_df['DOS'] = line_items_df['DOS'].apply(lambda x: parse_date(str(x)) if pd.notna(x) else None)
    
    # Group DOS by Order_ID, keeping only valid dates
    dos_groups = (line_items_df[line_items_df['DOS'].notna()]
                 .groupby('Order_ID')['DOS']
                 .agg(list)
                 .reset_index())
    dos_groups.columns = ['Order_ID', 'DOS_List']
    
    # Merge orders with DOS list
    df = pd.merge(orders_df, dos_groups, on='Order_ID', how='left')
    
    # Select required columns
    df = df[['Order_ID', 'FileMaker_Record_Number', 'Patient_Last_Name', 'Patient_First_Name', 'PatientName', 'DOS_List']]
    
    # Normalize name columns
    for col in ['Patient_Last_Name', 'Patient_First_Name', 'PatientName']:
        df[col] = df[col].apply(normalize_text)
    
    print(f"Loaded {len(df)} records from Parquet files")
    return df, line_items_df

def get_cpts_for_order(order_id, line_items_df):
    """Get CPT codes for an order from line items DataFrame."""
    cpts = line_items_df[line_items_df['Order_ID'] == order_id]['CPT'].dropna().unique()
    return {str(cpt).strip() for cpt in cpts}

def process_mapping_s3():
    """Process JSON files and find matches in FileMaker database."""
    df_orders, df_line_items = load_orders_to_dataframe()
    
    # Get list of JSON files from valid prefix
    print("Listing files in S3...")
    all_keys = list_objects(VALID_PREFIX)
    json_keys = [k for k in all_keys if k.lower().endswith('.json') 
                and k.count('/') == 3  # Only process files directly in valid/
                and not any(x in k for x in ['mapped', 'unmapped', 'staging'])]
    
    print(f"Found {len(json_keys)} files to process")
    processed_files = 0
    results = []
    
    for key in json_keys:
        filename = os.path.basename(key)
        print(f"\nProcessing: {filename}")
        
        try:
            # Download JSON file
            local_json = os.path.join(tempfile.gettempdir(), filename)
            download(key, local_json)
            
            with open(local_json, 'r') as f:
                json_data = json.load(f)
            
            original_name = json_data.get("patient_info", {}).get("patient_name", "")
            json_name = normalize_text(original_name)
            
            dos_list = []
            for entry in json_data.get("service_lines", []):
                dos = parse_date(entry.get("date_of_service", ""))
                if dos:
                    dos_list.append(dos)
            
            if not json_name or not dos_list:
                print(f"❌ Missing name or DOS: {filename}")
                new_key = f"{UNMAPPED_PREFIX}{filename}"
                move(key, new_key)
                continue
            
            candidate_matches = []
            for _, row in df_orders.iterrows():
                token_sort_score = fuzz.token_sort_ratio(json_name, row['PatientName'])
                token_set_score = fuzz.token_set_ratio(json_name, row['PatientName'])
                composite_score = (token_sort_score + token_set_score) / 2
                
                if composite_score >= 90:
                    db_dos_list = row.get('DOS_List', [])
                    if not db_dos_list:
                        continue
                    
                    for json_dos in dos_list:
                        for db_dos in db_dos_list:
                            if date_diff_days(json_dos, db_dos) <= 14:
                                candidate_matches.append({
                                    'composite_score': composite_score,
                                    'token_sort_score': token_sort_score,
                                    'token_set_score': token_set_score,
                                    'row': row
                                })
                                break
                        else:
                            continue
                        break
            
            best_match = None
            if len(candidate_matches) == 1:
                best_match = candidate_matches[0]
            elif len(candidate_matches) > 1:
                # Enhanced logic: sort candidates by CPT match score, then name score
                json_cpts = {line.get("cpt_code", "").strip() 
                           for line in json_data.get("service_lines", []) 
                           if line.get("cpt_code")}
                
                enriched_matches = []
                for match in candidate_matches:
                    order_id = match['row']['Order_ID']
                    db_cpts = get_cpts_for_order(order_id, df_line_items)
                    cpt_score = len(json_cpts & db_cpts)
                    enriched_matches.append((cpt_score, match['composite_score'], match))
                
                enriched_matches.sort(reverse=True, key=lambda x: (x[0], x[1]))
                best_match = enriched_matches[0][2]
            
            if best_match:
                # Add FileMaker info to JSON in a new mapping_info section
                json_data["mapping_info"] = {
                    "order_id": str(best_match['row']['Order_ID']),
                    "filemaker_number": str(best_match['row']['FileMaker_Record_Number']),
                    "mapping_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Write updated JSON back to file
                with open(local_json, 'w') as f:
                    json.dump(json_data, f, indent=4)
                
                # Upload to mapped folder
                new_key = f"{MAPPED_PREFIX}{filename}"
                upload(local_json, new_key)
                move(key, new_key)
                print(f"✔ Mapped: {filename} -> Order {best_match['row']['Order_ID']}")
                
                results.append({
                    'json_filename': filename,
                    'json_name_original': original_name,
                    'json_name_normalized': json_name,
                    'db_name_original': best_match['row']['PatientName'],
                    'db_order_id': best_match['row']['Order_ID'],
                    'db_filemaker_number': best_match['row']['FileMaker_Record_Number'],
                    'token_sort_score': best_match['token_sort_score'],
                    'token_set_score': best_match['token_set_score'],
                    'composite_score': best_match['composite_score']
                })
            else:
                print(f"❌ No match found: {filename}")
                new_key = f"{UNMAPPED_PREFIX}{filename}"
                move(key, new_key)
            
            # Clean up
            os.remove(local_json)
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"\nProcessed {processed_files} files")

if __name__ == "__main__":
    process_mapping_s3()
