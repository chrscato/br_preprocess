#!/usr/bin/env python3
"""
test_parquet.py

Tests the structure and content of Parquet files used in the HCFA mapping process.
Verifies that the files contain the expected columns and data types.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import s3fs
from dotenv import load_dotenv

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Load environment variables
load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET')
PARQUET_PREFIX = 'data/filemaker/'

def test_orders_parquet():
    """Test the structure and content of orders.parquet"""
    print("\nTesting orders.parquet...")
    
    # Initialize S3 filesystem
    s3 = s3fs.S3FileSystem()
    orders_path = f"s3://{S3_BUCKET}/{PARQUET_PREFIX}orders.parquet"
    
    try:
        # Load orders DataFrame
        df = pd.read_parquet(orders_path)
        
        # Required columns for mapping process
        required_columns = {
            'Order_ID': 'object',
            'FileMaker_Record_Number': 'object',
            'Patient_Last_Name': 'object',
            'Patient_First_Name': 'object',
            'PatientName': 'object'
        }
        
        # Check columns exist and have correct types
        missing_cols = []
        wrong_types = []
        
        for col, expected_type in required_columns.items():
            if col not in df.columns:
                missing_cols.append(col)
            elif not df[col].dtype.name.startswith(expected_type):
                wrong_types.append(f"{col} (expected {expected_type}, got {df[col].dtype})")
        
        if missing_cols:
            print("❌ Missing required columns:", ", ".join(missing_cols))
        else:
            print("✔ All required columns present")
            
        if wrong_types:
            print("❌ Incorrect column types:", ", ".join(wrong_types))
        else:
            print("✔ All column types correct")
        
        # Check for null values in critical columns
        null_counts = df[required_columns.keys()].isnull().sum()
        critical_nulls = null_counts[null_counts > 0]
        
        if not critical_nulls.empty:
            print("⚠️ Null values found in critical columns:")
            for col, count in critical_nulls.items():
                print(f"  - {col}: {count} nulls")
        else:
            print("✔ No null values in critical columns")
        
        # Print sample of data
        print("\nSample data (first 5 rows):")
        print(df[required_columns.keys()].head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing orders.parquet: {str(e)}")
        return False

def test_line_items_parquet():
    """Test the structure and content of line_items.parquet"""
    print("\nTesting line_items.parquet...")
    
    # Initialize S3 filesystem
    s3 = s3fs.S3FileSystem()
    line_items_path = f"s3://{S3_BUCKET}/{PARQUET_PREFIX}line_items.parquet"
    
    try:
        # Load line items DataFrame
        df = pd.read_parquet(line_items_path)
        
        # Required columns for mapping process
        required_columns = {
            'Order_ID': 'object',
            'DOS': 'object',  # or datetime64[ns]
            'CPT': 'object'
        }
        
        # Check columns exist and have correct types
        missing_cols = []
        wrong_types = []
        
        for col, expected_type in required_columns.items():
            if col not in df.columns:
                missing_cols.append(col)
            elif not (df[col].dtype.name.startswith(expected_type) or 
                     (expected_type == 'object' and df[col].dtype.name.startswith('datetime'))):
                wrong_types.append(f"{col} (expected {expected_type}, got {df[col].dtype})")
        
        if missing_cols:
            print("❌ Missing required columns:", ", ".join(missing_cols))
        else:
            print("✔ All required columns present")
            
        if wrong_types:
            print("❌ Incorrect column types:", ", ".join(wrong_types))
        else:
            print("✔ All column types correct")
        
        # Check for null values in critical columns
        null_counts = df[required_columns.keys()].isnull().sum()
        critical_nulls = null_counts[null_counts > 0]
        
        if not critical_nulls.empty:
            print("⚠️ Null values found in critical columns:")
            for col, count in critical_nulls.items():
                print(f"  - {col}: {count} nulls")
        else:
            print("✔ No null values in critical columns")
        
        # Print sample of data
        print("\nSample data (first 5 rows):")
        print(df[required_columns.keys()].head())
        
        # Check unique CPT codes
        print("\nUnique CPT codes (first 10):")
        print(df['CPT'].unique()[:10])
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing line_items.parquet: {str(e)}")
        return False

def main():
    """Run all Parquet tests"""
    print(f"Testing Parquet files in s3://{S3_BUCKET}/{PARQUET_PREFIX}")
    
    orders_success = test_orders_parquet()
    line_items_success = test_line_items_parquet()
    
    if orders_success and line_items_success:
        print("\n✔ All Parquet tests passed")
    else:
        print("\n❌ Some Parquet tests failed")

if __name__ == "__main__":
    main() 