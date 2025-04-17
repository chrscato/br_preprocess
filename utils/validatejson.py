#!/usr/bin/env python3
"""
validatejson.py

Validates JSON files in S3 that were extracted from HCFA forms.
Moves valid files to a validated prefix and invalid ones to a review prefix.
"""
import os
import json
import re
import sys
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Load environment variables
load_dotenv()

# Import S3 helper functions
from utils.s3_utils import list_objects, download, upload, move

# S3 prefixes (override in .env if needed)
INPUT_PREFIX = os.getenv('VALIDATE_INPUT_PREFIX', 'data/hcfa_json/')
VALID_PREFIX = os.getenv('VALIDATE_VALID_PREFIX', 'data/hcfa_json/valid/')
INVALID_PREFIX = os.getenv('VALIDATE_INVALID_PREFIX', 'data/hcfa_json/invalid/')
LOG_PREFIX = os.getenv('VALIDATE_LOG_PREFIX', 'logs/validation_errors.log')
S3_BUCKET = os.getenv('S3_BUCKET')

# Precompiled regex patterns for critical validations
ZIP_PATTERN = re.compile(r"^\d{5}$")
CURRENCY_PATTERN = re.compile(r"^\$\d{1,5}(?:\.\d{2})?$")

def clean_name(name):
    """
    Clean a patient name by replacing Greek and other non-ASCII characters 
    with their Latin equivalents and removing any remaining invalid characters.
    """
    if not name:
        return name
    
    # Greek to Latin character mapping
    greek_to_latin = {
        '\u0391': 'A',  # Greek capital Alpha
        '\u0392': 'B',  # Greek capital Beta
        '\u0393': 'G',  # Greek capital Gamma
        '\u0394': 'D',  # Greek capital Delta
        '\u0395': 'E',  # Greek capital Epsilon
        '\u0396': 'Z',  # Greek capital Zeta
        '\u0397': 'H',  # Greek capital Eta
        '\u0398': 'TH', # Greek capital Theta
        '\u0399': 'I',  # Greek capital Iota
        '\u039A': 'K',  # Greek capital Kappa
        '\u039B': 'L',  # Greek capital Lambda
        '\u039C': 'M',  # Greek capital Mu
        '\u039D': 'N',  # Greek capital Nu (this is in your example)
        '\u039E': 'X',  # Greek capital Xi
        '\u039F': 'O',  # Greek capital Omicron
        '\u03A0': 'P',  # Greek capital Pi
        '\u03A1': 'R',  # Greek capital Rho
        '\u03A3': 'S',  # Greek capital Sigma
        '\u03A4': 'T',  # Greek capital Tau
        '\u03A5': 'Y',  # Greek capital Upsilon
        '\u03A6': 'F',  # Greek capital Phi
        '\u03A7': 'CH', # Greek capital Chi
        '\u03A8': 'PS', # Greek capital Psi
        '\u03A9': 'O',  # Greek capital Omega
        
        # Lowercase Greek letters
        '\u03B1': 'a',  # Greek small Alpha
        '\u03B2': 'b',  # Greek small Beta
        '\u03B3': 'g',  # Greek small Gamma
        '\u03B4': 'd',  # Greek small Delta
        '\u03B5': 'e',  # Greek small Epsilon
        '\u03B6': 'z',  # Greek small Zeta
        '\u03B7': 'h',  # Greek small Eta
        '\u03B8': 'th', # Greek small Theta
        '\u03B9': 'i',  # Greek small Iota
        '\u03BA': 'k',  # Greek small Kappa
        '\u03BB': 'l',  # Greek small Lambda
        '\u03BC': 'm',  # Greek small Mu
        '\u03BD': 'n',  # Greek small Nu
        '\u03BE': 'x',  # Greek small Xi
        '\u03BF': 'o',  # Greek small Omicron
        '\u03C0': 'p',  # Greek small Pi
        '\u03C1': 'r',  # Greek small Rho
        '\u03C3': 's',  # Greek small Sigma
        '\u03C4': 't',  # Greek small Tau
        '\u03C5': 'y',  # Greek small Upsilon
        '\u03C6': 'f',  # Greek small Phi
        '\u03C7': 'ch', # Greek small Chi
        '\u03C8': 'ps', # Greek small Psi
        '\u03C9': 'o',  # Greek small Omega
    }
    
    # First try standard normalization
    normalized = unicodedata.normalize('NFKD', name)
    
    # Replace any Greek characters that weren't properly normalized
    for greek, latin in greek_to_latin.items():
        normalized = normalized.replace(greek, latin)
    
    # Keep only ASCII characters, hyphens, commas, and spaces
    ascii_only = re.sub(r'[^\x00-\x7F\-,\s]', '', normalized)
    
    # Fix multiple spaces
    clean_name = re.sub(r'\s+', ' ', ascii_only).strip()
    
    return clean_name

def parse_date(date_str):
    """
    Try to parse a date string using formats MM/DD/YY and MM/DD/YYYY.
    Return the date in YYYY-MM-DD format if successful, otherwise None.
    """
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def validate_json(data):
    # Clean patient name if it exists
    if "patient_info" in data and "patient_name" in data["patient_info"]:
        data["patient_info"]["patient_name"] = clean_name(data["patient_info"]["patient_name"])

    required_fields = {
        "patient_info": ["patient_name", "patient_dob", "patient_zip"],
        "service_lines": ["date_of_service", "place_of_service", "cpt_code", "diagnosis_pointer", "charge_amount", "units"],
        "billing_info": ["billing_provider_name", "billing_provider_address", "billing_provider_tin", "billing_provider_npi", "total_charge", "patient_account_no"]
    }
    
    # Check required sections and fields
    for section, fields in required_fields.items():
        if section not in data:
            return False, f"Missing section: {section}"
        
        if section == "service_lines":
            if not isinstance(data[section], list) or not data[section]:
                return False, f"{section} should be a non-empty list"
            for service in data[section]:
                for field in fields:
                    if field not in service:
                        return False, f"Missing field: {field} in service_lines"
        else:
            for field in fields:
                if field not in data[section]:
                    return False, f"Missing field: {field} in {section}"
    
    # Validate only the critical formats
    if not ZIP_PATTERN.match(data["patient_info"]["patient_zip"]):
        return False, "Invalid ZIP format (expected 5 digits)"
    if not CURRENCY_PATTERN.match(data["billing_info"]["total_charge"]):
        return False, "Invalid total charge format (expected currency)"
    
    try:
        total_charge = float(data["billing_info"]["total_charge"].replace("$", ""))
    except ValueError:
        return False, "Invalid total charge value"
    
    try:
        sum_charges = sum(float(service["charge_amount"].replace("$", "")) for service in data["service_lines"])
    except ValueError:
        return False, "One of the service charge amounts is invalid"
    
    if abs(total_charge - sum_charges) > 5:
        return False, "Sum of line item charges does not match total charge (difference exceeds $5)"
    
    for service in data["service_lines"]:
        # Critical check for currency format on charge_amount remains.
        if not CURRENCY_PATTERN.match(service["charge_amount"]):
            return False, "Invalid charge amount format (expected currency)"
        
        # Validate and standardize date_of_service remains critical.
        dos = service["date_of_service"]
        date_range = dos.split(" - ")
        parsed_dates = []
        for date in date_range:
            parsed = parse_date(date)
            if parsed:
                parsed_dates.append(parsed)
        if not parsed_dates:
            return False, "Invalid Date of Service format (expected MM/DD/YY or MM/DD/YYYY)"
        service["date_of_service"] = parsed_dates[0]
    
    return True, "Valid JSON"

def process_validation_s3(limit=None):
    """Process JSON files from S3, validate them, and move to appropriate locations."""
    print(f"Starting validation run against bucket: {S3_BUCKET} (prefix: {INPUT_PREFIX})")
    
    # Get list of JSON files to process - only from root hcfa_json directory
    all_keys = list_objects(INPUT_PREFIX)
    json_keys = [k for k in all_keys if k.lower().endswith('.json') 
                and k.count('/') == 2  # Only process files in root hcfa_json directory
                and not any(x in k for x in ['valid', 'invalid', 'garbage', 'processed'])]  # Skip already processed files
    
    if limit:
        json_keys = json_keys[:int(limit)]

    for key in json_keys:
        print(f"→ Processing s3://{S3_BUCKET}/{key}")
        local_json = download(key, os.path.join(tempfile.gettempdir(), os.path.basename(key)))
        
        try:
            # Load and validate JSON
            with open(local_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Store original name for comparison
            original_name = data.get("patient_info", {}).get("patient_name", "N/A")
            
            is_valid, message = validate_json(data)
            
            # Write back cleaned/standardized data
            with open(local_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            if is_valid:
                # Move to valid directory
                dest_key = f"data/hcfa_json/valid/{os.path.basename(key)}"
                upload(local_json, dest_key)
                print(f"✔ Valid JSON moved to s3://{S3_BUCKET}/{dest_key}")
                
                # Delete original after successful upload
                move(key, dest_key)
                
                # Log name changes if any
                new_name = data.get("patient_info", {}).get("patient_name", "N/A")
                if new_name != original_name:
                    log_msg = f"Name standardized in {key}: {original_name} -> {new_name}"
                    log_local = tempfile.mktemp(suffix='.log')
                    with open(log_local, 'w', encoding='utf-8') as logf:
                        logf.write(log_msg + '\n')
                    upload(log_local, LOG_PREFIX)
                    os.remove(log_local)
            else:
                # Move invalid files to invalid directory
                dest_key = f"data/hcfa_json/invalid/{os.path.basename(key)}"
                move(key, dest_key)
                print(f"❌ Invalid JSON moved to s3://{S3_BUCKET}/{dest_key}")
                
                # Log validation error
                log_msg = f"Validation failed for {key}: {message}"
                log_local = tempfile.mktemp(suffix='.log')
                with open(log_local, 'w', encoding='utf-8') as logf:
                    logf.write(log_msg + '\n')
                upload(log_local, LOG_PREFIX)
                os.remove(log_local)
                
        except Exception as e:
            err = f"❌ Error processing {key}: {str(e)}"
            print(err)
            log_local = tempfile.mktemp(suffix='.log')
            with open(log_local, 'w', encoding='utf-8') as logf:
                logf.write(err + '\n')
            upload(log_local, LOG_PREFIX)
            os.remove(log_local)
            
        finally:
            # Clean up temporary files
            if os.path.exists(local_json):
                os.remove(local_json)

    print("Validation run complete.")

if __name__ == '__main__':
    process_validation_s3()