"""
main.py

Runs the complete HCFA preprocessing pipeline:
1. Split batch PDFs into individual files (split_hcfa_batch)
2. Generate PDF previews (pdf_preview)
3. Perform OCR on PDFs (ocr_hcfa)
4. Extract JSON from OCR text (llm_hcfa)
5. Validate and clean JSON files (validatejson)
6. Map JSON to FileMaker records (map_to_fm)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

# Load environment variables
load_dotenv()

# Import pipeline modules
from preprocess.utils import split_hcfa_batch
from preprocess.utils import pdf_preview
from preprocess.utils import ocr_hcfa
from preprocess.utils import llm_hcfa
from preprocess.utils import validatejson
from preprocess.utils.map_to_fm import process_mapping_s3

# Configure output encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

def run_pipeline():
    """Run the complete preprocessing pipeline"""
    print("\n=== Starting HCFA Preprocessing Pipeline ===\n")
    
    print("Step 1: Splitting batch PDFs...")
    try:
        split_hcfa_batch.process_batch_s3()
        print("[SUCCESS] Batch splitting complete\n")
    except Exception as e:
        if "No PDF batches found" in str(e):
            print("[INFO] No batch PDFs to process, continuing...\n")
        else:
            print(f"[ERROR] Error during batch splitting: {str(e)}\n")
        
    print("Step 2: Generating PDF previews...")
    try:
        pdf_preview.process_previews_s3()
        print("[SUCCESS] Preview generation complete\n")
    except Exception as e:
        if "No PDFs found" in str(e):
            print("[INFO] No PDFs to preview, continuing...\n")
        else:
            print(f"[ERROR] Error during preview generation: {str(e)}\n")
    
    print("Step 3: Performing OCR on PDFs...")
    try:
        ocr_hcfa.process_ocr_s3()
        print("[SUCCESS] OCR processing complete\n")
    except Exception as e:
        if "No PDFs found" in str(e):
            print("[INFO] No PDFs for OCR, continuing...\n")
        else:
            print(f"[ERROR] Error during OCR processing: {str(e)}\n")
    
    print("Step 4: Extracting JSON from OCR text...")
    try:
        llm_hcfa.process_llm_s3()
        print("[SUCCESS] JSON extraction complete\n")
    except Exception as e:
        if "No OCR files found" in str(e):
            print("[INFO] No OCR files for JSON extraction, continuing...\n")
        else:
            print(f"[ERROR] Error during JSON extraction: {str(e)}\n")
    
    print("Step 5: Validating and cleaning JSON files...")
    try:
        validatejson.process_validation_s3()
        print("[SUCCESS] JSON validation complete\n")
    except Exception as e:
        if "No JSON files found" in str(e):
            print("[INFO] No JSON files to validate, continuing...\n")
        else:
            print(f"[ERROR] Error during JSON validation: {str(e)}\n")
    
    print("Step 6: Mapping JSON to FileMaker records...")
    try:
        process_mapping_s3()
        print("[SUCCESS] FileMaker mapping complete\n")
    except Exception as e:
        if "No files found" in str(e):
            print("[INFO] No files to map to FileMaker, continuing...\n")
        else:
            print(f"[ERROR] Error during FileMaker mapping: {str(e)}\n")
    
    print("=== Pipeline Complete ===")
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
