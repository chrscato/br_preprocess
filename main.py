#!/usr/bin/env python3
"""
main.py

Main orchestration script for the HCFA preprocessing pipeline.
Coordinates the execution of batch uploading, splitting, OCR, LLM extraction,
validation, and FileMaker mapping steps.
"""
import os
import sys
import logging
import boto3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

# Import pipeline steps
from utils.data_janitor.upload_batch import upload_batch
from utils.split_hcfa_batch import main as split_batches
from utils.pdf_preview import generate_pdf_previews
from utils.ocr_hcfa import process_ocr_s3 as ocr_process
from utils.llm_hcfa import process_llm_s3 as llm_process
from utils.validatejson import process_validation_s3 as validate_json
from utils.map_to_fm import process_matches as map_filemaker

# Load environment variables
load_dotenv()

def check_s3_path_exists(bucket: str, prefix: str) -> bool:
    """Check if an S3 path exists."""
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return 'Contents' in response

def ensure_s3_path_exists(bucket: str, prefix: str):
    """Ensure S3 path exists by creating an empty marker object if needed."""
    if not check_s3_path_exists(bucket, prefix):
        s3_client = boto3.client('s3')
        s3_client.put_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/.keep")

def initialize_s3_folders():
    """Initialize required S3 folders at pipeline start."""
    logger = logging.getLogger("S3 Initialization")
    bucket = 'bill-review-prod'
    required_paths = [
        'data/hcfa_pdf/preview/',
        'data/hcfa_pdf/archived/'
    ]
    
    for path in required_paths:
        ensure_s3_path_exists(bucket, path)
        logger.info(f"Ensured S3 path exists: {path}")

def process_pdf_previews():
    """Generate previews for PDFs that don't already have them."""
    logger = logging.getLogger("PDF Previews")
    s3_client = boto3.client('s3')
    bucket = 'bill-review-prod'
    pdf_prefix = 'data/hcfa_pdf/archived/'
    preview_prefix = 'data/hcfa_pdf/preview/'
    
    # List all PDFs in archived folder
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pdf_files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=pdf_prefix):
            if 'Contents' in page:
                pdf_files.extend([obj['Key'] for obj in page['Contents'] 
                                if obj['Key'].lower().endswith('.pdf')])
        
        # Process each PDF that doesn't have previews
        for pdf_key in pdf_files:
            pdf_filename = os.path.basename(pdf_key)
            base_filename = os.path.splitext(pdf_filename)[0]
            preview_path = f"{preview_prefix}{base_filename}/"
            
            # Check if previews already exist
            if not check_s3_path_exists(bucket, preview_path):
                logger.info(f"Generating previews for {pdf_filename}")
                generate_pdf_previews(pdf_filename)
            else:
                logger.debug(f"Previews already exist for {pdf_filename}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing PDF previews: {str(e)}", exc_info=True)
        return False

# Configure logging
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/pipeline_{timestamp}.log")
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_pipeline_step(step_func, step_name):
    """Run a pipeline step with error handling."""
    logger = logging.getLogger(step_name)
    try:
        logger.info(f"Starting {step_name}")
        step_func()
        logger.info(f"Completed {step_name}")
        return True
    except Exception as e:
        logger.error(f"Error in {step_name}: {str(e)}", exc_info=True)
        return False

def main():
    """Main pipeline orchestration function."""
    logger = setup_logging()
    logger.info("Starting HCFA preprocessing pipeline")
    
    # Initialize S3 folders
    initialize_s3_folders()
    
    # Pipeline steps in order
    pipeline_steps = [
        (upload_batch, "Upload Batch"),
        (split_batches, "Split HCFA Batches"),
        (process_pdf_previews, "Generate PDF Previews"),
        (ocr_process, "OCR Processing"),
        (llm_process, "LLM Extraction"),
        (validate_json, "JSON Validation"),
        (map_filemaker, "FileMaker Mapping")
    ]
    
    # Track success/failure of each step
    results = []
    
    # Run pipeline steps
    for step_func, step_name in pipeline_steps:
        success = run_pipeline_step(step_func, step_name)
        results.append((step_name, success))
        
        # If a step fails, log but continue with next step
        if not success:
            logger.warning(f"{step_name} failed but continuing with pipeline")
    
    # Log final summary
    logger.info("\nPipeline Summary:")
    for step_name, success in results:
        status = "✔ Success" if success else "❌ Failed"
        logger.info(f"{status}: {step_name}")
    
    # Overall pipeline status
    if all(success for _, success in results):
        logger.info("Pipeline completed successfully")
        return 0
    else:
        logger.warning("Pipeline completed with some failures")
        return 1

if __name__ == "__main__":
    sys.exit(main())
