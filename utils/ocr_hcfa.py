#!/usr/bin/env python3
"""
ocr_hcfa_s3.py

Fetches HCFA PDFs from S3, runs OCR via Google Vision,
writes extracted text back to S3, archives processed PDFs,
and logs any errors.
"""
import os
import sys
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import boto3
from google.cloud import vision
from google.cloud.vision_v1 import types

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables from the root .env file
load_dotenv(PROJECT_ROOT / '.env')

# Set credentials path relative to project root
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(PROJECT_ROOT / 'googlecloud.json')

# Import S3 helper functions
from utils.s3_utils import list_objects, download, upload, move

# Initialize Vision API client
vision_client = vision.ImageAnnotatorClient()

# S3 prefixes
INPUT_PREFIX = os.getenv('OCR_INPUT_PREFIX', 'data/hcfa_pdf/')
ARCHIVE_PREFIX = os.getenv('OCR_ARCHIVE_PREFIX', 'data/hcfa_pdf/archived/')
OUTPUT_PREFIX = os.getenv('OCR_OUTPUT_PREFIX', 'data/hcfa_txt/')
LOG_PREFIX = os.getenv('OCR_LOG_PREFIX', 'logs/ocr_errors.log')
S3_BUCKET = os.getenv('S3_BUCKET')


def ocr_pdf_with_vision(local_pdf_path: str) -> str:
    """Run Google Vision Document Text Detection on the PDF file."""
    with open(local_pdf_path, 'rb') as f:
        content = f.read()

    input_config = types.InputConfig(
        content=content,
        mime_type='application/pdf'
    )
    feature = types.Feature(
        type_=types.Feature.Type.DOCUMENT_TEXT_DETECTION
    )
    request = types.AnnotateFileRequest(
        input_config=input_config,
        features=[feature]
    )

    response = vision_client.batch_annotate_files(requests=[request])
    texts = []
    for file_resp in response.responses:
        for page_resp in file_resp.responses:
            if page_resp.full_text_annotation:
                texts.append(page_resp.full_text_annotation.text)
    return "\n".join(texts)


def process_ocr_s3():
    """Process PDFs with OCR, save text output, and archive processed PDFs."""
    logger = logging.getLogger("OCR Processing")
    
    # List all PDFs in source folder (excluding archived)
    pdf_keys = [key for key in list_objects(INPUT_PREFIX) 
                if key.lower().endswith('.pdf') 
                and not key.startswith(ARCHIVE_PREFIX)]
    
    if not pdf_keys:
        logger.info("No PDFs found to process")
        return

    logger.info(f"Found {len(pdf_keys)} PDFs to process")
    
    for key in pdf_keys:
        pdf_name = Path(key).name
        logger.info(f"Processing {pdf_name}")
        
        try:
            # Create temp directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download PDF
                local_pdf = temp_path / pdf_name
                download(key, str(local_pdf))
                
                # Perform OCR
                extracted = ocr_pdf_with_vision(str(local_pdf))
                
                # Write text locally
                base_name = local_pdf.stem
                local_txt = temp_path / f"{base_name}.txt"
                with open(local_txt, 'w', encoding='utf-8') as f:
                    f.write(extracted)

                # Upload text to S3
                s3_txt_key = f"{OUTPUT_PREFIX}{base_name}.txt"
                upload(str(local_txt), s3_txt_key)
                logger.info(f"Saved OCR text: {s3_txt_key}")

                # Move processed PDF to archived folder
                archive_key = f"{ARCHIVE_PREFIX}{pdf_name}"
                move(key, archive_key)
                logger.info(f"Archived PDF to: {archive_key}")

        except Exception as e:
            logger.error(f"Error processing {pdf_name}: {str(e)}", exc_info=True)
            # Write error to log file
            log_local = temp_path / "error.log"
            with open(log_local, 'w', encoding='utf-8') as logf:
                logf.write(f"Error OCR {key}: {str(e)}\n")
            upload(str(log_local), LOG_PREFIX)

    logger.info("OCR processing complete")


if __name__ == '__main__':
    # Setup basic logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    process_ocr_s3()
