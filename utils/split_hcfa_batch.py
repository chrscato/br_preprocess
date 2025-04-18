#!/usr/bin/env python3
"""
split_hcfa_batch.py

Splits multi-page HCFA batch PDFs stored in S3 into single-page PDFs,
uploads them back to S3 with datetime_batch_page naming format,
and archives the original batch files.
"""
import os
import sys
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Load environment variables from .env
load_dotenv(project_root / '.env')

# Import S3 helper functions
from utils.s3_utils import list_objects, download, upload, move

# S3 prefixes
INPUT_PREFIX = os.getenv('INPUT_PREFIX', 'data/batches/')
OUTPUT_PREFIX = os.getenv('OUTPUT_PREFIX', 'data/hcfa_pdf/')  # Simplified output path
ARCHIVE_PREFIX = os.getenv('ARCHIVE_PREFIX', 'data/batches/archived/')


def split_and_upload(batch_key: str, batch_idx: int, timestamp: str):
    """Download a batch PDF, split pages, upload splits, and archive original."""
    logger = logging.getLogger("Split HCFA")
    bucket = os.getenv('S3_BUCKET')
    logger.info(f"Processing s3://{bucket}/{batch_key} (batch #{batch_idx})")

    try:
        # Download batch PDF to temp directory
        temp_dir = Path(tempfile.gettempdir())
        local_pdf = temp_dir / Path(batch_key).name
        download(batch_key, str(local_pdf))

        # Read and split PDF pages
        reader = PdfReader(str(local_pdf))
        for page_idx, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)

            # Write page to a temporary file
            local_out = Path(tempfile.mktemp(suffix=".pdf"))
            with open(local_out, "wb") as f:
                writer.write(f)

            # Create filename with datetime_batch_page format
            output_filename = f"{timestamp}_{batch_idx:02d}_{page_idx:03d}.pdf"
            s3_key = f"{OUTPUT_PREFIX}{output_filename}"
            
            upload(str(local_out), s3_key)
            logger.info(f"Uploaded {s3_key}")

            # Clean up local page file
            local_out.unlink()

        # Clean up batch PDF
        local_pdf.unlink()

        # Archive original in timestamped subfolder
        archive_subfolder = f"batch_{timestamp}"
        archived_key = f"{ARCHIVE_PREFIX}{archive_subfolder}/{Path(batch_key).name}"
        move(batch_key, archived_key)
        logger.info(f"Archived original to {archived_key}")

    except Exception as e:
        logger.error(f"Error processing batch {batch_key}: {str(e)}", exc_info=True)
        raise


def process_batch_s3():
    """Process all batch PDFs in S3, splitting them into individual pages."""
    logger = logging.getLogger("Split HCFA")
    try:
        # Create timestamp for this run (used in filenames)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # List all files in the input directory
        all_keys = list_objects(INPUT_PREFIX)
        logger.info(f"Found {len(all_keys)} total files in {INPUT_PREFIX}")
        
        # Filter for PDFs only in the root batches directory (not in subdirectories)
        pdf_keys = [
            k for k in all_keys 
            if k.lower().endswith('.pdf') 
            and k.count('/') == 2  # Only files directly in data/batches/
            and not 'archived' in k.lower()  # Exclude anything from archived folders
        ]
        
        if not pdf_keys:
            logger.warning(f"No PDF batches found to process in {INPUT_PREFIX}")
            return
            
        logger.info(f"Found {len(pdf_keys)} PDF batches to process:")
        for key in pdf_keys:
            logger.info(f"  - {key}")
            
        for idx, key in enumerate(pdf_keys, start=1):
            split_and_upload(key, idx, timestamp)
        logger.info("All batches processed.")
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        raise


def main():
    # Setup basic logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    process_batch_s3()


if __name__ == '__main__':
    main()
