#!/usr/bin/env python3
"""
split_hcfa_batch.py

Splits multi-page HCFA batch PDFs stored in S3 into single-page PDFs,
uploads them back to S3 with a date-run folder and batch/page numbering,
and archives the original batch files.
"""
import os
import tempfile
from datetime import datetime
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import S3 helper functions
from utils.s3_utils import list_objects, download, upload, move

# S3 prefixes
INPUT_PREFIX = os.getenv('INPUT_PREFIX', 'data/batches/')
OUTPUT_PREFIX = os.getenv('OUTPUT_PREFIX', 'data/pdf/')
ARCHIVE_PREFIX = os.getenv('ARCHIVE_PREFIX', 'data/batches/archived/')


def split_and_upload(batch_key: str, batch_idx: int, run_date: str):
    """Download a batch PDF, split pages, upload splits, and archive original."""
    bucket = os.getenv('S3_BUCKET')
    print(f"→ Processing s3://{bucket}/{batch_key} (batch #{batch_idx})")

    # Download batch PDF to temp directory
    local_pdf = download(batch_key, os.path.join(tempfile.gettempdir(), os.path.basename(batch_key)))

    # Read and split PDF pages
    reader = PdfReader(local_pdf)
    for page_idx, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)

        # Write page to a temporary file
        local_out = tempfile.mktemp(suffix=".pdf")
        with open(local_out, "wb") as f:
            writer.write(f)

        # Construct a concise S3 key: date-run folder / batch_page
        s3_key = f"{OUTPUT_PREFIX}{run_date}/{batch_idx:02d}_{page_idx:03d}.pdf"
        upload(local_out, s3_key)
        print(f"✔ Uploaded {s3_key}")

        # Clean up local page file
        os.remove(local_out)

    # Clean up batch PDF
    os.remove(local_pdf)

    # Archive original in S3
    archived_key = batch_key.replace(INPUT_PREFIX, ARCHIVE_PREFIX)
    move(batch_key, archived_key)
    print(f"✔ Archived original to {archived_key}\n")


def main():
    # Use today's date as run identifier
    run_date = datetime.now().strftime("%Y%m%d")
    # List all batch PDFs in S3
    pdf_keys = [k for k in list_objects(INPUT_PREFIX) if k.lower().endswith('.pdf')]
    for idx, key in enumerate(pdf_keys, start=1):
        split_and_upload(key, idx, run_date)
    print("All batches processed.")


if __name__ == '__main__':
    main()
