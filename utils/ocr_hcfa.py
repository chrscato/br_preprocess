#!/usr/bin/env python3
"""
ocr_hcfa_s3.py

Fetches HCFA PDFs from S3, runs OCR via Google Vision,
writes extracted text back to S3, archives processed PDFs,
and logs any errors.
"""
import os
import tempfile
from dotenv import load_dotenv
import boto3
from google.cloud import vision
from google.cloud.vision_v1 import types

# Load environment variables: AWS_*, S3_BUCKET,
# GOOGLE_APPLICATION_CREDENTIALS should be set in .env
load_dotenv()

# Initialize S3 client via boto3 in s3_utils or directly
from utils.s3_utils import list_objects, download, upload, move

# Initialize Vision API client
vision_client = vision.ImageAnnotatorClient()

# S3 prefixes
INPUT_PREFIX = os.getenv('OCR_INPUT_PREFIX', 'data/hcfa_pdf/')
OUTPUT_PREFIX = os.getenv('OCR_OUTPUT_PREFIX', 'data/hcfa_txt/')
ARCHIVE_PREFIX = os.getenv('OCR_ARCHIVE_PREFIX', 'data/hcfa_pdf/archived/')
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
    """Iterate all PDFs in S3 input prefix, OCR, upload text, and archive."""
    print(f"Starting OCR run against bucket: {S3_BUCKET} (prefix: {INPUT_PREFIX})")
    pdf_keys = [key for key in list_objects(INPUT_PREFIX) if key.lower().endswith('.pdf')]

    for key in pdf_keys:
        print(f"→ Processing s3://{S3_BUCKET}/{key}")
        local_pdf = download(key, os.path.join(tempfile.gettempdir(), os.path.basename(key)))
        local_txt = None
        try:
            # Perform OCR
            extracted = ocr_pdf_with_vision(local_pdf)
            # Write text locally
            local_txt = tempfile.mktemp(suffix='.txt')
            with open(local_txt, 'w', encoding='utf-8') as f:
                f.write(extracted)

            # Upload text to S3
            base = os.path.splitext(os.path.basename(key))[0]
            s3_txt_key = f"{OUTPUT_PREFIX}{base}.txt"
            upload(local_txt, s3_txt_key)
            print(f"✔ Uploaded text to s3://{S3_BUCKET}/{s3_txt_key}")

            # Archive original PDF
            archived_key = key.replace(INPUT_PREFIX, ARCHIVE_PREFIX)
            move(key, archived_key)
            print(f"✔ Archived PDF to s3://{S3_BUCKET}/{archived_key}\n")
        except Exception as e:
            err = f"❌ Error OCR {key}: {e}"
            print(err)
            # Write error to temp and upload
            log_local = tempfile.mktemp(suffix='.log')
            with open(log_local, 'a', encoding='utf-8') as logf:
                logf.write(err + '\n')
            upload(log_local, LOG_PREFIX)
            os.remove(log_local)
        finally:
            if os.path.exists(local_pdf):
                os.remove(local_pdf)
            if local_txt and os.path.exists(local_txt):
                os.remove(local_txt)

    print("OCR processing complete.")


if __name__ == '__main__':
    process_ocr_s3()
