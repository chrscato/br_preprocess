#!/usr/bin/env python3
"""
pdf_preview.py

Generates preview images from PDF files by converting the first page to an image
and creating three cropped sections (header, service lines, footer).
Uses PyMuPDF (fitz) for PDF processing without system dependencies.
"""
import os
import sys
import logging
import tempfile
import fitz  # PyMuPDF
import boto3
from PIL import Image
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Load environment variables from .env
load_dotenv(project_root / '.env')

# Import S3 helper functions
from utils.s3_utils import list_objects, download, upload, move

def generate_pdf_previews(pdf_filename: str):
    """
    Generate preview images from a PDF file stored in S3.
    
    Args:
        pdf_filename: Name of the PDF file in S3
    """
    logger = logging.getLogger("PDF Preview")
    s3_client = boto3.client('s3')
    bucket = os.getenv('S3_BUCKET', 'bill-review-prod')
    source_prefix = 'data/hcfa_pdf/'
    preview_prefix = 'data/hcfa_pdf/preview/'
    
    pdf_document = None
    temp_dir = None
    
    try:
        # Create temp directory for working files
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Download PDF from S3
        pdf_path = temp_path / pdf_filename
        s3_client.download_file(
            bucket,
            f"{source_prefix}{pdf_filename}",
            str(pdf_path)
        )
        logger.info(f"Downloaded {pdf_filename} from S3")
        
        # Open PDF and convert first page to image
        pdf_document = fitz.open(str(pdf_path))
        first_page = pdf_document[0]
        
        # Convert to high-quality image (300 DPI)
        zoom = 300 / 72  # zoom factor to achieve 300 DPI
        mat = fitz.Matrix(zoom, zoom)  # zoom matrix
        pix = first_page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image directly from pixmap bytes
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width, height = image.size
        
        # Calculate crop dimensions
        header_height = int(height * 0.25)
        service_lines_height = int(height * 0.40)
        footer_start = int(height * 0.75)
        
        # Crop sections
        header = image.crop((0, 0, width, header_height))
        service_lines = image.crop((0, header_height, width, header_height + service_lines_height))
        footer = image.crop((0, footer_start, width, height))
        
        # Get base filename without extension
        base_filename = Path(pdf_filename).stem
        
        # Save and upload each section
        sections = {
            'header.png': header,
            'service_lines.png': service_lines,
            'footer.png': footer
        }
        
        for filename, img in sections.items():
            # Save image to temp file
            temp_image_path = temp_path / filename
            img.save(temp_image_path, 'PNG')
            
            # Upload to S3 in preview folder with PDF name as prefix
            s3_key = f"{preview_prefix}{base_filename}/{filename}"
            s3_client.upload_file(
                str(temp_image_path),
                bucket,
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            logger.info(f"Uploaded preview {s3_key}")
            
    except Exception as e:
        logger.error(f"Error generating previews for {pdf_filename}: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up resources
        if pdf_document:
            pdf_document.close()
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {str(e)}")

def process_previews_s3():
    """Process all PDFs in the source directory that don't have previews."""
    logger = logging.getLogger("PDF Preview")
    s3_client = boto3.client('s3')
    bucket = os.getenv('S3_BUCKET', 'bill-review-prod')
    source_prefix = 'data/hcfa_pdf/'
    preview_prefix = 'data/hcfa_pdf/preview/'
    
    try:
        # List all PDFs in source directory
        pdf_files = [obj['Key'].split('/')[-1] for obj in s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=source_prefix
        ).get('Contents', []) if obj['Key'].lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.info("No PDFs found to process")
            return
        
        logger.info(f"Found {len(pdf_files)} PDFs to process")
        
        # Process each PDF that doesn't already have previews
        for pdf_file in pdf_files:
            try:
                base_name = Path(pdf_file).stem
                preview_path = f"{preview_prefix}{base_name}/"
                
                # Check if previews already exist
                try:
                    s3_client.head_object(Bucket=bucket, Key=f"{preview_path}header.png")
                    logger.debug(f"Previews already exist for {pdf_file}")
                    continue
                except:
                    logger.info(f"Generating previews for {pdf_file}")
                    generate_pdf_previews(pdf_file)
            except Exception as e:
                logger.error(f"Error in preview processing: {str(e)}", exc_info=True)
                continue
        
        logger.info("Preview generation complete")
        
    except Exception as e:
        logger.error(f"Error in preview processing: {str(e)}", exc_info=True)
        raise

# Alias for backward compatibility
process_pdf_previews = process_previews_s3

if __name__ == '__main__':
    # Setup basic logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    process_previews_s3() 