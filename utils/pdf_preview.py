#!/usr/bin/env python3
"""
pdf_preview.py

Generates preview images from PDF files by converting the first page to an image
and creating three cropped sections (header, service lines, footer).
Uses PyMuPDF (fitz) for PDF processing without system dependencies.
"""
import os
import tempfile
import fitz  # PyMuPDF
import boto3
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def generate_pdf_previews(pdf_filename: str):
    """
    Generate preview images from a PDF file stored in S3.
    
    Args:
        pdf_filename: Name of the PDF file in S3
    """
    s3_client = boto3.client('s3')
    bucket = 'bill-review-prod'
    source_prefix = 'data/hcfa_pdf/archived/'
    preview_prefix = 'data/hcfa_pdf/preview/'
    
    # Create temp directory for working files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download PDF from S3
        pdf_path = os.path.join(temp_dir, pdf_filename)
        s3_client.download_file(
            bucket,
            f"{source_prefix}{pdf_filename}",
            pdf_path
        )
        print(f"Downloaded {pdf_filename} from S3")
        
        # Open PDF and convert first page to image
        pdf_document = fitz.open(pdf_path)
        first_page = pdf_document[0]
        
        # Convert to high-quality image (300 DPI)
        pix = first_page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_data = pix.tobytes()
        
        # Convert to PIL Image
        image = Image.frombytes("RGB", [pix.width, pix.height], img_data)
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
        base_filename = os.path.splitext(pdf_filename)[0]
        preview_base_path = f"{preview_prefix}{base_filename}/"
        
        # Save and upload each section
        sections = {
            'header.png': header,
            'service_lines.png': service_lines,
            'footer.png': footer
        }
        
        for filename, img in sections.items():
            # Save image to temp file
            temp_image_path = os.path.join(temp_dir, filename)
            img.save(temp_image_path, 'PNG')
            
            # Upload to S3
            s3_key = f"{preview_base_path}{filename}"
            s3_client.upload_file(
                temp_image_path,
                bucket,
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            print(f"Uploaded {s3_key} to S3")
        
        # Clean up
        pdf_document.close()

if __name__ == '__main__':
    # Example usage
    generate_pdf_previews("example.pdf") 