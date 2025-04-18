#!/usr/bin/env python3
"""
llm_hcfa.py

Processes OCR text output through LLM to extract structured data.
"""
import os
import sys
import logging
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables from the root .env file
load_dotenv(PROJECT_ROOT / '.env')

# Import S3 helper functions - handle both direct script execution and module import
try:
    from utils.s3_utils import list_objects, download, upload, move
except ImportError:
    # If running directly from utils directory
    from s3_utils import list_objects, download, upload, move

# S3 prefixes (override in .env if needed)
INPUT_PREFIX = os.getenv('LLM_INPUT_PREFIX', 'data/hcfa_txt/')
OUTPUT_PREFIX = os.getenv('LLM_OUTPUT_PREFIX', 'data/hcfa_json/')
ARCHIVE_PREFIX = os.getenv('LLM_ARCHIVE_PREFIX', 'data/hcfa_txt/archived/')
LOG_PREFIX = os.getenv('LLM_LOG_PREFIX', 'logs/extract_errors.log')
S3_BUCKET = os.getenv('S3_BUCKET')

# Load prompt from project root
PROMPT_PATH = PROJECT_ROOT / 'preprocess' / 'utils' / 'gpt41_prompt.txt'

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Helper to clean charge strings
def clean_charge(charge: str) -> str:
    try:
        if not charge.startswith("$"):
            return charge
        amount = float(charge.replace("$", "").replace(",", ""))
        if amount >= 10000:
            for divisor in (100, 10):
                candidate = amount / divisor
                if 10 <= candidate <= 5000:
                    return f"${candidate:.2f}"
        return f"${amount:.2f}"
    except:
        return charge


def fix_all_charges(data: dict) -> dict:
    if 'service_lines' in data:
        for line in data['service_lines']:
            if 'charge_amount' in line:
                line['charge_amount'] = clean_charge(line['charge_amount'])
    if 'billing_info' in data and 'total_charge' in data['billing_info']:
        data['billing_info']['total_charge'] = clean_charge(data['billing_info']['total_charge'])
    return data


def extract_data_via_llm(prompt_text: str, ocr_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are an AI assistant that extracts structured data from CMS-1500 medical claim forms."},
        {"role": "user", "content": prompt_text + "\n---\n" + ocr_text}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
        max_tokens=1500
    )
    return response.choices[0].message.content


def clean_gpt_output(raw: str) -> str:
    txt = raw.strip()
    if txt.startswith("```json") and txt.endswith("```"):
        txt = txt[len("```json"): -len("```")]
    return txt.strip()


def process_llm_s3(limit=None):
    print(f"Starting LLM extraction run against bucket: {S3_BUCKET} (prefix: {INPUT_PREFIX})")
    # Load prompt template
    with open(PROMPT_PATH, 'r', encoding='utf-8') as pf:
        prompt = pf.read()

    txt_keys = [k for k in list_objects(INPUT_PREFIX) if k.lower().endswith('.txt')]
    if limit:
        txt_keys = txt_keys[:int(limit)]

    for key in txt_keys:
        print(f"→ Processing s3://{S3_BUCKET}/{key}")
        local_txt = download(key, os.path.join(tempfile.gettempdir(), os.path.basename(key)))
        json_local = None
        try:
            ocr_text = open(local_txt, 'r', encoding='utf-8').read()
            raw_output = extract_data_via_llm(prompt, ocr_text)
            cleaned = clean_gpt_output(raw_output)

            if not cleaned.startswith('{'):
                raise ValueError('LLM output not JSON')

            parsed = json.loads(cleaned)
            parsed = fix_all_charges(parsed)

            # Print JSON structure for debugging
            print("\nJSON Structure:")
            print(json.dumps(parsed, indent=2))
            print("\nChecking required fields:")
            print(f"patient_info.patient_name: {parsed.get('patient_info', {}).get('patient_name', 'MISSING')}")
            service_lines = parsed.get('service_lines', [])
            print(f"service_lines[0].date_of_service: {service_lines[0].get('date_of_service', 'MISSING') if service_lines else 'NO SERVICE LINES'}")

            # Write JSON locally
            json_local = tempfile.mktemp(suffix='.json')
            with open(json_local, 'w', encoding='utf-8') as jf:
                json.dump(parsed, jf, indent=4)

            # Upload JSON to S3
            base = os.path.splitext(os.path.basename(key))[0]
            s3_json_key = f"{OUTPUT_PREFIX}{base}.json"
            upload(json_local, s3_json_key)
            print(f"✔ Uploaded JSON to s3://{S3_BUCKET}/{s3_json_key}")

            # Archive original text
            archived_key = key.replace(INPUT_PREFIX, ARCHIVE_PREFIX)
            move(key, archived_key)
            print(f"✔ Archived text to s3://{S3_BUCKET}/{archived_key}\n")

        except Exception as e:
            err = f"❌ Extraction error {key}: {e}"
            print(err)
            log_local = tempfile.mktemp(suffix='.log')
            with open(log_local, 'a', encoding='utf-8') as logf:
                logf.write(err + '\n')
            upload(log_local, LOG_PREFIX)
            os.remove(log_local)

        finally:
            if os.path.exists(local_txt):
                os.remove(local_txt)
            if json_local and os.path.exists(json_local):
                os.remove(json_local)

    print("LLM extraction complete.")


if __name__ == '__main__':
    process_llm_s3()
