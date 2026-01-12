#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import requests
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ================= CONFIG =================
MODEL = "gpt-4o-mini"
ENDPOINT = "https://api.openai.com/v1/chat/completions"

MIN_WC = 150
MAX_WC = 200
TARGET_WC = 170
SLEEP_BETWEEN_REQUESTS = 0.1

# ================= HELPERS =================
_word_re = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

def word_count(text: str) -> int:
    return len(_word_re.findall(text))

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def ensure_sentence_end(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text[-1] in ".!?" else (text + ".")

def trim_to_words(text: str, n: int) -> str:
    """Trim text to approximately n words, keeping punctuation intact."""
    text = text.strip()
    words = _word_re.findall(text)
    if len(words) <= n:
        return normalize_ws(ensure_sentence_end(text))

    # Tokenize and count words, stopping at n words
    count = 0
    out_chars: List[str] = []
    token_pattern = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|\s+|[^\w\s]")
    
    for m in token_pattern.finditer(text):
        token = m.group(0)
        if _word_re.fullmatch(token):
            count += 1
            if count > n:
                break
        out_chars.append(token)

    trimmed = "".join(out_chars).strip()
    return normalize_ws(ensure_sentence_end(trimmed))

def read_rows(path: str) -> List[Dict[str, str]]:
    """Read CSV file and return list of row dictionaries."""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "title" not in reader.fieldnames:
                raise RuntimeError(f"CSV must contain a 'title' column. Found: {reader.fieldnames}")
            return list(reader)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {path}")

def write_output(path: str, rows: List[Dict[str, str]], append: bool = False) -> None:
    """Write rows to CSV file, with optional append mode."""
    fieldnames = ["title", "ai_generated_abstract"]
    file_exists = os.path.exists(path) and append
    
    mode = "a" if file_exists else "w"
    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ================= OPENAI CALL =================
def call_openai(payload: Dict, api_key: str) -> Dict:
    """Call OpenAI API with exponential backoff for rate limiting."""
    backoff = 2
    max_retries = 10
    
    for attempt in range(max_retries):
        try:
            r = requests.post(
                ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )

            if r.status_code == 429:
                print(f"[429] Rate limited. Waiting {backoff}s...", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"API call failed after {max_retries} attempts: {e}") from e
            print(f"Request failed, retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    
    raise RuntimeError("Too many rate-limit retries.")

def extract_text(resp: Dict) -> str:
    """Extract text from OpenAI API response."""
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("No text returned from model.")
    return content

# ================= GENERATION =================
def generate_single_abstract(title: str, api_key: str) -> str:
    """Generate a single abstract for a given title."""
    instructions = (
        "Generate 1 abstract for the given title.\n"
        "Do not try to imitate the human writing style.\n"
        "The abstract must be written as one paragraph only.\n"
        "The abstract must be between 150 and 200 words.\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "abstract": {"type": "string"}
        },
        "required": ["abstract"],
        "additionalProperties": False,
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Title: {title}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "abstract",
                "strict": True,
                "schema": schema,
            }
        },
        "max_tokens": 450,
    }

    resp = call_openai(payload, api_key)
    data = json.loads(extract_text(resp))
    
    abstract = normalize_ws(data["abstract"])
    # Enforce max length locally
    if word_count(abstract) > MAX_WC:
        abstract = trim_to_words(abstract, TARGET_WC)
    else:
        abstract = normalize_ws(ensure_sentence_end(abstract))
    
    return abstract

def generate_abstracts(title: str, api_key: str, num_abstracts: int) -> List[str]:
    """Generate N abstracts for a given title, each from a separate API call."""
    abstracts = []
    for i in range(num_abstracts):
        abstract = generate_single_abstract(title, api_key)
        abstracts.append(abstract)
        # Sleep between requests to avoid rate limiting
        if i < num_abstracts - 1:  # Don't sleep after the last one
            time.sleep(SLEEP_BETWEEN_REQUESTS)
    return abstracts

def combine_same_cell(abstracts: List[str]) -> str:
    """Combine multiple abstracts into one cell with spaces between them."""
    normalized = [normalize_ws(ensure_sentence_end(a)) for a in abstracts]
    return " ".join(normalized)

# ================= MAIN =================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate AI abstracts for papers from CSV"
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV file with paper titles"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file (default: input filename with '_ai_abstracts' suffix)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of abstracts to generate before saving (default: 10)"
    )
    parser.add_argument(
        "-n", "--num-abstracts",
        type=int,
        required=True,
        help="Number of abstracts to generate per paper (required)"
    )
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    input_csv = args.input_csv
    if args.output:
        output_csv = args.output
    else:
        base = os.path.splitext(input_csv)[0]
        output_csv = f"{base}_ai_abstracts.csv"
    
    batch_size = args.batch_size
    
    input_rows = read_rows(input_csv)
    total = len(input_rows)
    if total == 0:
        print("No rows found in input CSV.", file=sys.stderr)
        return 1

    # Remove existing file if it exists to start fresh
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    collected: List[Dict[str, str]] = []
    total_saved = 0
    count = 0
    bar_length = 40
    
    print(f"Generating abstracts for {total} papers (saving every {batch_size} abstracts)...\n")
    
    for row in input_rows:
        title = (row.get("title") or "").strip()
        count += 1
        
        # Show progress
        percentage = (count / total) * 100
        filled = int(bar_length * count / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\rProgress: [{bar}] {count}/{total} ({percentage:.1f}%)", end="", flush=True)
        
        if not title:
            collected.append({"title": "", "ai_generated_abstract": "ERROR: missing title"})
        else:
            try:
                abstracts = generate_abstracts(title, api_key, args.num_abstracts)
                combined = combine_same_cell(abstracts)
                collected.append({"title": title, "ai_generated_abstract": combined})
            except Exception as e:
                collected.append({"title": title, "ai_generated_abstract": f"ERROR: {e}"})
        
        # Save incrementally when batch size is reached
        if len(collected) >= batch_size:
            write_output(output_csv, collected, append=True)
            total_saved += len(collected)
            print(f" | Saved: {total_saved}")
            collected = []
        
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    # Save any remaining abstracts
    if collected:
        write_output(output_csv, collected, append=True)
        total_saved += len(collected)
    
    # Final progress bar
    bar = "█" * bar_length
    print(f"\rProgress: [{bar}] {count}/{total} (100.0%) | Saved: {total_saved}")
    print(f"\nCompleted! Saved {total_saved} abstracts -> {output_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
