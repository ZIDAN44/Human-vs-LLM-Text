#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import aiohttp
import random
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ================= CONFIG =================
MODEL = "gpt-4o-mini"
ENDPOINT = "https://api.openai.com/v1/chat/completions"

MIN_WC = 150
MAX_WC = 200
TARGET_WC = 170
MAX_CONCURRENT_REQUESTS = 20  # Limit concurrent API calls

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
    # Use utf-8-sig for Excel compatibility and proper Unicode handling
    with open(path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ================= ASYNC OPENAI CALL =================
async def call_openai_async(
    session: aiohttp.ClientSession,
    payload: Dict,
    api_key: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    async with semaphore:
        backoff = 1.0
        max_backoff = 60.0
        max_retries = 10

        for attempt in range(max_retries):
            try:
                async with session.post(
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    # Handle rate limiting with server guidance if present
                    if resp.status == 429:
                        #print("429 headers:", dict(resp.headers), file=sys.stderr)
                        
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after is not None:
                            wait_time = float(retry_after)
                        else:
                            wait_time = min(backoff, max_backoff)

                        # Add a little jitter to avoid thundering herd
                        wait_time *= (0.8 + 0.4 * random.random())

                        print(f"[429] Rate limited. Waiting {wait_time:.1f}s...", file=sys.stderr)
                        await asyncio.sleep(wait_time)
                        backoff = min(backoff * 2, max_backoff)
                        continue

                    # Retry transient server errors
                    if resp.status in (500, 502, 503, 504):
                        wait_time = min(backoff, max_backoff) * (0.8 + 0.4 * random.random())
                        text = await resp.text()
                        print(f"[{resp.status}] Server error. Retrying in {wait_time:.1f}s... Body: {text[:200]}", file=sys.stderr)
                        await asyncio.sleep(wait_time)
                        backoff = min(backoff * 2, max_backoff)
                        continue

                    resp.raise_for_status()
                    return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API call failed after {max_retries} attempts: {e}") from e
                wait_time = min(backoff, max_backoff) * (0.8 + 0.4 * random.random())
                print(f"Request failed ({type(e).__name__}), retrying in {wait_time:.1f}s...", file=sys.stderr)
                await asyncio.sleep(wait_time)
                backoff = min(backoff * 2, max_backoff)

        raise RuntimeError("Too many retries.")

def extract_text(resp: Dict) -> str:
    """Extract text from OpenAI API response."""
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("No text returned from model.")
    return content

# ================= ASYNC GENERATION =================
async def generate_single_abstract_async(
    session: aiohttp.ClientSession,
    title: str,
    api_key: str,
    semaphore: asyncio.Semaphore
) -> str:
    """Generate a single abstract for a given title asynchronously."""
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

    resp = await call_openai_async(session, payload, api_key, semaphore)
    # Ensure proper UTF-8 handling when parsing JSON
    content = extract_text(resp)
    data = json.loads(content, strict=False)
    
    # Ensure abstract is properly decoded as Unicode string
    abstract = str(data["abstract"])
    abstract = normalize_ws(abstract)
    # Enforce max length locally
    if word_count(abstract) > MAX_WC:
        abstract = trim_to_words(abstract, TARGET_WC)
    else:
        abstract = normalize_ws(ensure_sentence_end(abstract))
    
    return abstract

async def generate_abstracts_async(
    session: aiohttp.ClientSession,
    title: str,
    api_key: str,
    num_abstracts: int,
    semaphore: asyncio.Semaphore
) -> List[str]:
    """Generate N abstracts for a given title concurrently."""
    tasks = [
        generate_single_abstract_async(session, title, api_key, semaphore)
        for _ in range(num_abstracts)
    ]
    return await asyncio.gather(*tasks)

def combine_same_cell(abstracts: List[str]) -> str:
    """Combine multiple abstracts into one cell with spaces between them."""
    normalized = [normalize_ws(ensure_sentence_end(a)) for a in abstracts]
    return " ".join(normalized)

# ================= ASYNC MAIN =================
async def process_paper(
    session: aiohttp.ClientSession,
    title: str,
    api_key: str,
    num_abstracts: int,
    semaphore: asyncio.Semaphore
) -> Dict[str, str]:
    """Process a single paper and return result dictionary."""
    if not title:
        return {"title": "", "ai_generated_abstract": "ERROR: missing title"}
    
    try:
        abstracts = await generate_abstracts_async(
            session, title, api_key, num_abstracts, semaphore
        )
        combined = combine_same_cell(abstracts)
        return {"title": title, "ai_generated_abstract": combined}
    except Exception as e:
        return {"title": title, "ai_generated_abstract": f"ERROR: {e}"}

async def process_papers_async(
    input_rows: List[Dict[str, str]],
    api_key: str,
    num_abstracts: int,
    output_csv: str,
    batch_size: int,
    max_concurrent: int
) -> None:
    """Process all papers asynchronously with batching."""
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(input_rows)
    collected: List[Dict[str, str]] = []
    total_saved = 0
    bar_length = 40
    
    async with aiohttp.ClientSession() as session:
        # Process papers in batches to manage memory and progress updates
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_rows = input_rows[batch_start:batch_end]
            
            # Create tasks for all papers in this batch
            tasks = [
                process_paper(session, (row.get("title") or "").strip(), api_key, num_abstracts, semaphore)
                for row in batch_rows
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            collected.extend(results)
            
            # Save results
            write_output(output_csv, collected, append=True)
            total_saved += len(collected)
            
            # Update progress
            count = batch_end
            percentage = (count / total) * 100
            filled = int(bar_length * count / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\rProgress: [{bar}] {count}/{total} ({percentage:.1f}%) | Saved: {total_saved}", end="", flush=True)
            
            collected = []

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
        default=50,
        help="Number of papers to process before saving (default: 50)"
    )
    parser.add_argument(
        "-n", "--num-abstracts",
        type=int,
        required=True,
        help="Number of abstracts to generate per paper (required)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help=f"Maximum concurrent API requests (default: {MAX_CONCURRENT_REQUESTS})"
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
    
    print(f"Generating abstracts for {total} papers (saving every {batch_size} papers)...")
    print(f"Using {args.max_concurrent} concurrent requests for faster processing.\n")
    
    start_time = time.time()
    
    try:
        asyncio.run(process_papers_async(
            input_rows, api_key, args.num_abstracts, output_csv, batch_size, args.max_concurrent
        ))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results saved.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        return 1
    
    elapsed = time.time() - start_time
    
    # Final progress bar
    bar = "█" * 40
    print(f"\rProgress: [{bar}] {total}/{total} (100.0%)")
    print(f"\nCompleted! Saved {total} abstracts -> {output_csv}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
