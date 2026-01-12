# Human vs LLM Text

A dataset creation tool for generating a collection of human-written and AI-generated abstracts from machine learning research papers.

## Overview

This project creates a dataset by fetching machine learning papers from arXiv and generating corresponding AI abstracts using OpenAI's API. The dataset can be used for comparing human-written vs LLM-generated text.

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`

```bash
cp .env.example .env
```

Edit .env with your configuration

## Usage

### Step 1: Fetch Papers from arXiv

Fetch machine learning papers from arXiv:

```bash
python arxiv.py n [--batch-size BATCH_SIZE]
```

Where `n` is the number of papers you want to fetch (required).

**Options:**

- `--batch-size`: Number of papers to collect before saving to CSV (default: 50). Useful for large fetches to prevent data loss.

Examples:

```bash
python arxiv.py 50              # Fetches 50 papers (saves every 50)
python arxiv.py 100             # Fetches 100 papers (saves every 50)
python arxiv.py 2000            # Fetches 2000 papers (saves every 50)
python arxiv.py 2000 --batch-size 100  # Fetches 2000 papers (saves every 100)
```

**Note:** The script saves papers incrementally in batches to prevent data loss. Progress is saved to the CSV file periodically, so if the script is interrupted, you won't lose all your progress. For large fetches (2000+ papers), this is especially important.

This creates `arxiv_strict_ml_[n].csv` (e.g., `arxiv_strict_ml_100.csv`, `arxiv_strict_ml_50.csv`) with paper titles and human-written abstracts.

### Step 2: Generate AI Abstracts

Generate AI abstracts for the papers. Each abstract is generated from a separate API call, then merged into a single CSV cell.

```bash
python abstract_gen.py INPUT_CSV [OPTIONS]
```

**Arguments:**

- `INPUT_CSV`: Input CSV file with paper titles (required)
- `-n, --num-abstracts`: Number of abstracts to generate per paper (required)

**Options:**

- `-o, --output`: Output CSV filename (default: input filename with `_ai_abstracts` suffix)
- `--batch-size`: Number of papers to process before saving (default: 10). Useful for large batches to prevent data loss.

**Examples:**

```bash
# Generate 2 abstracts per paper
python abstract_gen.py arxiv_strict_ml_100.csv -n 2

# Generate 3 abstracts per paper
python abstract_gen.py arxiv_strict_ml_100.csv -n 3

# Custom output filename
python abstract_gen.py arxiv_strict_ml_2000.csv -n 2 -o my_abstracts.csv

# Generate 5 abstracts per paper, save every 20 papers
python abstract_gen.py arxiv_strict_ml_2000.csv -n 5 --batch-size 20
```

**Note:** Each abstract is generated from a separate API response. All abstracts for a paper are combined into one CSV cell. The script saves incrementally in batches to prevent data loss.

## Output

### Step 1 Output (`arxiv_strict_ml_[n].csv`)

Contains:

- Paper IDs, categories, links
- Paper titles
- Human-written abstracts

The filename reflects the number of papers fetched (e.g., `arxiv_strict_ml_50.csv` for 50 papers, `arxiv_strict_ml_2000.csv` for 2000 papers).

### Step 2 Output (`[input]_ai_abstracts.csv`)

Contains:

- Paper titles
- AI-generated abstracts (number specified by `-n/--num-abstracts`, each from a separate API call, merged into one cell)

## Files

- `arxiv.py` - Fetches ML papers from arXiv with strict filtering
- `abstract_gen.py` - Generates AI abstracts using OpenAI API (one abstract per API call, merged into CSV)
- `requirements.txt` - Python dependencies
