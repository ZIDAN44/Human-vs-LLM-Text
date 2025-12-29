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
python arxiv.py
```

This creates `arxiv_strict_ml_100.csv` with paper titles and human-written abstracts.

### Step 2: Generate AI Abstracts

Generate AI abstracts for the papers:

```bash
python abstract_gen.py
```

This reads `arxiv_strict_ml_100.csv` and creates `arxiv_strict_ml_ai_abstracts.csv` with AI-generated abstracts paired with their titles.

## Output

The final dataset (`arxiv_strict_ml_ai_abstracts.csv`) contains:

- Paper titles
- AI-generated abstracts (two abstracts per paper, combined in one cell)

The original dataset (`arxiv_strict_ml_100.csv`) contains:

- Paper IDs, categories, links
- Paper titles
- Human-written abstracts

## Files

- `arxiv.py` - Fetches ML papers from arXiv with strict filtering
- `abstract_gen.py` - Generates AI abstracts using OpenAI API
- `requirements.txt` - Python dependencies
