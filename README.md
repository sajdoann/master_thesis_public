# Master Thesis: Embedding Models for Czech RAG

A research project evaluating embedding models for Retrieval-Augmented Generation (RAG) systems in Czech language.

## Overview

This repository contains experiments comparing various embedding models for information retrieval tasks, 
focusing on Czech language datasets including DareCzech-Test and cusotm created Wikipedia-NLP.

## Structure

```
├── embedders_experiments/    # Embedding model evaluation and retrieval experiments
├── pdf_extractor/            # PDF extraction tools and dataset creation
├── dareczech_scripts/                # DareCzech dataset processing (redacted)
└── generation_eval/          # Generation evaluation notebooks
```

## Quick Start

### Setup

```bash
python3 -m venv .thesis_venv
source .thesis_venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in `embedders_experiments/` with:

```
HUGGINGFACE_HUB_TOKEN=your_token
EINFRA_API_KEY=your_key
OPENAI_API_KEY=your_key  # optional
```

### Run Experiments
From the project root run:
```bash
chmod +x ./embedders_experiments/run_retrieval.sh
./embedders_experiments/run_retrieval.sh
```
This should run the Default OpenWebUI embedder (all-MiniLM-L6-v2) on wikipedia_nlp_small dataset:)

For cluster execution, see the SLURM scripts (`s_cpu_retrieval.slurm`, `s_gpu_retrieval.slurm`). They were executed on [LRC](https://ufal.mff.cuni.cz/lrc/index.php?title=Main_Page).

## Documentation

- [Embedder Experiments](embedders_experiments/README.md) - Detailed setup and usage
- [PDF Extractor](pdf_extractor/README.md) - PDF extraction and dataset creation
- [Libraries Overview](embedders_experiments/LIBRARIES_OVERVIEW.md) - Technical documentation of embedder experiment

## Results

Experimental results are stored in `embedders_experiments/results/` including 
comparisons of various embedding models (BM25, sentence-transformers, OpenAI embeddings, etc.).

## Disclaimers
Note the scripts were created with help of Gemini and OpenAI models. 

The DareCzech dataset has licensing constraints and therefore it is redacted from this public repository. Nevertheless the experiments are still runnable on Wikipedia nlp dataset:)