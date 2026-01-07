# OpenAI Cost Protection

This document explains the cost protection features added to prevent overspending when using OpenAI embedding APIs.

## Features

### 1. Automatic Cost Estimation
Before running any OpenAI embedding experiment, the system automatically:
- Scans your dataset files to count documents and queries
- Estimates token usage based on text length
- Calculates estimated cost using current OpenAI pricing
- Displays a detailed cost breakdown

### 2. Budget Limit Protection
- **Default budget limit**: $10 USD
- The script will **abort** if estimated cost exceeds the budget limit
- Prevents accidental expensive runs

### 3. Cost Warnings
- **High cost (>$1)**: Shows a warning message
- **Medium cost (>$0.10)**: Shows an informational note
- **Low cost (<$0.10)**: No warning (proceeds silently)

## Usage

### Setting a Custom Budget Limit

Set the `OPENAI_BUDGET_LIMIT` environment variable before running:

```bash
# Set a $5 budget limit
export OPENAI_BUDGET_LIMIT=5.0

# Run your experiment
bash embedders_experiments/run_retrieval.sh
```

### Example Output

When running with OpenAI models, you'll see output like:

```
INFO: Estimating cost for OpenAI embedding API...
INFO: Cost estimation:
INFO:   Model: text-embedding-3-small
INFO:   Document chunks: 50
INFO:   Estimated doc tokens: 16,667
INFO:   Queries: 50
INFO:   Estimated query tokens: 1,667
INFO:   Total estimated tokens: 18,334
INFO:   Estimated cost: $0.0004 USD
INFO: Estimated cost: $0.0004 USD (budget limit: $10.00 USD)
```

If cost exceeds budget:
```
======================================================================
⚠️  ESTIMATED COST $15.23 EXCEEDS BUDGET LIMIT $10.00!
   This operation would cost approximately $15.23 USD.
   Set OPENAI_BUDGET_LIMIT environment variable to override (current: $10.00).
   Aborting to prevent overspending.
======================================================================
```

## Pricing Information

Current OpenAI embedding model pricing (as of 2024):
- `text-embedding-3-small`: $0.02 per 1M tokens
- `text-embedding-3-large`: $0.13 per 1M tokens  
- `text-embedding-ada-002`: $0.10 per 1M tokens

## Cost Estimation Accuracy

The cost estimation uses approximate token counting (3 characters ≈ 1 token). For more accurate estimates, install `tiktoken`:

```bash
pip install tiktoken
```

Then the system will use exact token counting if available.

## Safety Features

1. **Automatic abort**: Script stops before making API calls if budget is exceeded
2. **Clear warnings**: High-cost operations show prominent warnings
3. **Detailed logging**: All cost estimates are logged for review
4. **Graceful failure**: If cost estimation fails, script warns but continues (use caution)

## Notes

- Cost estimation happens **before** any API calls are made
- The estimate is based on dataset scanning, so it's accurate for your specific data
- Actual costs may vary slightly due to:
  - Token counting approximations
  - API rate limits causing retries
  - Changes in OpenAI pricing

## Disabling Cost Protection

If you need to disable cost protection (not recommended), you can:
1. Set a very high budget limit: `export OPENAI_BUDGET_LIMIT=1000.0`
2. Or modify the code to skip the check (not recommended)

