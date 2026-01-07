# Local Setup & Running Embedder Experiments

This guide walks you through setting up a Python virtual environment and running the embedder retrieval experiments on a local CPU.

## 1. Create a Virtual Environment

Create a new virtual environment in the project root:

```bash
python3 -m venv .thesis_venv
```

## 2. Activate the Virtual Environment

Activate the virtual environment:

```bash
source .thesis_venv/bin/activate
```

You should now see `(.thesis_venv)` at the beginning of your terminal prompt.

## 3. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## 4. Run Embedder Experiments (Local CPU)

Make the experiment script executable:

```bash
chmod +x embedders_experiments/run_retrieval.sh
```

Run the retrieval experiment:

```bash
./embedders_experiments/run_retrieval.sh
```

## 5. Expected Output

If everything is set up correctly, you should see output similar to the following:

```text
====================================================================
 Running experiment for embedder: minilm
====================================================================
2026-01-07 14:13:55,942 INFO embedders_experiments.experiments: Using embedder minilm (sentence-transformers/all-MiniLM-L6-v2)
model_name: sentence-transformers/all-MiniLM-L6-v2, embedder key minilm
...
```

This confirms that the embedder experiment is running successfully.

## 6. Cluster
You can run on GPU with s_gpu_retrieval.md or s_cpu_retrieval.md, but you will need to adjust to your username and cluster type:) this was run on LRC (https://ufal.mff.cuni.cz/lrc/index.php?title=Main_Page).

For everything to work you also need to get HUGGINGFACE_HUB_TOKEN, EINFRA_API_KEY and if you want to use OpenAi also OPENAI_API_KEY. The scripts look at .env in embedders_experiments folder