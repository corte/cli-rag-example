# CLI RAG example

Minimialistic user interface to work with RAG (Retrieval Augmented Generation) using Ollama endpoints and langchain tools.

## Setup

After installing Ollama, pull the model:

```bash
$ ollama pull hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
```

Ps: If you need to use another model don't forget to change the variable `OLLAMA_MODEL` in `main.py`.

## Running without debug logs

```bash
$ python3 main.py
```

## Running with debug logs

```bash
$ LOG=DEBUG python3 main.py
```