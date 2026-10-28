# EU Regulatory Intelligence

Research on the EU AI Act with verifiable official citations and transparent
comparison of retrieval strategies.

## Current status

Available now:

- A pinned Python environment and dependency lock.
- A compatibility script for imports, local models and optional OpenAI checks.
- Ruff lint/formatting and local pre-commit hooks.
- GitHub Actions configuration for lint and formatting; not yet run on GitHub.

Planned: document ingestion, dense/sparse/hybrid retrieval, evidence-grounded
answers with citation verification, and a Streamlit research interface.
Application functionality is not implemented yet.

## Planned architecture

FastAPI will provide the backend, with Streamlit as its presentation layer.
LlamaIndex will coordinate retrieval, PostgreSQL will store metadata and durable
state, and Qdrant will serve the search index. Embeddings and optional reranking
will run locally through FastEmbed; OpenAI will provide generation and claim
verification through separate Structured Outputs requests.

## Setup

Install **uv 0.13.6 or newer** using the
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Python is pinned in
[.python-version](.python-version), and dependencies are locked in [uv.lock](uv.lock).

From the repository root:

```bash
uv sync --locked
```

This creates the project environment, including the development tools.

## Checks

Check runtime compatibility, imports and local model inference:

```bash
uv run --locked python scripts/check_compatibility.py
```

The first compatibility run downloads the local models and requires internet
access. Ordinary runs do not call OpenAI. These synthetic smoke checks establish
technical compatibility, not retrieval or legal answer quality.

Run lint and formatting checks:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Install and run the local hooks:

```bash
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
```

Hooks may fix files; review the resulting changes before continuing.
CI checks lint, formatting and hook configuration on pushes to `main` and all
pull requests, using only development dependencies. It does not download models
or call OpenAI. The workflow has not yet been executed on GitHub Actions.

## Optional OpenAI check

Copy [.env.example](.env.example) to `.env` and fill in `OPENAI_API_KEY` and
`OPENAI_MODEL`. Keep the key private; `.env` is ignored by Git.
Load the file explicitly for the live check:

```bash
uv run --locked --env-file .env python scripts/check_compatibility.py --live
```

This sends two potentially paid synthetic requests: generation and verification.
It checks Structured Outputs parsing and validation, not legal quality.
The configured model is used without an automatic replacement or fallback.

## Scope and limitations

- v0.1 targets the original official English publication of Regulation (EU)
  2024/1689 from EUR-Lex.
- This historical text does not guarantee that answers reflect current law.
- The project is intended for research, not professional legal advice.
- GDPR and additional editions are outside v0.1.
