# EU Regulatory Intelligence

An evidence-grounded research tool for the EU AI Act. Designed to help readers
explore legal provisions, inspect the evidence behind answers, and follow
citations to the official EUR-Lex text.

The project also provides an engineering view of retrieval quality: how dense,
sparse and hybrid search find relevant provisions, and when the available
evidence is insufficient to support an answer.

## Planned capabilities

- Search the original official English edition of the EU AI Act by article,
  paragraph and point.
- Ask questions and inspect answers alongside supporting passages and official
  source links.
- Validate quoted text and assess whether the evidence supports each claim,
  withholding answers that fail verification.
- Compare dense, sparse and hybrid retrieval, with optional reranking.
- Explore results through a Streamlit research interface and Retrieval Lab.
- Review reproducible examples and measured evaluation results, including
  unsupported claims and questions that require abstention.

The project is in early development. The repository currently contains the
Python environment, dependency lock, development tools and CI configuration;
application features are not implemented yet.

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
or call OpenAI.

## Scope and limitations

- v0.1 targets the original official English publication of Regulation (EU)
  2024/1689 from EUR-Lex.
- This historical text does not guarantee that answers reflect current law.
- The project is intended for research, not professional legal advice.
- GDPR and additional editions are outside v0.1.
