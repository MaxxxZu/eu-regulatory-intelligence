"""Synthetic runtime/inference smoke; no application or legal-quality evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "streamlit": "streamlit",
    "plotly": "plotly.graph_objects",
    "llama-index-core": "llama_index.core",
    "llama-index-vector-stores-qdrant": "llama_index.vector_stores.qdrant",
    "qdrant-client": "qdrant_client",
    "fastembed": "fastembed",
    "sqlalchemy": "sqlalchemy",
    "psycopg": "psycopg",
    "psycopg-binary": "psycopg.pq",
    "alembic": "alembic",
    "httpx": "httpx",
    "openai": "openai",
    "onnxruntime": "onnxruntime",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "pandas": "pandas",
}
TEXTS = ["The green box contains three pencils.", "The blue basket holds two apples."]
MODEL_ARTIFACTS = {
    "dense": (
        "aa8f8b060edb00e03bfdd08813a2949946c8ba55",
        "60edb05d4745fc243991236be4471a51174da329a4e4b7b787e8f8f60fa40e94",
    ),
    "bm25": (
        "22b8d2af71a76161e18dd432d2cee0eefa66e412",
        "e5b3cf37ec42b0f55c271ab4199350438771c1ec13ed07f0d92a15cf00ce3fbf",
    ),
    "reranker": (
        "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
        "a71b6a1cee66a5df0806f9da9226f0cb7fc298b327b4b57a52b46d760156c4c6",
    ),
}


def emit(check: str, status: str, **details: object) -> None:
    print(
        json.dumps({"check": check, "status": status, **details}, sort_keys=True),
        flush=True,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def runtime() -> None:
    expected = (ROOT / ".python-version").read_text().strip()
    require(platform.python_implementation() == "CPython", "CPython required")
    require(platform.python_version() == expected, "Pinned Python patch required")
    require(sys.version_info[:2] == (3, 14), "Python 3.14 required")
    require(sys.version_info.releaselevel == "final", "Stable release required")
    require(not sysconfig.get_config_var("Py_GIL_DISABLED"), "GIL build required")
    require(sys._is_gil_enabled(), "Enabled GIL required")
    emit(
        "runtime",
        "passed",
        python=platform.python_version(),
        gil=True,
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        libc=platform.libc_ver(),
    )


def package_import(name: str, module: str) -> None:
    imported = importlib.import_module(module)
    if name == "pydantic":
        require(imported.__version__.startswith("2."), "Pydantic v2 required")
    if name == "psycopg-binary":
        require(imported.__impl__ == "binary", "Binary psycopg implementation required")
    emit(f"import:{name}", "passed", version=importlib.metadata.version(name))


def artifact_info(instance: object, description: dict) -> dict:
    # FastEmbed has no public artifact-manifest API. Inspect the selected implementation
    # in the pinned version; fail if its actual downloaded directory cannot be identified.
    implementation = instance.model
    directory = Path(implementation._model_dir)
    require(directory.is_dir(), "Missing artifact directory")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files.append(
                {
                    "file": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    require(bool(files), "Empty model artifacts")
    revision = directory.name if directory.parent.name == "snapshots" else None
    manifest_sha256 = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "registry": description,
        "revision": revision,
        "files": files,
        "manifest_sha256": manifest_sha256,
    }


def local_model(kind: str, cache: Path) -> None:
    import numpy as np
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from huggingface_hub import snapshot_download

    classes = {
        "dense": (TextEmbedding, "BAAI/bge-small-en-v1.5"),
        "bm25": (SparseTextEmbedding, "Qdrant/bm25"),
        "reranker": (TextCrossEncoder, "Xenova/ms-marco-MiniLM-L-6-v2"),
    }
    cls, name = classes[kind]
    description = next(item for item in cls.list_supported_models() if item["model"] == name)
    revision, expected_digest = MODEL_ARTIFACTS[kind]
    files = [description["model_file"], *description["additional_files"]]
    if kind != "bm25":
        files += [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
    directory = snapshot_download(
        repo_id=description["sources"]["hf"],
        revision=revision,
        cache_dir=str(cache),
        allow_patterns=files,
    )
    options = (
        {"k": 1.2, "b": 0.75, "avg_len": 256.0, "language": "english"} if kind == "bm25" else {}
    )
    model = cls(
        model_name=name,
        cache_dir=str(cache),
        threads=2,
        cuda=False,
        providers=["CPUExecutionProvider"],
        specific_model_path=directory,
        **options,
    )
    artifacts = artifact_info(model, description)
    require(artifacts["revision"] == revision, "Artifact revision mismatch")
    require(artifacts["manifest_sha256"] == expected_digest, "Artifact checksum mismatch")
    if kind == "dense":
        vectors = np.asarray(list(model.embed(TEXTS)))
        queries = np.asarray(list(model.query_embed(["Where are the pencils?"])))
        require(
            vectors.shape == (2, 384) and queries.shape == (1, 384),
            "Dense shape mismatch",
        )
        require(
            bool(np.isfinite(vectors).all() and np.isfinite(queries).all()),
            "Nonfinite dense values",
        )
        norms = np.linalg.norm(vectors, axis=1)
        require(bool(np.allclose(norms, 1.0, atol=1e-4)), "Dense vectors must be normalized")
        require(bool(np.linalg.norm(queries[0]) > 0), "Zero query vector")
        details = {
            "shape": list(vectors.shape),
            "query_shape": list(queries.shape),
            "norms": norms.tolist(),
        }
    elif kind == "bm25":
        documents = list(model.embed(TEXTS))
        queries = list(model.query_embed(["green pencils"]))
        require(len(documents) == 2 and len(queries) == 1, "Sparse count mismatch")
        for vector in documents + queries:
            indices, values = np.asarray(vector.indices), np.asarray(vector.values)
            require(
                indices.ndim == values.ndim == 1 and len(indices) == len(values) > 0,
                "Invalid sparse shape",
            )
            require(
                bool(np.issubdtype(indices.dtype, np.integer) and (indices >= 0).all()),
                "Invalid sparse indices",
            )
            require(len(set(indices.tolist())) == len(indices), "Duplicate sparse indices")
            require(
                bool(np.isfinite(values).all() and (values > 0).all()),
                "Invalid sparse values",
            )
        details = {
            "document_nonzeros": [len(v.indices) for v in documents],
            "query_nonzeros": len(queries[0].indices),
            "parameters": options,
            "requires_qdrant_idf": True,
            "server_tested": False,
        }
    else:
        pairs = [("Where are the pencils?", text) for text in TEXTS]
        scores = np.asarray(list(model.rerank_pairs(pairs)))
        require(scores.shape == (len(pairs),), "Reranking count mismatch")
        require(bool(np.isfinite(scores).all()), "Nonfinite reranking scores")
        details = {"pairs": len(pairs), "scores": scores.tolist()}
    emit(kind, "passed", model=name, artifacts=artifacts, **details)


def live_schemas() -> tuple[type, type]:
    from pydantic import BaseModel, ConfigDict

    class Generation(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        claim: str
        evidence_id: Literal["synthetic-1"]
        quote: str

    class Verification(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        verdict: Literal["supported", "insufficient", "contradicted"]
        evidence_id: Literal["synthetic-1"]

    return Generation, Verification


def live_request(client: object, model: str, schema: type, prompt: str) -> str:
    import openai
    from pydantic import ValidationError

    try:
        # Inspect completion/refusal before parsing: incomplete JSON must not be
        # misreported as a schema error by an eager SDK parse helper.
        response = client.responses.create(
            model=model,
            instructions="Use only the supplied synthetic evidence. Follow the schema.",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                }
            },
            max_output_tokens=1024,
            store=False,
        )
        if response.status == "incomplete":
            return "incomplete_response"
        if any(
            getattr(part, "type", None) == "refusal"
            for item in response.output
            for part in getattr(item, "content", [])
        ):
            return "refusal"
        if response.status != "completed":
            return "provider_failure"
        if not response.output_text:
            return "schema_parsing_error"
        parsed = schema.model_validate_json(response.output_text, strict=True)
        # This checks the tiny synthetic fixture, not legal groundedness.
        if schema.__name__ == "Generation":
            if parsed.claim != TEXTS[0] or parsed.quote != TEXTS[0]:
                return "synthetic_content_mismatch"
        elif parsed.verdict != "supported":
            return "synthetic_content_mismatch"
        return "passed"
    except openai.APITimeoutError:
        return "timeout"
    except (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
    ):
        return "access_unavailable"
    except openai.RateLimitError:
        return "rate_limited"
    except openai.BadRequestError:
        return "request_or_schema_rejected"
    except ValidationError, json.JSONDecodeError, openai.APIResponseValidationError:
        return "schema_parsing_error"
    except openai.LengthFinishReasonError:
        return "incomplete_response"
    except openai.ContentFilterFinishReasonError:
        return "refusal"
    except openai.APIError:
        return "provider_failure"


def openai_live() -> bool:
    from openai import OpenAI

    missing = [
        name for name in ("OPENAI_API_KEY", "OPENAI_MODEL") if not os.environ.get(name, "").strip()
    ]
    if missing:
        emit("openai", "configuration_missing", missing=missing)
        return False
    generation, verification = live_schemas()
    prompts = [
        f"Evidence synthetic-1: {TEXTS[0]} Return this exact sentence as both claim and quote.",
        f"Evidence synthetic-1: {TEXTS[0]} Claim: {TEXTS[0]} Classify support.",
    ]
    results = []
    # Explicit official endpoint, finite timeout, zero retries, no provider/model fallback.
    with OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1",
        timeout=30.0,
        max_retries=0,
    ) as client:
        for schema, prompt in zip((generation, verification), prompts, strict=True):
            result = live_request(client, os.environ["OPENAI_MODEL"], schema, prompt)
            emit(f"openai:{schema.__name__.lower()}", result)
            results.append(result == "passed")
    return all(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Send two potentially paid OpenAI requests"
    )
    args = parser.parse_args()
    failed = False

    def run(name: str, check: Callable[[], None]) -> None:
        nonlocal failed
        try:
            check()
        except Exception as error:
            # Exception messages/provider bodies may contain credentials; never print them.
            emit(name, "failed", error_type=type(error).__name__)
            failed = True

    run("runtime", runtime)
    for package, module in PACKAGES.items():
        run(f"import:{package}", lambda p=package, m=module: package_import(p, m))
    cache = ROOT / ".venv" / "model-cache"
    for kind in ("dense", "bm25", "reranker"):
        run(kind, lambda k=kind: local_model(k, cache))
    if args.live:

        def live() -> None:
            nonlocal failed
            if not openai_live():
                failed = True

        run("openai", live)
    else:
        emit("openai", "not_requested", live_gate="not_run")
    emit("summary", "failed" if failed else "passed", legal_quality_evaluated=False)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
