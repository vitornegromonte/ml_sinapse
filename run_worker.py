#!/usr/bin/env python3
"""Entrypoint to run any ML worker by name.

Usage:
    python run_worker.py ocr
    python run_worker.py llm
    python run_worker.py ner
    python run_worker.py embedding
    python run_worker.py linkage
    python run_worker.py cluster
"""

import importlib
import sys

# name -> "module:Class". Resolved in main(), not imported here: each worker
# pulls a different heavy stack (torch/vLLM, spaCy, sentence-transformers,
# scikit-learn), so a container used to load all six before starting the one it
# was asked for, and a missing optional dependency in any of them broke every
# worker's boot. This keeps `python run_worker.py <name>` paying only for <name>.
WORKERS = {
    "ocr": "workers.ocr_worker:OcrWorker",
    "llm": "workers.llm_worker:LlmWorker",
    "ner": "workers.ner_worker:NerWorker",
    "embedding": "workers.embedding_worker:EmbeddingWorker",
    "linkage": "workers.linkage_scoring_worker:LinkageScoringWorker",
    "cluster": "workers.cluster_worker:ClusterWorker",
}


def resolve(name: str):
    module_name, _, class_name = WORKERS[name].partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ocr"
    if name not in WORKERS:
        available = ", ".join(WORKERS)
        print(f"Unknown worker '{name}'. Available: {available}")
        sys.exit(1)

    worker = resolve(name)()
    worker.run()


if __name__ == "__main__":
    main()
