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

import sys

from workers.cluster_worker import ClusterWorker
from workers.embedding_worker import EmbeddingWorker
from workers.linkage_scoring_worker import LinkageScoringWorker
from workers.ocr_worker import OcrWorker
from workers.llm_worker import LlmWorker
from workers.ner_worker import NerWorker

WORKERS = {
    "ocr": OcrWorker,
    "llm": LlmWorker,
    "ner": NerWorker,
    "embedding": EmbeddingWorker,
    "linkage": LinkageScoringWorker,
    "cluster": ClusterWorker,
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ocr"
    if name not in WORKERS:
        available = ", ".join(WORKERS)
        print(f"Unknown worker '{name}'. Available: {available}")
        sys.exit(1)

    worker = WORKERS[name]()
    worker.run()


if __name__ == "__main__":
    main()
