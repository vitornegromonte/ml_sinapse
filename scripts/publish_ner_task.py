#!/usr/bin/env python3
"""Publish a sample NER task to Redis, to smoke-test the NER worker.

Unlike OCR (image files), NER just needs {"document_id", "text"} — no
fixtures required. The sample text below exercises several of the 11 domain
labels the mock model was trained on (ADR-005): PESSOA, PLACA, LOCAL, DATA,
ORGAO, ARTIGO_LEI.

Usage:
    python scripts/publish_ner_task.py [--redis URL] [--stream STREAM] [--text TEXT]

The NER worker only logs its result today (see ner_worker.py) — after
publishing, watch the worker's stdout/logs for the "NER result: ..." line
rather than expecting a side effect elsewhere.
"""

import argparse
import sys

DEFAULT_TEXT = (
    "No dia 15/03/2026, o Sr. João da Silva, portador do veículo de placa "
    "ABC-1234, compareceu à Delegacia de Homicídios da Capital para prestar "
    "depoimento, conforme Art. 304 do CPP."
)


def main():
    parser = argparse.ArgumentParser(description="Publish a NER task to Redis")
    parser.add_argument("--redis", default="redis://localhost:6380", help="Redis URL")
    parser.add_argument("--stream", default="ner:tasks", help="Redis stream name")
    parser.add_argument("--document-id", default="smoke-test-001")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    try:
        from redis import Redis
    except ImportError:
        print("redis not installed. Run: pip install redis")
        sys.exit(1)

    r = Redis.from_url(args.redis, decode_responses=True)

    task = {"document_id": args.document_id, "text": args.text}
    msg_id = r.xadd(args.stream, task, maxlen=1000)
    print(f"Published to {args.stream} [{msg_id}]")
    print(f"  document_id={args.document_id!r}")
    print(f"  text={args.text!r}")
    print("\nWatch the ner-worker container logs for the 'NER result: ...' line.")

    if r.ping():
        print("Redis connected OK")
    else:
        print("WARNING: Redis ping failed")


if __name__ == "__main__":
    main()
