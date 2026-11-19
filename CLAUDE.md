# ml_sinapse

`ml_sinapse` is the standalone Machine Learning (ML) service for SINAPSE.

It runs independent workers that process tasks from the backend (`sinapse` repository). Workers communicate with the backend through Redis Streams and shared PostgreSQL/MinIO resources.

**This repository is an independent service.** It must not depend on Python packages or internal modules from the `sinapse` repository.

## Guides

Read the guide before you work in its area. Do not repeat guide content in this file.

| Guide | Read it when you |
|---|---|
| [docs/.claude/architecture.md](docs/.claude/architecture.md) | change workers, streams, flows, storage, or deployment |
| [docs/.claude/code-standards.md](docs/.claude/code-standards.md) | write or change any code |
| [docs/.claude/testing.md](docs/.claude/testing.md) | write or change any test |
| [docs/README.md](docs/README.md) | need the ADR index (`docs/adr-*.md`) |

## Repository Layout

- `workers/` — one process per worker: `ocr`, `llm`, `ner`, `embedding`, `linkage`, `cluster`, `scoring`
- `workers/base_worker.py` — consumption, retry, dead-letter, lease, and `XAUTOCLAIM` recovery
- `services/` — database, storage, publishers, scoring, and linkage helpers
- `models/` — model wrappers (ADR-009)
- `config.py` — environment settings and Redis Stream names
- `scripts/` — backfills, profiling, and manual task publishers
- `artifacts/` — classifier and scoring weights, baked into the image
- `tests/` — one `test_<module>.py` per source module

## Commands

```bash
pip install -r requirements-dev.txt -r requirements-test.txt   # CPU-only
python -m pytest -q -rs                                        # no GPU, DB, Redis, or MinIO needed
python run_worker.py <ocr|llm|ner|embedding|linkage|cluster|scoring>
BACKEND_HOST=<ip> docker compose up --build
```

## Hard Rules

These rules apply to every change. The guides give the detail.

1. **Database.** Workers use the `sinapse_ml` role. They read `silver` and write `gold`. **Never write `silver`.** Never create, alter, or drop schemas or tables here. Schema and migrations belong to `sinapse`.
2. **Backend boundary.** Do not add HTTP calls or Python imports between workers and the backend, unless an ADR allows it. Do not move backend responsibilities into this repository.
3. **Redis Streams.** Stream names are in `config.py`. When you change a name, update both repositories. Do not rename payload fields silently.
4. **Workers.** Build on `workers/base_worker.py`. Do not acknowledge a message before processing succeeds. Make processing safe to retry.
5. **Models.** Keep models behind wrappers. Do not load a model at import time. Keep CPU-only development possible.
6. **Tests.** Add or update tests for every behavior change. Follow Arrange, Act, Assert. Never weaken a test to make a change pass.
7. **Docs.** Update the matching guide when you change worker behavior, stream contracts, environment variables, model integration, or deployment. Document only what the code does.
8. **ADRs.** Read the relevant ADR before you change an established pattern. Record a significant decision in a new `docs/adr-*.md`. Do not contradict an ADR silently.

## Git Hygiene

- `TASK.md`, `ML_review.md`, and `architecture.html` are git-ignored. Never force-add them.
- Do not commit secrets, credentials, tokens, or local environment files.
- Do not modify unrelated files or revert user changes.
- Do not commit unless the user asks.

## Workflow

Before you change code:

1. Read the relevant source, tests, and ADRs.
2. Read the matching guide in `docs/.claude/`.
3. Identify the backend dependencies and message contracts.

While you change code:

1. Make the smallest change that solves the problem.
2. Keep backward compatibility unless the user asks for a break.
3. Add or update tests and documentation.

Before you finish:

1. Review the diff for unintended changes.
2. Run `python -m pytest -q`.
3. Report the test results and the important files you changed.
4. State what you could not verify.

Do not claim that a test, command, or integration works unless you verified it.

## Writing Rules

Use Simplified Technical English (STE) at about 80% strictness of ASD-STE100.

- Use short, active sentences. Keep one main idea per sentence.
- Use simple, common words. Explain technical terms when they first appear.
- Use consistent names for technical concepts.
- Give instructions in the order of the steps. Use numbered lists for procedures.
- Be explicit. Remove vague or unnecessary wording.
