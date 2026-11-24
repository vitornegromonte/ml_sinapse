# Testing

How the test suite is organised and how a test is written. The short rule:
**every test is Arrange → Act → Assert, in that order, one Act per test.**

## Run

```bash
pip install -r requirements-dev.txt -r requirements-test.txt  # CPU-only
python -m pytest -q -rs
```

No GPU, Postgres, Redis or MinIO is needed: the suite mocks every external
boundary. CI runs the same command on Python 3.11
(`.github/workflows/tests.yml`).

## Arrange, Act, Assert

- **Arrange** — build the subject and its inputs, and set up the mocks. Use
  fixtures for setup shared by several tests, but keep the values this test
  depends on visible in its body.
- **Act** — exactly one call into the code under test. If a test needs two
  Acts, it is testing two behaviours; split it.
- **Assert** — check the observable outcome (return value, raised error,
  published message, DB write), not the private calls the code happens to make.

Every test documents the three blocks in its docstring, one label per line.
The body stays free of `# Arrange` noise. Example
(`tests/services/test_image_loader.py::TestImageLoader::test_from_base64`):

```python
def test_from_base64(self):
    """Arrange: an in-memory PNG encoded as base64.
    Act: load from image_base64.
    Assert: a PIL image of the original size is returned.
    """
    loader = ImageLoader()
    img = Image.new("RGB", (20, 30), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = {"image_base64": base64.b64encode(buf.getvalue()).decode()}

    result = loader.load(payload)

    assert isinstance(result, Image.Image)
    assert result.size == (20, 30)
```

An error path lists Act and Assert separately even though one statement both
calls and raises:

```python
def test_from_path_disabled_without_allowed_root(self):
    """Arrange: a loader with no allowed_root.
    Act: load an absolute image_path.
    Assert: ValueError — local paths are disabled, not arbitrary reads.
    """
    loader = ImageLoader()

    with pytest.raises(ValueError, match="disabled"):
        loader.load({"image_path": "/etc/passwd"})
```

A long Assert may continue over several lines, but it stays one block. The
labels are literal: `Arrange:`, `Act:`, `Assert:`.

Name tests after the behaviour they pin —
`test_<what>_<condition>_<expected>` — so a failure reads as a sentence:
`test_exhausted_retries_dead_letters_and_acks`.

### What belongs in each block

| Block | Belongs | Does not belong |
|---|---|---|
| Arrange | fixtures, payloads, mock return values, temp files | the call under test |
| Act | one `await` / call | assertions, more setup |
| Assert | outcome + side effects | re-running the Act to get another value |

Multiple assertions are fine when they describe one outcome (shape *and* the
size of the same image). Several unrelated assertions usually mean several
tests.

## Layout

The `tests/` folder mirrors the source tree. The test for `services/gold.py` is
`tests/services/test_gold.py`. Each folder is a Python package (it has an
`__init__.py`), so two folders can hold files with the same name.

```text
tests/
├── conftest.py              fixtures for all tests, and the vLLM stand-in
├── fixtures/                data files (compose mounts fixtures/images)
├── test_config.py           config.py
├── models/                  models/
│   ├── test_llm_model.py
│   ├── test_tasks.py
│   └── ocr/                 models/ocr/: test_base, test_dharma, test_glm,
│                            test_mlflow_model, test_registry, test_task
├── scripts/                 scripts/: test_profile_models, test_model_lock_tools,
│                            test_log_manifests_to_mlflow
├── services/                services/: one file per module
└── workers/                 workers/
    ├── test_base_worker.py, test_run_worker.py, test_startup_compat.py
    ├── test_{cluster,embedding,linkage_scoring,llm,ner,scoring}_worker.py
    └── ocr/                 a folder, because the OCR worker has many behaviors
        ├── conftest.py      fixtures: mock_model, worker
        ├── helpers.py       FakeRedis, scanned_pdf, echo_model, run_pdf
        └── test_process.py, test_permanent_errors.py, test_batching.py,
            test_resume.py, test_timeout.py
```

Rules:

1. Name a test file `test_<module>.py`, after the module it tests.
2. A class groups the tests for one entry point or one behavior.
3. A module with many behaviors may become a folder, as `workers/ocr/` did.
   Put its fixtures in the folder's `conftest.py`. Put plain helpers that
   several files share in `helpers.py`, and import them from there.
4. Put a fixture in `tests/conftest.py` only when tests in more than one folder
   use it (`mock_redis`, `sample_image`, `ocr_payload`).
5. Find data files from `Path(__file__)`, never from the working directory.
   Then the tests pass from any folder.

Run a part of the suite:

```bash
python -m pytest -q tests/workers                   # one area
python -m pytest -q tests/workers/ocr               # one folder
python -m pytest -q tests/services/test_gold.py     # one file
```

## Mocks and known gaps

- Mock the boundary, never the unit: Redis, DB sessions, MinIO, model classes.
  The worker under test is always real.
- `tests/conftest.py` installs a stand-in `vllm` module when the real package
  is missing (it is CUDA-only and not in `requirements-test.txt`). Tests that
  patch `vllm.LLM` / `vllm.SamplingParams` pass either way, but a real API
  rename is only caught on the GPU machine.
- Mocked tests cannot catch wrong call shapes on a real client — an `AsyncMock`
  accepts any keyword name. `tests/workers/test_base_worker.py` documents the
  `xautoclaim(start=...)` vs `start_id` incident for this reason.
- There is no Redis/Postgres/MinIO integration run. Statement-level tests such
  as `test_gold.py` pin the SQL sequence deliberately; they are the exception
  to "assert outcomes, not implementation", because without a database the
  sequence *is* the observable behaviour.
- GPU-only paths are exercised on the GPU machine, not in CI. State that in
  the test module docstring when a test depends on that run.

## Adding a test

1. Put it in the matching `tests/<folder>/test_<module>.py`. Use the folder that
   mirrors the source module (see Layout).
2. Write the Arrange/Act/Assert docstring and body first; add the fixture only
   if another test needs the same setup.
3. Assert behaviour. If you must pin an implementation detail, say why in one
   line above the assertion.
4. Run `python -m pytest -q tests/<folder>/test_<module>.py`.
