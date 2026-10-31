"""ScoringWorker: a pure score in, one message on the result stream out."""
import logging

import pytest

from workers.scoring_worker import ScoringWorker


@pytest.fixture
def worker(mock_redis):
    w = ScoringWorker()
    w.redis = mock_redis
    return w


async def test_publishes_the_score_with_its_weights_version(worker, mock_redis):
    """Arrange: a complete HOMICIDIO case.
    Act: process the task.
    Assert: one result message carries id, nature, score and weight version.
    """
    await worker.process(
        {
            "boletim_id": 42,
            "ato_infracional": "HOMICIDIO",
            "features": {
                "victim": True,
                "suspect": True,
                "address": True,
                "witnesses": True,
            },
            "pipeline_run_id": "run-1",
        }
    )

    stream, fields = mock_redis.xadd.call_args.args
    assert stream == "scoring:tasks:results"
    assert fields["boletim_id"] == "42"
    assert fields["ato_infracional"] == "HOMICIDIO"
    assert fields["score"] == "1.000000"
    assert fields["weights_version"] == "v1"
    assert fields["pipeline_run_id"] == "run-1"


async def test_penalty_reaches_the_published_score(worker, mock_redis):
    """Arrange: a HOMICIDIO without the victim flag.
    Act: process the task.
    Assert: the 10% penalty is already applied in the published score.
    """
    await worker.process(
        {
            "boletim_id": 43,
            "ato_infracional": "HOMICIDIO",
            "features": {"suspect": True, "address": True, "witnesses": True},
        }
    )

    _, fields = mock_redis.xadd.call_args.args
    assert fields["score"] == "0.050000"
    assert fields["pipeline_run_id"] == ""


async def test_unknown_nature_fails_the_task_without_publishing(worker, mock_redis):
    """Arrange: a nature absent from the weights (a config gap, not a data state).
    Act: process the task.
    Assert: it raises so BaseWorker retries/DLQs instead of publishing a zero.
    """
    with pytest.raises(ValueError, match="NATUREZA_DESCONHECIDA"):
        await worker.process(
            {
                "boletim_id": 44,
                "ato_infracional": "NATUREZA_DESCONHECIDA",
                "features": {},
            }
        )

    mock_redis.xadd.assert_not_awaited()


async def test_missing_boletim_id_is_a_payload_error(worker):
    """Arrange: a task without the BO id.
    Act/Assert: typed parsing rejects it before any score is computed.
    """
    with pytest.raises(ValueError, match="boletim_id"):
        await worker.process({"ato_infracional": "AMEACA", "features": {}})


async def test_warns_when_the_result_stream_sits_at_its_cap(
    worker, mock_redis, caplog
):
    """Arrange: the result stream already at maxlen.
    Act: process one task with warnings captured.
    Assert: the score is still published and the silent eviction is logged.
    """
    mock_redis.xlen.return_value = worker.cfg.scoring_result_stream_maxlen

    with caplog.at_level(logging.WARNING, logger=worker.logger.name):
        await worker.process(
            {"boletim_id": 45, "ato_infracional": "AMEACA", "features": {}}
        )

    mock_redis.xadd.assert_awaited_once()
    assert "cap" in caplog.text
