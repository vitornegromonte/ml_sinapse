"""Consumes scoring:tasks and publishes the BO maturity score.

The task carries `ato_infracional` + `features`; extracting those flags from
the BO text is a separate step and is not done here. The score itself lives in
services/occurrence_scoring.py (pure), so this worker is only transport.

Result goes to `scoring:tasks:results` for the backend to persist: this repo
has no `gold.bo_score` table or grant, and inventing one here would couple the
ML service to a backend migration. The message carries `weights_version` so a
score can always be traced back to the weight set that produced it.
"""

from config import settings
from models.tasks import ScoringTask
from services.occurrence_scoring import ScoringWeights, load_weights
from workers.base_worker import BaseWorker


class ScoringWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.scoring_stream, cfg)
        # Loaded at boot, not per task: a missing or unbalanced weights file is
        # a deployment error and should fail the container, not every message.
        self._weights: ScoringWeights = load_weights(cfg.scoring_weights_path)
        self._result_stream = f"{cfg.scoring_stream}:results"

    async def process(self, data: dict) -> None:
        task = ScoringTask(**data)
        score = self._weights.score(task.ato_infracional, task.features)

        await self.redis.xadd(
            self._result_stream,
            {
                "boletim_id": str(task.boletim_id),
                "ato_infracional": task.ato_infracional,
                "score": f"{score:.6f}",
                "weights_version": self._weights.version,
                "pipeline_run_id": task.pipeline_run_id or "",
            },
            maxlen=self.cfg.scoring_result_stream_maxlen,
        )

        # Same reason as ResultPublisher: an approximate maxlen evicts the
        # oldest score silently once the backend consumer falls behind.
        if self.cfg.scoring_result_stream_maxlen:
            length = await self.redis.xlen(self._result_stream)
            if length >= self.cfg.scoring_result_stream_maxlen:
                self.logger.warning(
                    "Results stream %s is at its %d-entry cap — the consumer is "
                    "behind and the oldest scores are being dropped.",
                    self._result_stream,
                    self.cfg.scoring_result_stream_maxlen,
                )

        self.logger.info(
            "BO score: boletim_id=%s ato=%s score=%.4f weights=%s",
            task.boletim_id,
            task.ato_infracional,
            score,
            self._weights.version,
        )
