from __future__ import annotations

import logging
import os
from typing import Any

from models.base_model import BaseModelWrapper

logger = logging.getLogger(__name__)


LLM_ENGINE_TYPES = {
    "vllm": "vllm",
    "transformers": "transformers",
    "llama.cpp": "llama_cpp",
    "bentoml": "bentoml",
    "mlflow": "mlflow",
}


class LLMModel(BaseModelWrapper):
    def __init__(
        self,
        model_path: str,
        engine: str = "vllm",
        device: str = "auto",
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        sampling_params: dict | None = None,
        **kwargs: Any,
    ):
        self.model_path = model_path
        self.engine = engine
        self.device = device
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.sampling_params = sampling_params or {}
        self.kwargs = kwargs

        self._llm: Any = None
        self._engine_type: str = ""

    def load(self) -> None:
        engine_key = self.engine.lower().strip()
        if engine_key not in LLM_ENGINE_TYPES:
            raise ValueError(
                f"Unknown engine '{self.engine}'. "
                f"Available: {list(LLM_ENGINE_TYPES.keys())}"
            )

        if engine_key == "vllm":
            self._load_vllm()
        elif engine_key == "transformers":
            self._load_transformers()
        elif engine_key == "llama.cpp":
            self._load_llamacpp()
        elif engine_key == "bentoml":
            self._load_bentoml()
        elif engine_key == "mlflow":
            self._load_mlflow()
        else:
            raise NotImplementedError(f"Engine '{self.engine}' not implemented")
 
    def _load_vllm(self) -> None:
        from vllm import LLM as VLLM
        from transformers import AutoTokenizer
        self._engine_type = "vllm"

        # vLLM pins a specific GPU index via CUDA_VISIBLE_DEVICES, not a
        # constructor kwarg — same convention as models/ocr/dharma.py. Was
        # previously a no-op: `self.device` was stored but never applied, so
        # vLLM always defaulted to whichever GPU it saw first (device 0).
        if self.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]

        kwargs: dict[str, Any] = {
        "model": self.model_path,
        "dtype": self.dtype,
        "gpu_memory_utilization": self.gpu_memory_utilization,
        "tensor_parallel_size": self.tensor_parallel_size,
        }

        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        # Was nested inside the `if` above — extra constructor kwargs (self.kwargs)
        # were silently dropped whenever max_model_len wasn't set. Unconditional now.
        kwargs.update(self.kwargs)

        self._llm = VLLM(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        logger.info(
            "vLLM engine loaded (model=%s, gpu_mem=%.2f, tp=%d)",
            self.model_path,
            self.gpu_memory_utilization,
            self.tensor_parallel_size,
        )

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._engine_type = "transformers"
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
        )
        model.eval()
        self._llm = (model, tokenizer)
        logger.info(
            "Transformers engine loaded (model=%s, device=%s)",
            self.model_path,
            self.device,
        )

    def _load_llamacpp(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Run: pip install llama-cpp-python"
            )

        self._engine_type = "llama.cpp"
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_gpu_layers": -1 if self.device != "cpu" else 0,
        }
        kwargs.update(self.kwargs)

        self._llm = Llama(**kwargs)
        logger.info(
            "llama.cpp engine loaded (model=%s)",
            self.model_path,
        )

    def _load_bentoml(self) -> None:
        import bentoml  # lazy: optional engine, bentoml is not a hard dependency

        self._engine_type = "bentoml"
        ref: Any = bentoml.models.get(self.model_path)
        self._llm = bentoml.Runner(
            ref.to_runner(),
            runnable_options={"gpu_memory_utilization": self.gpu_memory_utilization},
        )
        self._llm.init_local()
        logger.info(
            "BentoML engine loaded (model=%s)", self.model_path
        )

    def _load_mlflow(self) -> None:
        import mlflow

        self._engine_type = "mlflow"
        mlflow.set_tracking_uri(mlflow.tracking.get_tracking_uri())
        self._llm = mlflow.pyfunc.load_model(self.model_path)
        logger.info(
            "MLflow engine loaded (model=%s)", self.model_path
        )

    def _build_sampling_params(self) -> Any:
        if self._engine_type == "vllm":
            from vllm import SamplingParams

            return SamplingParams(**self.sampling_params)

        return self.sampling_params

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        merged_params = {**self.sampling_params, **kwargs}

        if self._engine_type == "vllm":
            from vllm import SamplingParams

            sampling_params = SamplingParams(**merged_params)
            prompts = [inputs] if isinstance(inputs, str) else inputs

            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                formatted = []
                for p in prompts:
                    messages = [{"role": "user", "content": p}]
                    formatted.append(self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ))
                prompts = formatted

            outputs = self._llm.generate(prompts, sampling_params)
            return [
                {"text": o.outputs[0].text, "tokens": o.outputs[0].token_ids}
                for o in outputs
            ]

        if self._engine_type == "transformers":
            import torch
            from transformers import AutoTokenizer

            model, tokenizer = self._llm
            inputs_batched = [inputs] if isinstance(inputs, str) else inputs
            results = []
            for prompt in inputs_batched:
                enc = tokenizer(prompt, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=merged_params.get("max_tokens", 256),
                        do_sample=merged_params.get("do_sample", True),
                        temperature=merged_params.get("temperature", 0.7),
                    )
                out = tokenizer.decode(
                    gen[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                results.append({"text": out})
            return results

        if self._engine_type == "llama.cpp":
            prompts = [inputs] if isinstance(inputs, str) else inputs
            return [
                {"text": self._llm(p, **merged_params)["choices"][0]["text"]}
                for p in prompts
            ]

        if self._engine_type == "bentoml":
            prompts = [inputs] if isinstance(inputs, str) else inputs
            results = self._llm.run(prompts, **merged_params)
            if isinstance(results, list) and len(results) == 1:
                return [{"text": results[0]}]
            return [{"text": r} for r in results]

        if self._engine_type == "mlflow":
            prompts = [inputs] if isinstance(inputs, str) else inputs
            return [
                {"text": self._llm.predict(p)[0]}
                for p in prompts
            ]

        raise NotImplementedError(f"Engine '{self.engine}' not supported for predict")
    
    def unload(self) -> None:
	    self._llm = None
	    self._tokenizer = None
	    import gc
	    import torch

	    gc.collect()
	    if torch.cuda.is_available():
	        torch.cuda.empty_cache()
	    logger.info("Model unloaded")
