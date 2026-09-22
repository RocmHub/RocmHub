"""LLM and deterministic rule providers for Autonomous AI Engineer."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.core.errors import LLMProviderError


class AgentAction(BaseModel):
    """Next action decided by the provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(..., description="Tool to invoke.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool.")
    thought: Optional[str] = Field(default=None, description="Internal reasoning or explanation.")
    is_final: bool = Field(default=False, description="Whether this action concludes the session.")


class LLMProvider(Protocol):
    """Protocol for AI Engineer decision providers."""

    def decide_action(self, context: Dict[str, Any]) -> AgentAction:
        """Evaluate context and decide the next engineering action."""
        ...


class AutonomousRulesProvider:
    """Deterministic, offline-first rule provider for autonomous model preparation on AMD ROCm."""

    def decide_action(self, context: Dict[str, Any]) -> AgentAction:
        """Determines the next optimal action in the OBSERVE -> PLAN -> ACT -> EVALUATE loop."""
        # Check if previous step had an error
        last_error = context.get("last_error")
        if last_error and not context.get("error_diagnosed", False):
            build_dir = context.get("output_dir") or context.get("plan", {}).get("output_dir")
            return AgentAction(
                tool_name="read_build_errors",
                tool_args={"build_dir": build_dir} if build_dir else {},
                thought=f"Encountered error '{last_error}'. Diagnosing failure details from build directory.",
            )

        # 1. Inspect model if not done
        if not context.get("model_inspected", False):
            return AgentAction(
                tool_name="inspect_model",
                tool_args={
                    "model_id": context["model_id"],
                    "revision": context.get("revision"),
                },
                thought="Initial observation: inspect Hugging Face model architecture and parameters.",
            )

        # 2. Inspect hardware if not done
        if not context.get("hardware_inspected", False):
            return AgentAction(
                tool_name="inspect_hardware",
                tool_args={},
                thought="Initial observation: probe local system hardware and AMD GPU presence.",
            )

        # 3. Check capability if not done
        if not context.get("capability_checked", False):
            return AgentAction(
                tool_name="check_capability",
                tool_args={
                    "model_id": context["model_id"],
                    "revision": context.get("revision"),
                    "device_id": 0,
                    "precision": context.get("precision", "fp16"),
                },
                thought="Evaluate model compatibility against detected environment and available VRAM.",
            )

        # 4. Create forge plan if not created
        if not context.get("plan_created", False):
            return AgentAction(
                tool_name="create_forge_plan",
                tool_args={
                    "model_id": context["model_id"],
                    "revision": context.get("revision"),
                    "precision": context.get("precision", "fp16"),
                    "target_gpu": context.get("target_gpu"),
                    "output_dir": context.get("output_dir"),
                },
                thought="Generate deterministic Model Forge recipe and build plan.",
            )

        # 5. Check if full weights materialization is explicitly required
        objective = context.get("objective", "BASE_PREPARATION")
        allow_full_weights = context.get("allow_full_weights", False)
        requires_full_weights = (objective in ("FULL_PREPARATION", "AMD_EXECUTION", "MAX_THROUGHPUT", "MIN_LATENCY"))

        if requires_full_weights and allow_full_weights and not context.get("weights_materialized", False):
            return AgentAction(
                tool_name="materialize_model",
                tool_args={
                    "model_id": context["model_id"],
                    "revision": context.get("revision"),
                    "download_weights": True,
                },
                thought="Materialize full weights as requested by objective and allowed by budget.",
            )

        # 6. Execute forge build if not done
        if not context.get("build_executed", False):
            download_weights = requires_full_weights and allow_full_weights
            return AgentAction(
                tool_name="execute_forge_build",
                tool_args={
                    "plan": context["plan"],
                    "download_weights": download_weights,
                    "force": True,
                },
                thought=f"Execute forge build steps (download_weights={download_weights}, force=True).",
            )

        # 7. Execution and benchmark on AMD GPU if requested
        if objective in ("AMD_EXECUTION", "MAX_THROUGHPUT", "MIN_LATENCY"):
            has_amd = context.get("has_amd_gpu", False)
            if not has_amd:
                # Host is macOS or non-AMD: explain and conclude safely without fake GPU execution
                return AgentAction(
                    tool_name="save_engineer_report",
                    tool_args={
                        "summary": "Completed model forge build. AMD GPU execution skipped because no AMD ROCm device is present on this host.",
                    },
                    thought="No AMD ROCm GPU present on host. Safely stopping before real GPU execution.",
                    is_final=True,
                )

            # If AMD GPU is present
            if not context.get("baseline_run", False):
                return AgentAction(
                    tool_name="run_baseline",
                    tool_args={
                        "model_id": context["model_id"],
                        "revision": context.get("revision"),
                        "precision": context.get("precision", "fp16"),
                    },
                    thought="Execute real baseline inference on AMD ROCm GPU.",
                )

            if not context.get("benchmark_run", False):
                return AgentAction(
                    tool_name="run_benchmark",
                    tool_args={
                        "model_id": context["model_id"],
                        "revision": context.get("revision"),
                        "precision": context.get("precision", "fp16"),
                    },
                    thought="Measure baseline performance on AMD ROCm GPU.",
                )

        # 8. All steps complete -> conclude
        return AgentAction(
            tool_name="save_engineer_report",
            tool_args={
                "summary": "Autonomous AI Engineer preparation completed successfully.",
            },
            thought="All required engineering steps executed. Saving final report.",
            is_final=True,
        )


class OpenAICompatibleProvider:
    """External LLM provider supporting OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ROCMHUB_LLM_API_KEY")
        self._base_url = (base_url or os.environ.get("ROCMHUB_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self._model = model or os.environ.get("ROCMHUB_LLM_MODEL", "gpt-4o")
        self._timeout_seconds = timeout_seconds

    def decide_action(self, context: Dict[str, Any]) -> AgentAction:
        """Call external LLM chat completion API and parse structured action JSON."""
        if not self._api_key:
            raise LLMProviderError(
                "OpenAI-compatible LLM provider requires ROCMHUB_LLM_API_KEY environment variable or explicit api_key.",
                details={"model": self._model, "base_url": self._base_url},
            )

        endpoint = f"{self._base_url}/chat/completions"
        system_prompt = (
            "You are an autonomous AMD ROCm AI Systems Engineer. Your job is to prepare and optimize open-source "
            "causal language models for AMD ROCm GPUs using only allowed tools. Respond ONLY with a valid JSON "
            "object matching: {\"tool_name\": string, \"tool_args\": object, \"thought\": string, \"is_final\": boolean}."
        )

        sanitized_context = {k: v for k, v in context.items() if "token" not in k.lower() and "key" not in k.lower()}
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Current engineering session context:\n{json.dumps(sanitized_context, default=str)}",
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choice = resp_data["choices"][0]["message"]["content"]
                action_data = json.loads(choice)
                return AgentAction.model_validate(action_data)
        except urllib.error.HTTPError as exc:
            # Mask API key from error
            raise LLMProviderError(
                f"LLM API request failed with HTTP {exc.code}: {exc.reason}",
                details={"status_code": exc.code, "endpoint": endpoint},
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to communicate with LLM provider: {exc}",
                details={"endpoint": endpoint, "error": str(exc)},
            ) from exc
