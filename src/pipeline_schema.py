"""
Shared schema helpers for the repository-QA evaluation pipeline.

This module exists to stop field drift across evaluate.py, score.py,
analyze.py, and visualize.py.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable


ERROR_TIMEOUT = "timeout"
ERROR_MALFORMED_JSON = "malformed_json_unrecovered"
ERROR_TOOL = "tool_error"
ERROR_EMPTY = "empty_answer"
ERROR_NO_CITATION = "no_citation"

KNOWN_ERROR_TYPES = {
    ERROR_TIMEOUT,
    ERROR_MALFORMED_JSON,
    ERROR_TOOL,
    ERROR_EMPTY,
    ERROR_NO_CITATION,
}


def normalize_reasoning_type(record: Dict[str, Any]) -> str:
    return (
        record.get("reasoning_type")
        or record.get("expected_reasoning_type")
        or "unknown"
    )


def normalize_error_type(error_type: str | None) -> str | None:
    if not error_type:
        return None
    error_type = str(error_type).strip()
    legacy = {
        "json_parse_error": ERROR_MALFORMED_JSON,
        "tool_error_fatal": ERROR_TOOL,
    }
    return legacy.get(error_type, error_type)


def resolve_recursive_top_k(run_config: Dict[str, Any], default: int = 10) -> int:
    recursive_cfg = run_config.get("recursive", {})
    if isinstance(recursive_cfg, dict) and recursive_cfg.get("top_k") is not None:
        return int(recursive_cfg["top_k"])
    if run_config.get("top_k_rlm") is not None:
        return int(run_config["top_k_rlm"])
    return default


def quantize_judge_score(value: float | None) -> float | None:
    if value is None:
        return None
    allowed = [0.0, 0.5, 1.0]
    value = max(0.0, min(1.0, float(value)))
    return min(allowed, key=lambda target: abs(target - value))


def ensure_required_fields(record: Dict[str, Any], required: Iterable[str], context: str) -> None:
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")

