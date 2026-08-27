"""Gateway constants: model whitelist, prompt templates, and prices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from .models import CallTrace, PromptTemplate


@dataclass(frozen=True)
class ModelConfig:
    provider_model: str
    base_url: str
    api_key_env: str
    supports_structured_output: bool
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    provider_type: Literal["openai_chat", "openai_responses", "anthropic"] = "openai_chat"


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "general-primary": ModelConfig(
        provider_model=os.getenv("PRIMARY_PROVIDER_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("PRIMARY_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
    ),
    "general-backup": ModelConfig(
        provider_model=os.getenv("BACKUP_PROVIDER_MODEL", "deepseek-chat"),
        base_url=os.getenv("BACKUP_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_BACKUP_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
    ),
    "anthropic-primary": ModelConfig(
        provider_model=os.getenv("ANTHROPIC_PROVIDER_MODEL", "claude-sonnet-4-5"),
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        api_key_env="ANTHROPIC_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
        provider_type="anthropic",
    ),
    "openai-responses": ModelConfig(
        provider_model=os.getenv("RESPONSES_PROVIDER_MODEL", "gpt-4.1-mini"),
        base_url=os.getenv("RESPONSES_BASE_URL", "https://api.openai.com"),
        api_key_env="OPENAI_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
        provider_type="openai_responses",
    ),
}


PROMPT_TEMPLATES: dict[tuple[str, str], PromptTemplate] = {
    (
        "knowledge_decision",
        "v1",
    ): PromptTemplate(
        name="knowledge_decision",
        version="v1",
        system_template=(
            "你是${product_name}的知识库决策器。"
            "资料不足时搜索，资料充分时结束回答。不得编造制度内容。"
        ),
    )
}


# Prices are keyed by the platform model name used in LLMRequest.
PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "general-primary": {"input": 1.0, "output": 4.0},
    "general-backup": {"input": 0.8, "output": 3.2},
    "anthropic-primary": {"input": 3.0, "output": 15.0},
    "openai-responses": {"input": 0.4, "output": 1.6},
}


CALL_TRACES: list[CallTrace] = []

