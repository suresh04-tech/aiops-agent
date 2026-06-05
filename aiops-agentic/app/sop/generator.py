"""
app/sop/generator.py
─────────────────────
Core SOP generation logic.

Pipeline (simplified):
  LLM call → Markdown string → returned to process_sop.py → stored in DB

Public API:
  generate_sop_from_incident(...) → str  (Markdown runbook)
  generate_sop_from_prompt(...)   → str  (Markdown runbook)

No JSON parsing, no renderer, no validator needed.
"""

import logging
import os
import re

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from app.utils.aws_connector import BedrockClientFactory
from app.sop.prompts import (
    SOP_SYSTEM_PROMPT,
    build_incident_user_prompt,
    build_prompt_user_prompt,
    build_alert_user_prompt,
)

logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
SOP_BEDROCK_MODEL  = os.environ.get(
    "SOP_BEDROCK_MODEL_ID",
    "anthropic.claude-3-haiku-20240307-v1:0",
)
SOP_BEDROCK_REGION = os.environ.get("AWS_REGION", "ap-south-1")

# Markdown output is leaner than JSON (~25–30% fewer output tokens).
# 3000 is sufficient for a full production-grade runbook.
# Raise to 4096 only if very large runbooks are being truncated.
SOP_MAX_TOKENS  = int(os.environ.get("SOP_MAX_TOKENS", "3000"))
SOP_TEMPERATURE = float(os.environ.get("SOP_TEMPERATURE", "0.1"))


# ── LLM builder ───────────────────────────────────────────────────────────────

def _build_sop_llm() -> ChatBedrock:
    bedrock_factory = BedrockClientFactory()
    bedrock_client  = bedrock_factory.get_bedrock_runtime_client()
    region          = bedrock_factory.region or SOP_BEDROCK_REGION

    logger.info(
        f"[SOP-LLM] model={SOP_BEDROCK_MODEL} "
        f"max_tokens={SOP_MAX_TOKENS} "
        f"region={region}"
    )

    return ChatBedrock(
        model_id=SOP_BEDROCK_MODEL,
        region_name=region,
        client=bedrock_client,
        model_kwargs={
            "max_tokens":  SOP_MAX_TOKENS,
            "temperature": SOP_TEMPERATURE,
        },
    )


# ── Markdown cleanup ──────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """
    Light cleanup of LLM Markdown output.

    Handles two edge cases:
      1. LLM wraps output in ```markdown ... ``` fences (rare but possible).
      2. LLM adds a preamble line before the first # heading.

    Does NOT reformat content — preserves all code blocks and structure.
    """
    text = text.strip()

    # Strip outer ```markdown ... ``` fence if present
    fence_match = re.match(r"^```(?:markdown)?\s*\n(.*?)\n?```\s*$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
        logger.info("[SOP-LLM] Stripped outer markdown fence from response.")

    # If response doesn't start with a heading, trim any preamble before first #
    if not text.startswith("#"):
        heading_pos = text.find("\n#")
        if heading_pos != -1:
            trimmed = text[heading_pos:].strip()
            logger.info(
                f"[SOP-LLM] Trimmed {heading_pos} chars of preamble before first heading."
            )
            text = trimmed

    return text


# ── Stop reason guard ──────────────────────────────────────────────────────────

def _check_stop_reason(response) -> None:
    """Warn clearly if the model hit max_tokens — main cause of truncation."""
    stop_reason = None
    if hasattr(response, "response_metadata"):
        stop_reason = response.response_metadata.get("stop_reason")

    if stop_reason == "max_tokens":
        logger.warning(
            f"[SOP-LLM] ⚠️  stop_reason=max_tokens — output truncated! "
            f"Current SOP_MAX_TOKENS={SOP_MAX_TOKENS}. "
            f"Increase to 4096 in your .env if runbooks are being cut off."
        )
    else:
        logger.info(f"[SOP-LLM] stop_reason={stop_reason} (clean finish ✓)")


# ── Core invoke ───────────────────────────────────────────────────────────────

def _invoke_sop_llm(user_message: str) -> str:
    """
    Send system + user messages to the SOP LLM.

    Returns:
        Clean Markdown string of the complete runbook.

    Raises:
        ValueError  — LLM returned empty content.
        Any Bedrock exception is propagated to the caller.
    """
    llm = _build_sop_llm()

    messages = [
        SystemMessage(content=SOP_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    logger.info("[SOP-LLM] Invoking Bedrock for SOP generation…")
    response = llm.invoke(messages)

    # ── Token usage logging ────────────────────────────────────────────────
    logger.info("========== BEDROCK TOKEN USAGE ==========")
    if hasattr(response, "usage_metadata"):
        logger.info(f"usage_metadata={response.usage_metadata}")
    if hasattr(response, "response_metadata"):
        logger.info(f"response_metadata={response.response_metadata}")
    logger.info("==========================================")

    # ── Truncation warning ─────────────────────────────────────────────────
    _check_stop_reason(response)

    raw_content = response.content if hasattr(response, "content") else str(response)

    if not raw_content or not raw_content.strip():
        raise ValueError("[SOP-LLM] Bedrock returned empty content.")

    logger.info(f"[SOP-LLM] Raw response length: {len(raw_content)} chars")

    # ── Clean up and return ────────────────────────────────────────────────
    content_md = _clean_markdown(raw_content)

    logger.info(
        f"[SOP-LLM] Generated runbook — "
        f"{len(content_md)} chars, "
        f"~{content_md.count(chr(10))} lines"
    )

    return content_md


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_sop_from_incident(
    sop_id: str,
    incident: dict,
    rca_result: dict,
    evidence_data: dict,
) -> str:
    """
    Generate a Markdown runbook from incident + RCA context.

    Args:
        sop_id        : Caller-supplied SOP identifier (e.g. "SOP-201").
        incident      : Raw insight_incidents row as dict.
        rca_result    : Parsed analysis_result JSON from the incident.
        evidence_data : Raw incident_evidence row as dict.

    Returns:
        Markdown string of the complete runbook.
    """
    user_message = build_incident_user_prompt(
        sop_id=sop_id,
        incident=incident,
        rca_result=rca_result,
        evidence_data=evidence_data,
    )
    return _invoke_sop_llm(user_message)


def generate_sop_from_alert(
    sop_id: str,
    current_alert: dict,
    historical_context: dict[str, dict],
) -> str:
    """
    Generate a Markdown runbook from an alert + optional historical RCA context.

    Args:
        sop_id:             Caller-supplied SOP identifier.
        current_alert:      Current alert row from insight_alerts.
        historical_context: Merged dict keyed by incident_id with RCA, evidence, etc.

    Returns:
        Markdown string of the complete runbook.
    """
    user_message = build_alert_user_prompt(
        sop_id=sop_id,
        current_alert=current_alert,
        historical_context=historical_context,
    )
    return _invoke_sop_llm(user_message)


def generate_sop_from_prompt(sop_id: str, user_prompt: str) -> str:
    """
    Generate a Markdown runbook from a free-form user description.

    Args:
        sop_id      : Caller-supplied SOP identifier.
        user_prompt : Free-form context the user provided.

    Returns:
        Markdown string of the complete runbook.
    """
    user_message = build_prompt_user_prompt(sop_id=sop_id, user_prompt=user_prompt)
    return _invoke_sop_llm(user_message)