"""
app/agent/graph.py
──────────────────
LangGraph ReAct agent for autonomous AWS incident investigation (v4).

Key changes in v4
─────────────────
• Two-phase LLM approach:
    Phase 1 — ReAct loop (llm + bind_tools): agent calls tools iteratively.
    Phase 2 — Structured extraction (llm.with_structured_output): once the
              loop ends, the final agent message is sent to a SECOND LLM call
              that uses Bedrock's native structured output to parse the result
              into a guaranteed-valid schema. This eliminates all JSON parse
              failures caused by truncated or malformed text output.

• Output schema trimmed to exactly 5 fields:
    probable_root_cause (str)
    confidence          (int 0-100)
    evidence            (list[str])
    dependency_impact   (list[str])
    recommended_actions (list[str])

• DB status updates wired to WORKFLOW_STATES — only Python writes these.
• Pre-triage generates investigation hypotheses.
• All incidents continue through the LangGraph investigation workflow.
• Stop-condition: has_sufficient_evidence() checked after each tool.
"""

import json
import logging
import os
import time
from typing import Annotated, Literal, Sequence

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrock
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import ALL_TOOLS
from app.agent.rules import WORKFLOW_STATES
from app.agent.evaluators import (
    correlate_timeline,
    extract_rca_signals,
    has_sufficient_evidence,
    pre_triage_targets,
)
from app.utils.aws_connector import BedrockClientFactory

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BEDROCK_MODEL  = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
BEDROCK_REGION = os.environ.get("AWS_REGION", "ap-south-1")
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "12"))

# ── Output schema (5 fields only) ────────────────────────────────────────────

# Pydantic-like schema dict for with_structured_output
RCA_SCHEMA = {
    "title": "IncidentRCA",
    "description": "Root cause analysis result for an AWS incident",
    "type": "object",
    "properties": {
        "probable_root_cause": {
            "type": "string",
            "description": "One sentence: the exact failure cause naming the instance or service",
        },
        "confidence": {
            "type": "integer",
            "description": "Confidence score 0-100",
            "minimum": 0,
            "maximum": 100,
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of direct evidence strings (log errors, metric values) that prove the root cause",
        },
        "dependency_impact": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of downstream services/endpoints affected and how",
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly 3 items: [fix, verify, prevent]",
        },
    },
    "required": [
        "probable_root_cause",
        "confidence",
        "evidence",
        "dependency_impact",
        "recommended_actions",
    ],
}


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:        Annotated[Sequence[BaseMessage], add_messages]
    incident_id:     str
    tool_calls_made: list[str]
    rca_signals:     dict
    resolved_targets: list[dict]
    alb_meta:        dict
    investigation_state: dict
    evidence: list[str]
    investigation_findings: list[dict]


# ── LLM builders ─────────────────────────────────────────────────────────────

def _build_bedrock_base() -> ChatBedrock:
    """Shared Bedrock client (no tools, no structured output)."""
    bedrock_factory = BedrockClientFactory()
    bedrock_client  = bedrock_factory.get_bedrock_runtime_client()
    bedrock_region  = bedrock_factory.region or BEDROCK_REGION

    return ChatBedrock(
        model_id=BEDROCK_MODEL,
        region_name=bedrock_region,
        client=bedrock_client,
        model_kwargs={
            "max_tokens":  1500,
            "temperature": 0.1,
            "top_p":       0.9,
        },
    )


def _build_llm_with_tools() -> ChatBedrock:
    """Phase 1 LLM — ReAct reasoning loop with tool access."""
    return _build_bedrock_base().bind_tools(ALL_TOOLS)


def _build_llm_structured() -> ChatBedrock:
    """Phase 2 LLM — structured output extraction (no tools)."""
    return _build_bedrock_base().with_structured_output(RCA_SCHEMA)


# ── Throttle-aware invoke ─────────────────────────────────────────────────────

def safe_llm_invoke(llm, messages):
    for attempt in range(5):
        try:
            return llm.invoke(messages)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                sleep_time = min(5 * (attempt + 1), 30)
                logger.warning(f"[Bedrock] throttled, retrying in {sleep_time}s")
                time.sleep(sleep_time)
            else:
                raise
    raise RuntimeError("Bedrock retry limit exceeded")


# ── Node: agent (ReAct reasoning) ────────────────────────────────────────────

def agent_node(state: AgentState, llm_with_tools) -> dict:
    logger.info(f"[Agent] Invoking LLM with {len(state['messages'])} messages")
    response = safe_llm_invoke(llm_with_tools, state["messages"])
    tool_call_count = len(response.tool_calls) if hasattr(response, "tool_calls") else 0
    logger.info(f"[Agent] turn complete | tool_calls={tool_call_count}")
    
    if hasattr(response, "content") and response.content:
        logger.info(f"[Agent] LLM returned content: {response.content}")
    if tool_call_count > 0:
        logger.info(f"[Agent] LLM requested tool calls: {json.dumps(response.tool_calls, default=str)}")
        
    return {"messages": [response]}


# ── Node: tool executor ───────────────────────────────────────────────────────

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def tool_node(state: AgentState) -> dict:
    last_message    = state["messages"][-1]
    tool_results    = []
    incident_id     = state.get("incident_id", "")
    tool_calls_made = list(state.get("tool_calls_made", []))
    rca_signals     = dict(state.get("rca_signals", {}))
    investigation_state = dict(state.get("investigation_state", {}))
    evidence        = list(state.get("evidence", []))
    investigation_findings = list(state.get("investigation_findings", []))

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id   = tc["id"]

        logger.info(f"[ToolNode] {tool_name} | args={json.dumps(tool_args)}")

        # Update DB status at key investigation milestones
        if incident_id:
            from app.processor.process_incident import _update_status
            _STATUS_MAP = {
                "get_ec2_analysis":             ("infra_analysis", WORKFLOW_STATES["infra_analysis"]),
                "get_compressed_logs":          ("logs_analysis",  WORKFLOW_STATES["logs_analysis"]),
                "correlate_instances":          ("ai_reasoning",   WORKFLOW_STATES["ai_reasoning"]),
                "get_infra_events":             ("infra_analysis",  WORKFLOW_STATES["infra_analysis"]),
                "investigate_network_path":     ("infra_analysis",  WORKFLOW_STATES["infra_analysis"]),
                "check_cloudtrail_sg_changes":  ("infra_analysis",  WORKFLOW_STATES["infra_analysis"]),
            }
            if tool_name in _STATUS_MAP:
                status, pct = _STATUS_MAP[tool_name]
                _update_status(incident_id, status, pct)

        if tool_name == "resolve_incident_targets" and state.get("resolved_targets"):
            logger.info("[ToolNode] Skipping resolve_incident_targets tool (using pre-resolved cache)")
            result = {
                "targets": state.get("resolved_targets", []),
                "alb_meta": state.get("alb_meta", {}),
                "_cached": True
            }
            result_content = json.dumps(result, default=str)
            # Accumulate RCA signals just in case, though they were pre-extracted already
            _accumulate_alb_signals(result, rca_signals)
        else:
            tool_fn = _TOOL_MAP.get(tool_name)
            if tool_fn is None:
                result_content = json.dumps({"error": f"Unknown tool: {tool_name}"})
            else:
                try:
                    result         = tool_fn.invoke(tool_args)
                    result_content = json.dumps(result, default=str)

                    # Accumulate RCA signals from tool results
                    if tool_name == "get_ec2_analysis":
                        _accumulate_ec2_signals(result, rca_signals)
                    elif tool_name == "get_compressed_logs":
                        _accumulate_log_signals(result, rca_signals)
                    elif tool_name == "resolve_incident_targets":
                        _accumulate_alb_signals(result, rca_signals)
                    elif tool_name == "investigate_network_path":
                        _accumulate_network_signals(result, rca_signals)
                    elif tool_name == "check_cloudtrail_sg_changes":
                        _accumulate_cloudtrail_sg_signals(result, rca_signals)

                    # Accumulate generic findings
                    for finding in result.get("findings", []):
                        if "message" in finding:
                            evidence.append(finding["message"])
                        investigation_findings.append(finding)
                        if "category" in finding:
                            investigation_state[f"{finding['category']}_validated"] = True

                except Exception as exc:
                    logger.error(f"[ToolNode] {tool_name} raised: {exc}")
                    result_content = json.dumps({"error": str(exc)})

        logger.info(f"[ToolNode] {tool_name} result: {result_content}...")
        logger.info(f"[ToolNode] {tool_name} full result sent to LLM: {result_content}")
        tool_results.append(ToolMessage(content=result_content, tool_call_id=tool_id))
        tool_calls_made.append(tool_name)

    return {
        "messages":        tool_results,
        "tool_calls_made": tool_calls_made,
        "rca_signals":     rca_signals,
        "investigation_state": investigation_state,
        "evidence":        evidence,
        "investigation_findings": investigation_findings,
    }


def _accumulate_ec2_signals(result: dict, rca_signals: dict):
    for iid, data in result.get("instances", {}).items():
        state = data.get("details", {}).get("state")
        if state:
            new = extract_rca_signals([], ec2_state=state)
            _merge_signals(rca_signals, new)


def _accumulate_log_signals(result: dict, rca_signals: dict):
    samples = []
    for e in result.get("top_errors", []):
        if isinstance(e, str):
            samples.append(e)
        elif isinstance(e, dict):
            sample = e.get("sample")
            if sample:
                samples.append(sample)
    for group_data in result.get("group_summaries", {}).values():
        for stage_data in group_data.values():
            for cluster in stage_data.get("top_clusters", []):
                if cluster.get("sample"):
                    samples.append(cluster["sample"])
    if samples:
        new = extract_rca_signals(samples)
        _merge_signals(rca_signals, new)


def _accumulate_alb_signals(result: dict, rca_signals: dict):
    reasons = [
        t.get("target_reason", "") for t in result.get("targets", [])
        if t.get("target_reason")
    ]
    if reasons:
        new = extract_rca_signals([], alb_target_reasons=reasons)
        _merge_signals(rca_signals, new)


def _accumulate_network_signals(result: dict, rca_signals: dict):
    """Extract RCA signals from investigate_network_path result."""
    blocked_layer = result.get("blocked_layer")
    blocked_port  = result.get("blocked_port")
    summary       = result.get("summary", "")

    if not blocked_layer:
        return  # network path clear — no new primary signal

    rca_type_map = {
        "security_group": "sg_blocked_port",
        "nacl":           "nacl_blocked_port",
        "route_table":    "route_table_missing",
    }
    description_map = {
        "security_group": f"Security Group outbound rule blocks TCP port {blocked_port}.",
        "nacl":           f"Network ACL denies outbound traffic to port {blocked_port}.",
        "route_table":    "Route table missing default route — subnet cannot reach external hosts.",
    }

    rca_type    = rca_type_map.get(blocked_layer, "network_path_blocked")
    description = description_map.get(blocked_layer, summary)

    new_signal = {
        "primary_root_cause": {
            "type":        "primary_root_cause",
            "rca_type":    rca_type,
            "description": description,
            "confidence":  0.90,
            "source":      "network_path_investigation",
            "evidence":    summary[:300],
        },
        "supporting_signals":    [],
        "infra_symptoms":        [],
        "downstream_impacts":    [],
        "all_signals": [{
            "type":        "primary_root_cause",
            "rca_type":    rca_type,
            "description": description,
            "confidence":  0.90,
            "source":      "network_path_investigation",
            "evidence":    summary[:300],
        }],
        "has_strong_signal": True,
    }
    _merge_signals(rca_signals, new_signal)


def _accumulate_cloudtrail_sg_signals(result: dict, rca_signals: dict):
    """Upgrade confidence when CloudTrail confirms SG change before incident."""
    probable_cause = result.get("probable_cause")
    changes        = result.get("changes", [])

    if not probable_cause or not changes:
        return

    # Find the most recent change before the incident
    before_changes = [c for c in changes if c.get("minutes_before_incident", -1) > 0]
    if not before_changes:
        return

    latest = before_changes[0]
    high_conf_signal = {
        "primary_root_cause": {
            "type":        "primary_root_cause",
            "rca_type":    "sg_change_caused_outage",
            "description": probable_cause,
            "confidence":  0.97,
            "source":      "cloudtrail_sg_audit",
            "evidence":    probable_cause[:300],
        },
        "supporting_signals":    [],
        "infra_symptoms":        [],
        "downstream_impacts":    [],
        "all_signals": [{
            "type":        "primary_root_cause",
            "rca_type":    "sg_change_caused_outage",
            "description": probable_cause,
            "confidence":  0.97,
            "source":      "cloudtrail_sg_audit",
            "evidence":    probable_cause[:300],
        }],
        "has_strong_signal": True,
    }
    _merge_signals(rca_signals, high_conf_signal)


def _merge_signals(existing: dict, new: dict):
    if not existing:
        existing.update(new)
        return

    seen = {s["rca_type"]: s for s in existing.get("all_signals", [])}
    for s in new.get("all_signals", []):
        key = s["rca_type"]
        if key not in seen or s["confidence"] > seen[key]["confidence"]:
            seen[key] = s
    all_sigs = list(seen.values())
    existing["all_signals"]        = all_sigs
    existing["supporting_signals"] = [s for s in all_sigs if s["type"] == "supporting_signal"]
    existing["infra_symptoms"]     = [s for s in all_sigs if s["type"] == "infra_symptom"]
    existing["downstream_impacts"] = [s for s in all_sigs if s["type"] == "downstream_impact"]

    primaries = [s for s in all_sigs if s["type"] == "primary_root_cause"]
    if primaries:
        best = max(primaries, key=lambda x: x["confidence"])
        existing["primary_root_cause"]    = best
        existing["has_deterministic_rca"] = best["confidence"] >= 0.85
    else:
        existing.setdefault("primary_root_cause",    None)
        existing.setdefault("has_deterministic_rca", False)


# ── Routing ───────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        rca_signals     = state.get("rca_signals", {})
        investigation_state = state.get("investigation_state", {})
        evidence        = state.get("evidence", [])

        candidate_rca   = rca_signals.get("primary_root_cause")
        
        stop, reason    = has_sufficient_evidence(candidate_rca, evidence, investigation_state)
        if stop:
            logger.info(f"[StopCondition] Early stop — {reason}")
            return "__end__"
        return "tools"

    return "__end__"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph():
    llm = _build_llm_with_tools()

    def _agent_node(state: AgentState):
        return agent_node(state, llm)

    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── Phase 2: structured extraction ───────────────────────────────────────────

def _extract_structured_result(
    conversation_messages: list,
    accumulated_signals: dict,
    investigation_findings: list[dict],
    incident_id: str,
) -> dict | None:
    """
    Phase 2: take the full conversation (context + tool results) and ask a
    fresh LLM call with structured output to produce the guaranteed-schema JSON.

    This avoids all text-parsing failures — Bedrock's native tool-calling
    mechanism forces the model to emit a valid JSON object matching RCA_SCHEMA.
    """
    # 1. Find the final AI message from the investigation
    last_ai_message = next(
        (m for m in reversed(conversation_messages) if m.type == "ai" and hasattr(m, "content") and isinstance(m.content, str)),
        None
    )
    final_agent_output = last_ai_message.content if last_ai_message else ""

    # 2. Build fresh extraction conversation without tool history
    sys_msg = SystemMessage(
        content=(
            "You are an RCA extraction assistant. Based on the provided investigation details, "
            "produce the final root cause analysis in the required JSON schema. "
            "evidence[] must contain direct quotes, not paraphrases. "
            "recommended_actions[] must have exactly 3 items: fix, verify, prevent.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST prioritize Investigation Findings over raw log symptoms.\n"
            "2. Do not generate generic RCA such as 'Database unreachable', 'Service unavailable', or 'Connection timeout' if deterministic investigation findings exist.\n"
            "3. When investigation findings identify a failed infrastructure component (e.g., 'Security group blocks outbound 5432'), that component is the RCA.\n"
            "4. Confidence Calculation: Increase confidence when deterministic findings exist. For example, if only a log symptom exists (e.g. Postgres timeout), confidence is 40-60%. If a network blockage is found, confidence is 90-95%. If network blockage + CloudTrail correlation is found, confidence is 95-99%."
        )
    )

    findings_text = json.dumps([f.get("message") for f in investigation_findings], indent=2) if investigation_findings else "[]"

    human_msg = HumanMessage(
        content=f"""
Incident ID: {incident_id}

Investigation Findings:
{findings_text}

Accumulated Signals:
{json.dumps(accumulated_signals)}

Final Agent Reasoning:
{final_agent_output}

Return valid JSON only matching the RCA_SCHEMA.
"""
    )

    messages_for_extraction = [sys_msg, human_msg]

    logger.info("[Extraction] Starting fresh extraction conversation")
    logger.info(f"[Extraction] Fresh message count={len(messages_for_extraction)}")

    try:
        llm_structured = _build_llm_structured()
        result = safe_llm_invoke(llm_structured, messages_for_extraction)

        if isinstance(result, dict):
            logger.info("[Extraction] Structured output success")
            return _normalise_structured(result)

        logger.warning("[Extraction] with_structured_output returned non-dict, falling back")
    except Exception as exc:
        logger.warning(f"[Extraction] Structured output failed: {exc}")

    # ── Fallback 1: try plain JSON parse of last AI message ───────────────────
    for msg in reversed(conversation_messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            raw = _parse_json_text(msg.content)
            if raw and isinstance(raw, dict) and "probable_root_cause" in raw:
                logger.info("[Extraction] Recovered JSON from last AI message text")
                return _normalise_structured(raw)

    # ── Fallback 2: build from strong signals if JSON extraction completely fails ──
    # Note: This is a post-investigation recovery mechanism and not a replacement
    # for the LLM's investigation. It only runs if the LLM finishes its tools
    # but completely fails to generate valid structured JSON output twice.
    if accumulated_signals.get("has_strong_signal"):
        logger.warning("[Extraction] Using strong-signal fallback due to LLM extraction failure")
        return _signals_to_structured(accumulated_signals, incident_id)

    return None


def _normalise_structured(raw: dict) -> dict:
    """Normalise / coerce field types to match the expected schema."""
    confidence = raw.get("confidence", 50)
    try:
        confidence = int(float(confidence))
        # Accept 0.0-1.0 float (multiply up to int)
        if confidence <= 1:
            confidence = int(confidence * 100)
        confidence = max(0, min(100, confidence))
    except (TypeError, ValueError):
        confidence = 50

    return {
        "probable_root_cause": str(raw.get("probable_root_cause") or raw.get("root_cause") or ""),
        "confidence":          confidence,
        "evidence":            [str(e) for e in (raw.get("evidence") or [])],
        "dependency_impact":   [str(d) for d in (raw.get("dependency_impact") or [])],
        "recommended_actions": [str(a) for a in (raw.get("recommended_actions") or [])],
    }


def _parse_json_text(text: str) -> dict | None:
    """Try to extract a JSON block from text (last-resort text parser)."""
    try:
        if "```json" in text:
            block = text.split("```json")[1].split("```")[0].strip()
            return json.loads(block)
        stripped = text.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
    except (json.JSONDecodeError, IndexError):
        pass
    return None


def _signals_to_structured(signals: dict, incident_id: str) -> dict:
    """Build a minimal structured result from pre-extracted signals."""
    primary  = signals.get("primary_root_cause") or {}
    desc     = primary.get("description", "Root cause could not be determined.")
    conf     = int(primary.get("confidence", 0.4) * 100)
    rca_type = primary.get("rca_type", "unknown")

    evidence = [primary.get("evidence", "")] if primary.get("evidence") else []
    for sig in signals.get("supporting_signals", [])[:2]:
        if sig.get("evidence"):
            evidence.append(sig["evidence"])

    return {
        "probable_root_cause": desc,
        "confidence":          conf,
        "evidence":            evidence,
        "dependency_impact":   [
            s["description"] for s in signals.get("downstream_impacts", [])
        ],
        "recommended_actions": _default_actions_for(rca_type),
    }


def _default_actions_for(rca_type: str) -> list[str]:
    _MAP = {
        "database_config_error":      [
            "Correct the DATABASE_URL / DSN environment variable on the instance.",
            "Verify DB connectivity: psql <corrected-dsn> and check for successful connection.",
            "Add DSN format validation to the application startup health check.",
        ],
        "instance_stopped":           [
            "Start the EC2 instance via: aws ec2 start-instances --instance-ids <iid>",
            "Confirm ALB target returns to healthy state within 60 seconds.",
            "Enable CloudWatch alarm to detect stopped instances automatically.",
        ],
        "iam_permission_error":       [
            "Attach the missing IAM policy to the instance or task role.",
            "Verify the fix: re-run the failing API call and confirm no AccessDenied.",
            "Implement least-privilege IAM reviews as part of deployment checklist.",
        ],
        "oom_kill":                   [
            "Restart the application service: sudo systemctl restart <service>",
            "Confirm memory usage drops below 80% after restart.",
            "Increase instance type or tune application heap/memory limits.",
        ],
        "disk_full":                  [
            "Free disk space: sudo du -sh /* | sort -rh | head -20, then remove stale files.",
            "Confirm disk usage drops below 80%: df -h",
            "Set up a CloudWatch disk-space alarm at 80% threshold.",
        ],
    }
    return _MAP.get(rca_type, [
        f"Investigate and remediate {rca_type}.",
        "Verify the fix by confirming service health.",
        "Add monitoring to detect recurrence early.",
    ])


# ── Initial HumanMessage builder ──────────────────────────────────────────────

def _build_initial_human_message(
    incident_context: dict,
    triage_result:    dict | None,
    rca_signals:      dict | None,
    timeline:         dict | None,
    similar_incidents: list[dict],
    resolved_targets:  list[dict] | None = None,
) -> str:
    lines = [
        "## Incident Investigation Context",
        "",
        f"**Incident ID:** {incident_context.get('incident_id')}",
        f"**Monitor:** {incident_context.get('monitor_name')} ({incident_context.get('monitor_type')})",
        f"**Down Message:** {incident_context.get('down_message')}",
        f"**Down Time:** {incident_context.get('incident_down_time')}",
        f"**AWS Region:** {incident_context.get('aws_region')}",
        f"**Monitor URL:** {incident_context.get('monitor_url', 'N/A')}",
        "",
        "**Dependencies:**",
        json.dumps(incident_context.get("dependencies", []), indent=2),
        "",
    ]

    if rca_signals and rca_signals.get("primary_root_cause"):
        primary = rca_signals["primary_root_cause"]
        lines += [
            "## Pre-Extracted RCA Signals",
            f"**Primary Root Cause:** `{primary['rca_type']}` — {primary['description']}",
            f"**Confidence:** {primary['confidence']:.0%}  |  **Source:** {primary['source']}",
            f"**Evidence:** {primary.get('evidence', '')}",
            "> Treat this as your leading hypothesis. Confirm with tools.",
            "",
        ]

    if resolved_targets:
        lines += [
            "## Resolved Targets",
            "Resolved targets are already available in state.",
            "Do NOT call resolve_incident_targets unless resolved_targets is empty.",
            json.dumps(resolved_targets, indent=2),
            "",
        ]

    if triage_result:
        lines += [
            "## Investigation Hypothesis",
            "> This is an ALB-derived hypothesis only. Validate with additional evidence before concluding root cause.",
            f"**Likely Issue:** {triage_result.get('likely_issue')}",
            f"**Target Reason:** {triage_result.get('target_reason')}",
            f"**Affected Instances:** {', '.join(triage_result.get('affected_instances', []))}",
            f"priority_hints (skip_metrics={triage_result.get('skip_metrics')} | skip_logs={triage_result.get('skip_logs')}): If True, less likely to contain root cause but you MUST still investigate if evidence leads there.",
            "",
        ]

    if timeline and timeline.get("causal_chain"):
        lines += ["## Temporal Correlation", ""]
        for evt in timeline["causal_chain"]:
            mins = evt.get("minutes_before_incident", 0)
            lines.append(
                f"  [{evt['type'].upper()}] {evt['timestamp']} "
                f"({abs(mins):.1f} min {'before' if mins > 0 else 'at'} incident): {evt['event']}"
            )
        lines.append("")

    if similar_incidents:
        lines += ["## Similar Past Incidents", ""]
        for si in similar_incidents[:2]:
            lines.append(f"  - {si.get('monitor_name')}: {si.get('summary', '')[:100]}")
        lines.append("")

    lines += [
        "## Task",
        "",
        "Investigate using the available tools. Stop as soon as you have enough evidence.",
        "End your final message with ONLY the JSON block from your system prompt.",
    ]

    return "\n".join(lines)


# ── Public entry point ────────────────────────────────────────────────────────

def run_agent_investigation(
    incident_id:       str,
    incident_context:  dict,
    triage_result:     dict | None = None,
    rca_signals:       dict | None = None,
    timeline:          dict | None = None,
    similar_incidents: list[dict] | None = None,
    resolved_targets:  list[dict] | None = None,
    alb_meta:          dict | None = None,
) -> dict:
    """
    Run the investigation for a single incident.

    Returns dict with:
        incident_id, message_count, tool_call_count,
        structured_result (5-field schema).
    """
    logger.info(f"[Agent] Starting investigation for incident {incident_id}")

    similar_incidents = similar_incidents or []
    rca_signals       = rca_signals or {}

    # ── Phase 1: ReAct investigation loop ────────────────────────────────────
    graph = build_agent_graph()

    initial_human = _build_initial_human_message(
        incident_context, triage_result, rca_signals, timeline, similar_incidents, resolved_targets
    )

    initial_state: AgentState = {
        "messages":        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=initial_human)],
        "incident_id":     incident_id,
        "tool_calls_made": [],
        "rca_signals":     rca_signals,
        "resolved_targets": resolved_targets or [],
        "alb_meta":         alb_meta or {},
        "investigation_state": {
            "network_validated": False,
            "compute_validated": False,
            "dependency_validated": False,
            "configuration_validated": False,
            "change_history_validated": False
        },
        "evidence": [],
        "investigation_findings": [],
    }

    config      = RunnableConfig(recursion_limit=MAX_ITERATIONS * 2)
    final_state = graph.invoke(initial_state, config=config)

    tool_call_count = sum(
        len(m.tool_calls) if hasattr(m, "tool_calls") and m.tool_calls else 0
        for m in final_state["messages"]
    )

    accumulated_sigs = final_state.get("rca_signals", rca_signals)

    logger.info(
        f"[Agent] ReAct loop done | incident={incident_id} | "
        f"messages={len(final_state['messages'])} | tool_calls={tool_call_count}"
    )

    # ── Phase 2: structured extraction ───────────────────────────────────────
    structured_result = _extract_structured_result(
        list(final_state["messages"]),
        accumulated_sigs,
        final_state.get("investigation_findings", []),
        incident_id,
    )

    logger.info(
        f"[Agent] Complete | incident={incident_id} | "
        f"structured={'yes' if structured_result else 'no'}"
    )

    return {
        "incident_id":       incident_id,
        "message_count":     len(final_state["messages"]),
        "tool_call_count":   tool_call_count,
        "structured_result": structured_result,
    }