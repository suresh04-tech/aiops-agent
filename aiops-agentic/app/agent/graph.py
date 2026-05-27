"""
app/agent/graph.py
──────────────────
LangGraph ReAct agent for autonomous AWS incident investigation.

Architecture
────────────
  ┌─────────┐     ┌──────────────┐     ┌──────────────┐
  │  START  │────▶│  agent_node  │────▶│  tool_node   │
  └─────────┘     │  (LLM)       │◀────│  (AWS calls) │
                  └──────┬───────┘     └──────────────┘
                         │ (no more tool calls)
                         ▼
                      ┌──────┐
                      │  END │
                      └──────┘

The agent node runs the LLM with ALL tools available.
The LLM decides which tool(s) to call next.
The tool node executes the tool and returns results.
This loop continues until the LLM produces a final answer (no tool calls).

Key design decisions
────────────────────
• We use Claude 3 Sonnet via Bedrock (same infra you already use).
• The agent state carries the full message history — the LLM always sees
  what it has already observed.
• Max iterations is set high (30) because thorough investigation beats speed.
• Tool calls are always sequential per iteration (no parallel tool calls in
  the base ReAct pattern), but the LLM can call multiple tools across turns.
"""

import json
import logging
import os
from typing import Annotated, Literal, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from app.agent.tools import ALL_TOOLS
from app.agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BEDROCK_MODEL    = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
BEDROCK_REGION   = os.environ.get("AWS_REGION", "ap-south-1")
MAX_ITERATIONS   = int(os.environ.get("AGENT_MAX_ITERATIONS", "30"))


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── Build the LLM with tools bound ───────────────────────────────────────────

def _build_llm(aws_factory=None):
    """
    Build a ChatBedrock LLM with all investigation tools bound.
    Uses the AWSClientFactory credentials if provided, else env vars.
    """
    kwargs = {
        "model_id":    BEDROCK_MODEL,
        "region_name": BEDROCK_REGION,
        "model_kwargs": {
            "max_tokens":  8192,
            "temperature": 0.1,
            "top_p":       0.9,
        },
    }

    # If we have a factory with explicit credentials, use them
    if aws_factory and hasattr(aws_factory, "access_key"):
        import boto3
        session = boto3.Session(
            aws_access_key_id=aws_factory.access_key,
            aws_secret_access_key=aws_factory.secret_key,
            region_name=aws_factory.region or BEDROCK_REGION,
        )
        kwargs["client"] = session.client("bedrock-runtime", region_name=BEDROCK_REGION)

    llm = ChatBedrock(**kwargs)
    return llm.bind_tools(ALL_TOOLS)


# ── Node: agent (LLM decides next action) ─────────────────────────────────────

def agent_node(state: AgentState, llm_with_tools) -> AgentState:
    """
    Run the LLM on the current message history.
    The LLM either calls tool(s) or produces its final answer.
    """
    response = llm_with_tools.invoke(state["messages"])
    logger.info(
        f"[Agent] turn complete | "
        f"tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}"
    )
    return {"messages": [response]}


# ── Node: tool executor ───────────────────────────────────────────────────────

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def tool_node(state: AgentState) -> AgentState:
    """
    Execute all tool calls from the last AI message.
    Returns ToolMessage results back into the message history.
    """
    last_message = state["messages"][-1]
    tool_results = []

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id   = tc["id"]

        logger.info(f"[ToolNode] Calling {tool_name} | args={json.dumps(tool_args)[:200]}")

        tool_fn = _TOOL_MAP.get(tool_name)
        if tool_fn is None:
            result_content = json.dumps({"error": f"Unknown tool: {tool_name}"})
        else:
            try:
                result = tool_fn.invoke(tool_args)
                result_content = json.dumps(result, default=str)
            except Exception as exc:
                logger.error(f"[ToolNode] {tool_name} raised: {exc}")
                result_content = json.dumps({"error": str(exc)})

        logger.info(f"[ToolNode] {tool_name} result: {result_content[:300]}")

        tool_results.append(
            ToolMessage(content=result_content, tool_call_id=tool_id)
        )

    return {"messages": tool_results}


# ── Routing: continue loop or end ─────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """If the last AI message has tool calls → run tools. Otherwise → done."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph(aws_factory=None):
    """
    Build and compile the LangGraph agent.
    Call once per incident (aws_factory carries per-incident credentials).
    """
    llm = _build_llm(aws_factory)

    # Bind the LLM into the node via closure
    def _agent_node(state: AgentState):
        return agent_node(state, llm)

    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public entry: run the full investigation ──────────────────────────────────

def run_agent_investigation(incident_id: str, aws_factory) -> dict:
    """
    Run the autonomous investigation agent for a single incident.

    This is called by process_incident.py instead of the old Bedrock single-prompt approach.

    Returns the final state with the full message history.
    The agent itself stores results in DB via store_rca_result tool.
    """
    logger.info(f"[Agent] Starting agentic investigation for incident {incident_id}")

    graph = build_agent_graph(aws_factory)

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Start investigating incident ID: {incident_id}\n\n"
                    f"Follow the investigation protocol. Call get_incident_context() first "
                    f"to load the full incident details from the database.\n\n"
                    f"Be thorough. Use all available tools. "
                    f"Do not conclude until you have checked EC2, metrics, logs, and CloudTrail.\n"
                    f"Store the final RCA using store_rca_result() before finishing."
                )
            ),
        ]
    }

    config = RunnableConfig(recursion_limit=MAX_ITERATIONS * 2)

    final_state = graph.invoke(initial_state, config=config)

    # Count tool calls made
    tool_call_count = sum(
        len(m.tool_calls) if hasattr(m, "tool_calls") and m.tool_calls else 0
        for m in final_state["messages"]
    )

    logger.info(
        f"[Agent] Investigation complete for {incident_id} | "
        f"messages={len(final_state['messages'])} | "
        f"tool_calls={tool_call_count}"
    )

    return {
        "incident_id":     incident_id,
        "message_count":   len(final_state["messages"]),
        "tool_call_count": tool_call_count,
        "final_message":   final_state["messages"][-1].content if final_state["messages"] else "",
    }
