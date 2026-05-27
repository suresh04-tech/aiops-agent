# AIOps Agentic Incident Investigation

Autonomous AWS incident root cause analysis using a LangGraph ReAct agent with Bedrock Claude.

## What changed from the MVP

| | MVP (before) | Agentic (now) |
|--|--|--|
| **AI approach** | One giant prompt → one Bedrock call | ReAct agent decides what to investigate next |
| **Investigation** | Fixed pipeline, always same steps | Adaptive — agent adjusts based on evidence |
| **Accuracy** | LLM overwhelmed by massive context | LLM sees focused, relevant data at each step |
| **Transparency** | Black box single response | Full tool call history logged per incident |
| **Extensibility** | Add to prompt template | Add a new `@tool` function |

## Architecture

```
POST /queue/enqueue {"incident_id": "123"}
       │
       ▼
  QueueManager (asyncio.Queue)
       │
       ▼
  Worker (ThreadPoolExecutor)
       │
       ▼
  process_incident(payload)
       │
       ├─ Load incident from DB
       ├─ Build AWSClientFactory (creds from connector)
       ├─ init_tools(factory, incident_row)
       │
       └─ run_agent_investigation()
              │
              ▼
         LangGraph ReAct Loop
         ┌──────────────────────────────────────────────────┐
         │  agent_node (Claude via Bedrock)                  │
         │    ↓ decides which tool to call                   │
         │  tool_node executes ONE tool                      │
         │    ↓ returns result to agent                      │
         │  agent_node observes result, reasons, decides...  │
         │    ↓ loop until agent calls store_rca_result()    │
         └──────────────────────────────────────────────────┘
              │
              └─ Results stored in DB by agent itself
```

## Available agent tools

| Tool                                  | What it does                                                |
|---------------------------------------|-------------------------------------------------------------|
| `get_incident_context`                | Load incident details from DB                               |
| `resolve_incident_targets`            | EC2 / ALB → concrete instance list                          |
| `get_ec2_details`                     | Instance state, status checks, tags                         |
| `get_ec2_metrics`                     | CPU, disk, network, status_check_failed                     |
| `get_compressed_logs`                 | Full log pipeline: anchor → 3-stage → weighted compression  |
| `get_infra_events`                    | CloudTrail: deployments, IAM, network changes               |
| `get_alb_target_health`               | ALB target health snapshot                                  |
| `query_logs_insights`                 | Drill into specific log patterns                            |
| `correlate_instances`                 | Cross-instance comparison and scenario detection            |
| `update_investigation_status`         | Live progress updates to frontend                           |
| `store_raw_evidence`                  | Write EC2/metrics/logs to incident_logs table               |
| `store_rca_result`                    | Write final RCA and remediation to incident table           |

## Project structure

```
app/
├── agent/
│   ├── tools.py          ← 12 AWS investigation tools
│   ├── graph.py          ← LangGraph ReAct agent
│   └── prompts.py        ← Agent system prompt
├── processor/
│   ├── process_incident.py  ← Entry point (now just boots the agent)
│   ├── worker.py            ← Unchanged
│   ├── dependency_resolver.py  ← Unchanged (called via tool)
│   ├── log_processor.py        ← Unchanged (called via tool)
│   ├── correlation_engine.py   ← Unchanged (called via tool)
│   └── Cloudtrail_processor.py ← Unchanged (called via tool)
├── queue/
│   └── manager.py        ← Unchanged
├── api/
│   └── routes/queue.py   ← Unchanged
├── utils/
│   ├── aws_connector.py  ← Unchanged
│   └── db.py             ← Unchanged
└── main.py               ← Unchanged
```

## Setup

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your DB and AWS credentials

# 3. Run
docker compose up
# or locally:
uvicorn app.main:app --reload

# 4. Trigger an investigation
curl -X POST http://localhost:8000/queue/enqueue \
  -H "Content-Type: application/json" \
  -d '{"incident_id": "your-incident-id"}'

# 5. Monitor progress
curl http://localhost:8000/health
```

## Model selection

Best results with Claude 3 Sonnet or Claude 3.5 Sonnet (supports tool calling + long reasoning).
Set `BEDROCK_MODEL_ID` in `.env`:

```
# Best accuracy
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# Good balance (default)
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Fastest (less thorough)
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
```

## DB schema (unchanged)

The agent reads from and writes to the same tables as before:
- `meyiconnect.insight_incidents` — incident record with analysis results
- `meyiconnect.incident_logs`     — raw EC2/metrics/log evidence
- `meyiconnect.insight_connectors` — AWS credentials per connector
