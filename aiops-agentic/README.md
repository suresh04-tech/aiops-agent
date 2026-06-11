# AIOps Agentic Incident Investigation & SOP Generation

This repository contains two core agentic workflows built with LangGraph, FastAPI, and Amazon Bedrock (Claude):
1. **Autonomous Incident Root Cause Analysis (RCA)** 
2. **Dynamic SOP / Runbook Generation**

---

## 1. Agentic Root Cause Analysis (RCA)

An autonomous AWS incident investigator powered by a LangGraph ReAct agent. Instead of relying on a static pipeline or a single massive prompt, the agent intelligently navigates through AWS infrastructure to find the root cause of an incident.

### How it works
- **Dynamic Investigation**: The agent is given an `incident_id` and decides which tools to call based on the evolving context.
- **Rich Toolset**: Equipped with multiple custom-built AWS tools to query EC2 state, CloudWatch metrics, CloudTrail logs, ALB targets, Security Group rules, and perform deep Network Path investigations.
- **Causal Reasoning**: The LLM analyzes the output of each tool, reasons about the failure path, and continues investigating until it can definitively prove the root cause.
- **Evidence-Backed**: The final output includes the probable root cause, confidence score, direct evidence quotes, dependency impacts, and exact remediation commands.

### Available Agent Tools
| Tool | Description |
|---|---|
| `get_incident_context` | Load initial incident details from the database |
| `resolve_incident_targets` | Map dependencies (EC2 / ALB / Domain) to concrete instance lists |
| `get_ec2_details` | Retrieve instance state, status checks, and tags |
| `get_ec2_metrics` | Fetch CPU, memory, disk, network, and status check metrics |
| `get_compressed_logs` | Extract and compress relevant logs using anchor-based search |
| `get_infra_events` | Query CloudTrail for deployments, IAM changes, and network updates |
| `investigate_network_path` | Analyze DNS, Route Tables, NACLs, and Security Groups for blocked traffic |
| `get_security_group_rules` | Fetch detailed inbound/outbound rules for a specific SG |
| `check_cloudtrail_sg_changes` | Audit who modified a Security Group and exactly what rules were changed |
| `get_alb_target_health` | Snapshot ALB target health status |
| `query_logs_insights` | Execute targeted CloudWatch Logs Insights queries |
| `correlate_instances` | Cross-instance comparison to detect shared vs. isolated failures |
| `update_investigation_status` | Push live progress updates to the frontend |
| `store_raw_evidence` | Save collected evidence to the database |
| `store_rca_result` | Write the final RCA conclusion and remediation steps |

---

## 2. Dynamic SOP / Runbook Generation

A dedicated pipeline for generating production-grade, Markdown-formatted Standard Operating Procedures (SOPs).

### Two Generation Modes

1. **Alert-Based Flow (Automated Context)**
   - **Trigger**: Receives an `alert_id`.
   - **Enrichment**: Automatically scans the database for historical incidents linked to similar alerts. 
   - **Context Merge**: Collects and merges historical Root Cause Analyses (RCAs), deep investigation findings, and raw evidence.
   - **Output**: Generates a highly tailored SOP incorporating known failure modes, verified remediation steps, and historical context—without hitting token limits.

2. **Prompt-Based Flow (Free-Form)**
   - **Trigger**: Receives a free-form user `prompt` describing their architecture and the issue.
   - **Guardrails**: Intercepts prompts using an intelligent 3-layer guardrail system (Off-Topic, Context-Length, and Infra-Relevance) to block invalid requests before incurring LLM costs.
   - **Output**: Generates a general, best-practice SOP matching the described technology stack.

---

## Architecture

Both workflows operate asynchronously using background workers and an in-process queue to ensure the FastAPI server remains non-blocking.

```text
POST /queue/enqueue (RCA)  ──► queue_manager     ──► process_incident() ──► LangGraph ReAct Loop
POST /sop/enqueue   (SOP)  ──► sop_queue_manager ──► process_sop()      ──► Bedrock LLM Generation
```

### Project Structure

```text
app/
├── agent/                # LangGraph ReAct agent and RCA tools
├── api/                  # FastAPI routes (/queue, /sop)
├── processor/            # RCA orchestrator and specialized infra processors
├── sop/                  # SOP generator, prompt guardrails, and context loaders
├── queue/                # Asyncio queue managers
├── utils/                # DB and AWS clients
└── main.py               # FastAPI application entry point
```

---

## Setup & Execution

### 1. Requirements & Configuration
```bash
pip install -r requirements.txt
cp .env.example .env
# Configure DB credentials and AWS settings in .env
```

### 2. Run the Service
```bash
docker compose up
# Or locally:
uvicorn app.main:app --reload
```

### 3. Trigger Jobs
**Trigger an RCA Investigation:**
```bash
curl -X POST http://localhost:8000/queue/enqueue \
  -H "Content-Type: application/json" \
  -d '{"incident_id": "your-incident-id"}'
```

**Trigger an Alert-based SOP:**
```bash
curl -X POST http://localhost:8000/sop/enqueue \
  -H "Content-Type: application/json" \
  -d '{"sop_id": "SOP-123", "alert_id": "your-alert-id"}'
```

**Trigger a Prompt-based SOP:**
```bash
curl -X POST http://localhost:8000/sop/enqueue \
  -H "Content-Type: application/json" \
  -d '{"sop_id": "SOP-124", "prompt": "We run a Python FastAPI app on ECS..."}'
```

### 4. Monitor
Check queue depth and service health:
```bash
curl http://localhost:8000/health
```
Logs are automatically routed and rotated:
- `logs/aiops.log` (RCA and general operations)
- `logs/sop.log` (Dedicated SOP generation logs)
