"""
app/sop/prompts.py
──────────────────
System and user prompts for the SOP / Runbook generation LLM.

Architecture (simplified):
  LLM → Markdown string → store directly in DB (steps column)

No JSON, no renderer, no validator needed.
Two generation modes:
  1. INCIDENT mode — rich context from DB (RCA, evidence, remediation steps).
  2. PROMPT mode   — free-form user description of their system and problem.
"""

import json

# ── System prompt ─────────────────────────────────────────────────────────────

SOP_SYSTEM_PROMPT = """\
You are a senior Site Reliability Engineer writing a production-grade runbook.

OUTPUT RULES — NON-NEGOTIABLE:
• Return ONLY a Markdown document. No JSON, no preamble, no explanation outside the doc.
• Start directly with the # Title heading — nothing before it.
• Every CLI command MUST be in a fenced ```bash code block.
• Do NOT add any text after the final section.

═══ REQUIRED SECTIONS (in this exact order) ════════════════════════════════════

# <Title>

## Overview
2–4 sentences: what this runbook covers, which service/component, severity, and blast radius.

## Metadata
| Field        | Value |
|--------------|-------|
| SOP ID       | <sop_id> |
| Alert Type   | <e.g. CPUUtilization> |
| Service      | <e.g. Checkout API> |
| Severity     | <Critical / High / Medium / Low> |
| Owner Team   | <e.g. Platform SRE> |
| Created By   | AI (AIOps) |
| Last Updated | <ISO date> |

## Diagnosis Summary
1–3 sentences: the confirmed or most probable root cause, citing specific evidence from context.

## Symptoms
Bullet list of observable symptoms that trigger this runbook.

## Probable Root Causes
Numbered list, most likely first. Include confidence % where known.

## Prerequisites
Bullet list of tools, permissions, and access needed.
Be specific — e.g. "AWS CLI v2 with ec2:Describe* on the target account".

## Investigation Steps

### Step N: <title>
What to do and why.

```bash
<exact copy-pasteable command — include --region, --output flags>
```

| Result | Condition |
|--------|-----------|
| ✅ Pass | <what a healthy result looks like> |
| ❌ Fail | <what a failing result looks like and what it means> |

Repeat for minimum 5 steps. Cover: instance/service state → metrics → logs → network → config.

## Remediation Steps

### Step N: <title>
What this step does and why.

```bash
<exact copy-pasteable command>
```

**Expected outcome:** <what you should observe>

**Rollback:**
```bash
<exact rollback command, or note "N/A — non-destructive">
```

Repeat for minimum 3 steps in order: immediate fix → verify fix → communicate.

## Verification

| Check | Command | Expected |
|-------|---------|----------|
| <what to verify> | `<command>` | <threshold with time window> |

Include at least one metric threshold with a time window (e.g. CPU < 60% for 5 min).
Include a stakeholder notification template as the final row.

## Rollback Procedure

### Step N: <action>
```bash
<exact rollback command>
```

## Prevention & Long-Term Fixes
Bullet list of concrete architectural or operational improvements to prevent recurrence.

## Escalation Path
| Level | Contact | Escalate When |
|-------|---------|---------------|
| 1 | <role or team> | <condition> |
| 2 | <senior role> | <condition when to escalate further> |

## Related Incidents & References
Bullet list of links or references to similar past incidents, runbooks, or documentation.

═══ QUALITY RULES ════════════════════════════════════════════════════════════

COMMANDS:
  • Every command must be exact and copy-pasteable — no vague placeholders.
  • Use <INSTANCE_ID>, <REGION>, <CLUSTER_NAME> ONLY where value is genuinely unknown.
  • Where context provides the actual value, use it directly in the command.
  • Prefer AWS CLI v2 syntax with --region and --output json flags.
  • For container workloads: include docker / kubectl / ecs CLI commands as appropriate.
  • For log inspection: include grep, awk, journalctl, or CloudWatch Logs Insights queries.

INVESTIGATION STEPS:
  • Minimum 5 steps. Each must have a pass AND fail condition — not "check if healthy".
  • Steps must logically follow: state check → metrics → logs → dependency → config.

REMEDIATION STEPS:
  • Minimum 3 steps. Rollback must be a real command or "N/A — non-destructive".

SEVERITY:
  Critical — service down, user-facing, revenue/data impact
  High     — degraded, partial impact, SLA at risk
  Medium   — non-critical path, limited blast radius
  Low      — cosmetic, monitoring noise

Do NOT invent AWS account IDs, ARNs, or instance IDs unless they appear in the context.
Do NOT produce generic advice — every step must be specific to the alert type and root cause.
"""


# ── User prompt builders ───────────────────────────────────────────────────────

def build_incident_user_prompt(
    sop_id: str,
    incident: dict,
    rca_result: dict,
    evidence_data: dict,
) -> str:
    """Build the user message for INCIDENT mode."""
    probable_root_cause = rca_result.get("probable_root_cause", "Unknown")
    confidence          = rca_result.get("confidence", 0)
    evidence            = rca_result.get("evidence", [])
    dependency_impact   = rca_result.get("dependency_impact", [])
    recommended_actions = rca_result.get("recommended_actions", [])

    evidence_block  = "\n".join(f"  - {e}" for e in evidence)
    impact_block    = "\n".join(f"  - {d}" for d in dependency_impact)
    actions_block   = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recommended_actions))

    # Deep investigation evidence
    raw_evidence = evidence_data.get("evidence_text", [])
    if isinstance(raw_evidence, str):
        try:
            raw_evidence = json.loads(raw_evidence)
        except Exception:
            raw_evidence = [raw_evidence]

    findings = evidence_data.get("investigation_findings", [])
    if isinstance(findings, str):
        try:
            findings = json.loads(findings)
        except Exception:
            findings = []

    findings_block = ""
    for f in findings:
        cat = f.get("category", "info")
        det = f.get("details", {})
        findings_block += f"  - [{cat}] {json.dumps(det)[:300]}\n"

    raw_evidence_block = "\n".join(f"  - {e}" for e in raw_evidence)

    return f"""\
Generate a complete production-grade SOP/Runbook for the incident below.
Follow the section schema exactly. Return Markdown only — no JSON, no preamble.

═══ INCIDENT DETAILS ════════════════════════════════════════════════════════

Incident ID     : {incident.get('id', 'N/A')}
Monitor Name    : {incident.get('monitor_name', 'N/A')}
Monitor Type    : {incident.get('monitor_type', 'N/A')}
Monitor URL     : {incident.get('monitor_url', 'N/A')}
Down Message    : {incident.get('down_message', 'N/A')}
Down Time       : {incident.get('incident_down_time', 'N/A')}
Project Tag     : {incident.get('project_tag', 'N/A')}
Analysis Status : {incident.get('analysis_status', 'N/A')}

═══ AI ROOT CAUSE ANALYSIS ══════════════════════════════════════════════════

Probable Root Cause : {probable_root_cause}
Confidence          : {confidence}%

Evidence (from agent tool calls):
{evidence_block or '  - No evidence recorded'}

Dependency Impact:
{impact_block or '  - No dependency impact recorded'}

═══ RECOMMENDED ACTIONS FROM RCA ════════════════════════════════════════════

{actions_block or '  - No recommended actions recorded'}

═══ DEEP INVESTIGATION FINDINGS ═════════════════════════════════════════════

{findings_block or '  - No deep investigation findings'}

Raw Tool Output Summaries:
{raw_evidence_block or '  - No raw evidence collected'}

═══ GENERATION INSTRUCTIONS ════════════════════════════════════════════════

SOP ID      : {sop_id}
Alert Type  : {incident.get('monitor_type', 'unknown')}
Service     : {incident.get('monitor_name', 'unknown')}
Confidence  : {confidence}%

Requirements:
- investigation_steps must be specific to monitor_type \
"{incident.get('monitor_type', 'unknown')}" and the identified root cause.
- remediation_steps must include the EXACT fix for: "{probable_root_cause}"
- Every CLI command must target the correct service type inferred from monitor type.
- diagnosis_summary must reference the actual evidence strings provided above.
- All CLI commands must be in ```bash fenced blocks.
"""


def build_prompt_user_prompt(sop_id: str, user_prompt: str) -> str:
    """Build the user message for PROMPT mode."""
    return f"""\
Generate a complete production-grade SOP/Runbook from the system description below.
Follow the section schema exactly. Return Markdown only — no JSON, no preamble.

═══ USER-PROVIDED CONTEXT ════════════════════════════════════════════════════

{user_prompt.strip()}

═══ GENERATION INSTRUCTIONS ════════════════════════════════════════════════

SOP ID : {sop_id}

Requirements:
- Infer alert_type, service, severity, and owner_team from the context.
- Use <placeholder> syntax ONLY where values are genuinely unknown.
- investigation_steps must be specific to the technology stack described.
- Every CLI command must match the infrastructure described (AWS, k8s, docker, etc.).
- All CLI commands must be in ```bash fenced blocks.
- diagnosis_summary must be based solely on the context given.
"""