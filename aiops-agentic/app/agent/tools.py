"""
app/agent/tools.py
──────────────────
All AWS investigation tools exposed to the LangGraph agent.

Design principles
─────────────────
• Every tool is a plain Python function decorated with @tool.
• Tools accept minimal, specific inputs — NOT giant blobs of context.
• Tools return structured dicts that the agent can reason over.
• Tools are NEVER called with hardcoded data — the agent decides when and
  how to call them based on what it has observed so far.
• All AWS calls go through AWSClientFactory (reads creds from DB connector).
• Errors are caught and returned as {"error": "..."} so the agent can
  decide to retry, pivot, or conclude gracefully.

The agent receives these tools and decides which to call next after
reading the incident context — no predetermined pipeline.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Tool registry helper ──────────────────────────────────────────────────────
# The factory and DB references are injected at runtime, not at import time.
# This lets tools work regardless of import order.

_aws_factory = None
_incident_row = None


def init_tools(aws_factory, incident_row: dict):
    """Call once per incident before running the agent."""
    global _aws_factory, _incident_row
    _aws_factory = aws_factory
    _incident_row = incident_row


def _factory():
    if _aws_factory is None:
        raise RuntimeError("Tools not initialised — call init_tools() first")
    return _aws_factory


def _incident():
    if _incident_row is None:
        raise RuntimeError("Tools not initialised — call init_tools() first")
    return _incident_row


def _region() -> str:
    return _incident().get("region") or "ap-south-1"


def _safe(fn, *args, **kwargs) -> dict:
    """Wrap any boto3 call — return error dict instead of raising."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning(f"[Tool error] {fn.__name__ if hasattr(fn,'__name__') else '?'}: {exc}")
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Load incident context from DB
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_incident_context() -> dict:
    """
    Return the full incident record: issue, severity, dependencies,
    incident_down_time, dependency_context, connector_id.

    Call this FIRST to understand what you are investigating.
    """
    row = _incident()
    return {
        "incident_id":      str(row.get("id", "")),
        "issue":            row.get("issue", ""),
        "severity":         row.get("severity", "medium"),
        "incident_down_time": str(row.get("incident_down_time", "")),
        "dependencies":     row.get("dependencies", []),
        "dependency_context": row.get("dependency_context", {}),
        "connector_id":     str(row.get("connector_id", "")),
        "service_name":     row.get("service_name", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Resolve dependencies (EC2 / ALB → EC2 instance list)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def resolve_incident_targets() -> dict:
    """
    Resolve the incident dependencies into concrete EC2 instance targets.

    Handles both direct EC2 (type=ec2) and ALB (type=alb) dependencies.
    For ALB: discovers all registered targets and their health status.

    Returns a list of targets and ALB metadata if applicable.
    Always call this before any EC2/CloudWatch tool.
    """
    from app.processor.dependency_resolver import resolve_dependencies

    row      = _incident()
    raw_deps = row.get("dependencies") or []
    if isinstance(raw_deps, str):
        try:
            raw_deps = json.loads(raw_deps)
        except Exception:
            raw_deps = []

    # Normalise legacy schema
    normalised = []
    default_region = _region()
    for d in raw_deps:
        if "type" not in d:
            d = {
                "type":           "ec2",
                "resource_id":    d.get("instance_id", ""),
                "region":         d.get("region", default_region),
                "log_group_name": d.get("log_group_name") or d.get("log_group_names") or [],
            }
        normalised.append(d)

    if not normalised:
        return {"error": "No dependencies found in incident record"}

    def _resolve():
        targets, alb_meta = resolve_dependencies(normalised, _factory())
        return {"targets": targets, "alb_meta": alb_meta}

    return _safe(_resolve)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — EC2 instance details + status checks
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_ec2_details(instance_id: str, region: str = "") -> dict:
    """
    Describe an EC2 instance: state, type, AZ, status checks (ok / impaired),
    security groups, tags, launch time.

    Args:
        instance_id: EC2 instance ID e.g. 'i-0abc1234'
        region: AWS region (uses incident region if omitted)
    """
    region = region or _region()
    ec2    = _factory().get_client("ec2", region_name=region)

    def _run():
        details, status = {}, {}

        resp = ec2.describe_instances(InstanceIds=[instance_id])
        ress = resp.get("Reservations", [])
        if ress:
            inst = ress[0]["Instances"][0]
            details = {
                "instance_id":       inst.get("InstanceId"),
                "instance_type":     inst.get("InstanceType"),
                "state":             inst.get("State", {}).get("Name"),
                "private_ip":        inst.get("PrivateIpAddress"),
                "public_ip":         inst.get("PublicIpAddress"),
                "availability_zone": inst.get("Placement", {}).get("AvailabilityZone"),
                "launch_time":       str(inst.get("LaunchTime", "")),
                "ami_id":            inst.get("ImageId"),
                "security_groups": [
                    {"id": sg["GroupId"], "name": sg["GroupName"]}
                    for sg in inst.get("SecurityGroups", [])
                ],
                "tags": {t["Key"]: t["Value"] for t in inst.get("Tags", [])},
            }

        resp2 = ec2.describe_instance_status(
            InstanceIds=[instance_id], IncludeAllInstances=True
        )
        stats = resp2.get("InstanceStatuses", [])
        if stats:
            s = stats[0]
            status = {
                "instance_status": s.get("InstanceStatus", {}).get("Status"),
                "system_status":   s.get("SystemStatus", {}).get("Status"),
            }

        return {"details": details, "status_checks": status}

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — CloudWatch metrics for an EC2 instance
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_ec2_metrics(instance_id: str, region: str = "",
                    window_minutes: int = 15) -> dict:
    """
    Fetch key CloudWatch metrics for an EC2 instance over the last N minutes:
    CPUUtilization, NetworkIn/Out, DiskReadOps/WriteOps, StatusCheckFailed.

    Args:
        instance_id: EC2 instance ID
        region: AWS region (uses incident region if omitted)
        window_minutes: how many minutes back to look (default 15)
    """
    region = region or _region()
    cw     = _factory().get_client("cloudwatch", region_name=region)

    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=window_minutes)

    def _metric(name: str, stat: str = "Average") -> float | None:
        resp = cw.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName=name,
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=[stat],
        )
        pts = sorted(resp.get("Datapoints", []),
                     key=lambda x: x["Timestamp"], reverse=True)
        return pts[0].get(stat) if pts else None

    def _run():
        return {
            "instance_id":        instance_id,
            "window_minutes":     window_minutes,
            "cpu_percent":        _metric("CPUUtilization"),
            "network_in_bytes":   _metric("NetworkIn"),
            "network_out_bytes":  _metric("NetworkOut"),
            "disk_read_ops":      _metric("DiskReadOps"),
            "disk_write_ops":     _metric("DiskWriteOps"),
            "status_check_failed": _metric("StatusCheckFailed", "Sum"),
        }

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — Fetch and compress CloudWatch logs (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_compressed_logs(log_groups: list[str], region: str = "") -> dict:
    """
    Run the full log analysis pipeline (Phase A-E) for the given log groups:
    adaptive windowing, first-error anchoring, 3-stage fetch,
    weighted compression, cascade attribution.

    Returns structured per-stage summaries and top error signals.

    Args:
        log_groups: list of CloudWatch log group names
        region: AWS region (uses incident region if omitted)
    """
    from app.processor.log_processor import fetch_and_compress_logs

    row    = _incident()
    region = region or _region()

    down_time_raw = row.get("incident_down_time")
    if not down_time_raw:
        return {"error": "incident_down_time missing from incident record"}

    if isinstance(down_time_raw, str):
        down_time = datetime.fromisoformat(down_time_raw.replace("Z", "+00:00"))
    elif isinstance(down_time_raw, datetime):
        down_time = down_time_raw if down_time_raw.tzinfo else down_time_raw.replace(tzinfo=timezone.utc)
    else:
        return {"error": f"Unrecognised incident_down_time type: {type(down_time_raw)}"}

    logs_client = _factory().get_client("logs", region_name=region)

    def _run():
        return fetch_and_compress_logs(
            logs_client=logs_client,
            log_groups=[g for g in log_groups if g],
            incident_down_time=down_time,
            severity=row.get("severity", "medium"),
            issue=row.get("issue", ""),
            dependency_context=row.get("dependency_context") or {},
            status_callback=None,
        )

    result = _safe(_run)

    # Return a trimmed version so the agent context doesn't overflow
    if "error" in result:
        return result

    return {
        "adaptive_window":  result.get("adaptive_window"),
        "anchor":           result.get("anchor"),
        "top_errors":       result.get("top_errors", [])[:10],
        "total_raw_lines":  result.get("total_raw_lines", 0),
        "stage_summaries": {
            group: {
                stage: {
                    "stage_label": data.get("stage_label"),
                    "window":      data.get("window"),
                    "total_raw":   data.get("total_raw", 0),
                    "error_count": data.get("error_count", 0),
                    "warn_count":  data.get("warn_count", 0),
                    "top_clusters": [
                        {
                            "level":        c["level"],
                            "weight_label": c.get("weight_label"),
                            "count":        c["count"],
                            "is_rare":      c.get("is_rare", False),
                            "cascade_suspect": c.get("cascade_suspect", False),
                            "upstream_services": c.get("upstream_services", []),
                            "sample":       (c["samples"][0] if c["samples"] else c["fingerprint"])[:300],
                        }
                        for c in data.get("clusters", [])[:8]
                    ],
                }
                for stage, data in stages.items()
            }
            for group, stages in result.get("per_group", {}).items()
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — Cross-instance correlation
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def correlate_instances(instance_analyses_json: str) -> dict:
    """
    Run the correlation engine across multiple instance analyses to detect
    failure scenario (A/B/C/D), identify the primary suspect instance,
    and classify errors as common (shared dep) vs isolated (host-level).

    Args:
        instance_analyses_json: JSON string of {instance_id: analysis_dict}
            where each analysis_dict has keys: ec2, metrics, log_summary,
            top_errors, total_error_count, target_health, target_reason.
    """
    from app.processor.correlation_engine import correlate_instances as _correlate

    try:
        instance_analyses = json.loads(instance_analyses_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    def _run():
        alb_meta = _incident().get("_alb_meta_cache", {})
        return _correlate(instance_analyses, alb_meta)

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — CloudTrail infrastructure events
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_infra_events(instance_id: str, region: str = "") -> dict:
    """
    Fetch CloudTrail infrastructure change events around the incident window:
    deployments, scaling, IAM changes, network changes, DB modifications.

    Returns high-risk events ranked by proximity to the first log error,
    root cause hypotheses based on timing, and failed API calls.

    Args:
        instance_id: EC2 instance ID (used for context/logging)
        region: AWS region
    """
    from app.processor.cloudtrail_processor import fetch_infra_context, format_infra_context_for_prompt

    row    = _incident()
    region = region or _region()

    down_time_raw = row.get("incident_down_time")
    if not down_time_raw:
        return {"error": "incident_down_time missing"}

    if isinstance(down_time_raw, str):
        down_time = datetime.fromisoformat(down_time_raw.replace("Z", "+00:00"))
    elif isinstance(down_time_raw, datetime):
        down_time = down_time_raw if down_time_raw.tzinfo else down_time_raw.replace(tzinfo=timezone.utc)
    else:
        return {"error": "Unrecognised incident_down_time type"}

    ct_client = _factory().get_client("cloudtrail", region_name=region)

    def _run():
        ctx = fetch_infra_context(
            cloudtrail_client=ct_client,
            region=region,
            instance_id=instance_id,
            down_time=down_time,
            anchor={},   # no log anchor yet — agent may call this before logs
            severity=row.get("severity", "medium"),
            issue=row.get("issue", ""),
        )
        return {
            "total_events":     ctx.get("total_events", 0),
            "high_risk_events": ctx.get("high_risk_events", [])[:10],
            "hypotheses":       ctx.get("hypotheses", [])[:5],
            "failed_api_calls": ctx.get("failed_api_calls", []),
            "by_user":          ctx.get("by_user", {}),
            "formatted":        format_infra_context_for_prompt(ctx),
        }

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 8 — ALB target health (standalone, without full resolver)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_alb_target_health(alb_dns_or_arn: str, region: str = "") -> dict:
    """
    Check ALB target health for all registered targets.
    Use this when you need a quick health snapshot without full dependency resolution.

    Args:
        alb_dns_or_arn: ALB DNS name or ARN
        region: AWS region
    """
    region  = region or _region()
    elbv2   = _factory().get_client("elbv2", region_name=region)

    def _run():
        # Resolve DNS → ARN if needed
        if not alb_dns_or_arn.startswith("arn:"):
            dns = alb_dns_or_arn.lower().replace("http://","").replace("https://","").strip("/")
            pag = elbv2.get_paginator("describe_load_balancers")
            alb_arn = None
            for page in pag.paginate():
                for lb in page.get("LoadBalancers", []):
                    if lb.get("DNSName", "").lower() == dns:
                        alb_arn = lb["LoadBalancerArn"]
                        break
                if alb_arn:
                    break
            if not alb_arn:
                return {"error": f"No ALB found for DNS: {alb_dns_or_arn}"}
        else:
            alb_arn = alb_dns_or_arn

        # Get target groups
        tg_resp = elbv2.describe_target_groups(LoadBalancerArn=alb_arn)
        results = []
        for tg in tg_resp.get("TargetGroups", []):
            health = elbv2.describe_target_health(TargetGroupArn=tg["TargetGroupArn"])
            for thd in health.get("TargetHealthDescriptions", []):
                t  = thd.get("Target", {})
                th = thd.get("TargetHealth", {})
                results.append({
                    "instance_id": t.get("Id"),
                    "port":        t.get("Port"),
                    "state":       th.get("State"),
                    "reason":      th.get("Reason", ""),
                    "description": th.get("Description", ""),
                    "tg_name":     tg.get("TargetGroupName"),
                })

        healthy   = [r for r in results if r["state"] == "healthy"]
        unhealthy = [r for r in results if r["state"] != "healthy"]
        return {
            "alb_arn":         alb_arn,
            "total_targets":   len(results),
            "healthy_count":   len(healthy),
            "unhealthy_count": len(unhealthy),
            "targets":         results,
        }

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 9 — Specific log query (Logs Insights)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def query_logs_insights(log_groups: list[str], query: str,
                        lookback_minutes: int = 30, region: str = "") -> dict:
    """
    Run a CloudWatch Logs Insights query to drill into specific patterns.
    Use this to follow up on top errors found in get_compressed_logs.

    Example queries:
        'fields @timestamp, @message | filter @message like /timeout/ | sort @timestamp desc | limit 20'
        'stats count(*) as cnt by bin(5m) | sort cnt desc'

    Args:
        log_groups: list of log group names
        query: Logs Insights query string
        lookback_minutes: time window in minutes (default 30)
        region: AWS region
    """
    import time
    region      = region or _region()
    logs_client = _factory().get_client("logs", region_name=region)

    row = _incident()
    down_time_raw = row.get("incident_down_time")
    if isinstance(down_time_raw, str):
        ref = datetime.fromisoformat(down_time_raw.replace("Z", "+00:00"))
    elif isinstance(down_time_raw, datetime):
        ref = down_time_raw if down_time_raw.tzinfo else down_time_raw.replace(tzinfo=timezone.utc)
    else:
        ref = datetime.now(timezone.utc)

    end_ts   = int(ref.timestamp()) + 600          # +10 min after incident
    start_ts = end_ts - lookback_minutes * 60

    def _run():
        resp     = logs_client.start_query(
            logGroupNames=log_groups,
            startTime=start_ts,
            endTime=end_ts,
            queryString=query,
            limit=50,
        )
        qid = resp["queryId"]
        for _ in range(30):
            time.sleep(1)
            res = logs_client.get_query_results(queryId=qid)
            if res["status"] in ("Complete", "Failed", "Cancelled"):
                break

        rows = res.get("results", [])
        return {
            "query":      query,
            "status":     res["status"],
            "row_count":  len(rows),
            "results": [
                {field["field"]: field["value"] for field in row}
                for row in rows[:20]
            ],
        }

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 10 — Update investigation progress in DB
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def update_investigation_status(status: str, percent: int) -> dict:
    """
    Update the incident analysis_status and analysis_percent in the DB.
    Use this at key milestones so the frontend can show live progress.

    Valid status values:
        resolving_deps, fetching_ec2, fetching_metrics, fetching_logs,
        correlating, building_rca, completed, failed

    Args:
        status: status string
        percent: completion percentage 0-100
    """
    from app.utils.db import get_db

    row = _incident()
    incident_id = str(row.get("id", ""))

    def _run():
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meyiconnect.insight_incidents
                    SET analysis_status = %s,
                        analysis_percent = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, percent, incident_id),
                )
        return {"updated": True, "status": status, "percent": percent}

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 11 — Store RCA result back to DB
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def store_rca_result(
    root_cause: str,
    rca_report_json: str,
    remediation_json: str,
    confidence_score: float,
    ai_model_used: str = "langgraph-agent",
) -> dict:
    """
    Write the final RCA and remediation to the incident record in DB.
    Call this as the LAST step after all analysis is complete.

    Args:
        root_cause: one-paragraph root cause statement
        rca_report_json: JSON string with full rca_report dict
        remediation_json: JSON string with remediation_steps dict
        confidence_score: float 0.0–1.0
        ai_model_used: label for which model/agent produced this
    """
    from app.utils.db import get_db

    row         = _incident()
    incident_id = str(row.get("id", ""))

    # Validate JSON inputs
    try:
        json.loads(rca_report_json)
        json.loads(remediation_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in rca or remediation: {e}"}

    confidence_pct = round(float(confidence_score) * 100, 2)

    def _run():
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meyiconnect.insight_incidents
                    SET
                        analysis_status       = 'completed',
                        analysis_percent      = 100,
                        analysis_result       = %s,
                        remediation_steps     = %s,
                        confidence_score      = %s,
                        ai_model_used         = %s,
                        analysis_completed_at = NOW(),
                        updated_at            = NOW()
                    WHERE id = %s
                    """,
                    (
                        rca_report_json,
                        remediation_json,
                        str(confidence_pct),
                        ai_model_used,
                        incident_id,
                    ),
                )
                logger.info(f"[StoreRCA] Updated {cur.rowcount} row(s) for incident {incident_id}")
            conn.commit()

        return {
            "stored": True,
            "incident_id": incident_id,
            "confidence_pct": confidence_pct,
        }

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 12 — Store raw collected data (incident_logs table)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def store_raw_evidence(
    primary_instance_id: str,
    ec2_details_json: str,
    metrics_json: str,
    raw_logs_json: str,
    logs_count: int,
) -> dict:
    """
    Store the raw EC2, metrics, and log data in incident_logs table.
    Call this after gathering all evidence and before calling store_rca_result.

    Args:
        primary_instance_id: the primary suspect instance ID
        ec2_details_json: JSON of EC2 details dict
        metrics_json: JSON of metrics dict
        raw_logs_json: JSON of log summary dict
        logs_count: total raw log lines fetched
    """
    from app.utils.db import get_db

    row         = _incident()
    incident_id = str(row.get("id", ""))

    def _run():
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO meyiconnect.incident_logs (
                        incident_id, ec2_details, ec2_status_checks,
                        cloudwatch_metrics, raw_logs, logs_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (incident_id) DO UPDATE
                        SET cloudwatch_metrics = EXCLUDED.cloudwatch_metrics,
                            raw_logs           = EXCLUDED.raw_logs,
                            logs_count         = EXCLUDED.logs_count,
                            updated_at         = NOW()
                    RETURNING id
                    """,
                    (
                        incident_id,
                        ec2_details_json,
                        "{}",          # status_checks merged into ec2_details
                        metrics_json,
                        raw_logs_json,
                        logs_count,
                    ),
                )
                log_id = (cur.fetchone() or {}).get("id")
            conn.commit()
        return {"stored": True, "log_id": log_id}

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool list exported for the agent
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    get_incident_context,
    resolve_incident_targets,
    get_ec2_details,
    get_ec2_metrics,
    get_compressed_logs,
    correlate_instances,
    get_infra_events,
    get_alb_target_health,
    query_logs_insights,
    update_investigation_status,
    store_rca_result,
    store_raw_evidence,
]
