"""
app/agent/tools.py
──────────────────
AWS investigation tools for the LangGraph agent (v3).

Changes in v3
─────────────
P2 — RCA signal extractor with deterministic root-cause rules.
     Classifies signals as: primary_root_cause / supporting_signal /
     infra_symptom / downstream_impact.
     Rules: invalid DSN, AccessDenied, OOM, connection refused, EC2 stopped, etc.

P3 — LLM output schema tightened.  The agent is now asked for a compact
     Pydantic-style JSON (probable_root_cause, confidence 0-100, summary,
     dependency_impact[], recommended_actions[]).

P6 — Investigation stop-condition helper: has_sufficient_evidence() is
     called by graph.py to short-circuit the reasoning loop early.

P7 — Temporal correlator: correlate_timeline() links deployment/config
     events → app errors → ALB degradation in a causal chain.

P8 — Similar incident lookup: find_similar_incidents() queries the
     incident_logs table for past RCAs with overlapping signals.

P9 — Signal classifier taxonomy embedded in extract_rca_signals().
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Tool registry helper ──────────────────────────────────────────────────────

_aws_factory  = None
_incident_row = None


def init_tools(aws_factory, incident_row: dict):
    """Call once per incident before running the agent."""
    global _aws_factory, _incident_row
    _aws_factory  = aws_factory
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
    return _incident().get("aws_region") or "ap-south-1"


def _safe(fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning(f"[Tool error] {fn.__name__ if hasattr(fn, '__name__') else '?'}: {exc}")
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Resolve dependencies
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def resolve_incident_targets() -> dict:
    """
    Resolve incident dependencies into concrete EC2 instance targets.
    Handles EC2 (type=ec2) and ALB (type=alb) dependencies.
    Returns targets list and ALB metadata.
    Always call this first.
    """
    from app.processor.dependency_resolver import resolve_dependencies

    row      = _incident()
    raw_deps = row.get("dependencies") or []
    if isinstance(raw_deps, str):
        try:
            raw_deps = json.loads(raw_deps)
        except Exception:
            raw_deps = []

    normalised     = []
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
# TOOL 2 — EC2 analysis (details + metrics, batched)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_ec2_analysis(instance_ids: list[str], region: str = "",
                     window_minutes: int = 15) -> dict:
    """
    Fetch EC2 details AND CloudWatch metrics for one or more instances in one
    tool call.  Returns a dict keyed by instance_id with:
      - details  (state, type, AZ, IPs, SGs, tags)
      - status_checks  (instance_status, system_status)
      - metrics  (CPU, network, disk, status_check_failed)

    Args:
        instance_ids: e.g. ['i-0abc1234', 'i-0def5678']
        region: AWS region (uses incident region if omitted)
        window_minutes: CloudWatch lookback (default 15)
    """
    region = region or _region()
    ec2    = _factory().get_client("ec2",        region_name=region)
    cw     = _factory().get_client("cloudwatch", region_name=region)

    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=window_minutes)

    def _metric(iid: str, name: str, stat: str = "Average") -> float | None:
        resp = cw.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName=name,
            Dimensions=[{"Name": "InstanceId", "Value": iid}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=[stat],
        )
        pts = sorted(resp.get("Datapoints", []),
                     key=lambda x: x["Timestamp"], reverse=True)
        return pts[0].get(stat) if pts else None

    def _run():
        results = {}

        resp = ec2.describe_instances(InstanceIds=instance_ids)
        instance_map = {}
        for res in resp.get("Reservations", []):
            for inst in res.get("Instances", []):
                instance_map[inst.get("InstanceId")] = inst

        resp2 = ec2.describe_instance_status(
            InstanceIds=instance_ids, IncludeAllInstances=True
        )
        status_map = {}
        for s in resp2.get("InstanceStatuses", []):
            iid = s.get("InstanceId")
            status_map[iid] = {
                "instance_status": s.get("InstanceStatus", {}).get("Status"),
                "system_status":   s.get("SystemStatus",   {}).get("Status"),
            }

        for iid in instance_ids:
            inst    = instance_map.get(iid, {})
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
            } if inst else {"error": f"Instance {iid} not found"}

            metrics = {
                "cpu_percent":         _metric(iid, "CPUUtilization"),
                "network_in_bytes":    _metric(iid, "NetworkIn"),
                "network_out_bytes":   _metric(iid, "NetworkOut"),
                "disk_read_ops":       _metric(iid, "DiskReadOps"),
                "disk_write_ops":      _metric(iid, "DiskWriteOps"),
                "status_check_failed": _metric(iid, "StatusCheckFailed", "Sum"),
                "window_minutes":      window_minutes,
            }

            results[iid] = {
                "details":       details,
                "status_checks": status_map.get(iid, {}),
                "metrics":       metrics,
            }

        return {"instances": results}

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Fetch and compress CloudWatch logs
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_compressed_logs(log_groups: list[str], region: str = "") -> dict:
    """
    Run the full log analysis pipeline for the given log groups.
    Returns compact summaries with top error signals (trimmed payload).

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
            severity="medium",
            issue=row.get("down_message", ""),
            dependency_context={},
            status_callback=None,
        )

    result = _safe(_run)
    if "error" in result:
        return result

    group_summaries = {}
    for group, stages in result.get("per_group", {}).items():
        group_summaries[group] = {
            stage: {
                "stage_label": data.get("stage_label"),
                "window":      data.get("window"),
                "total_raw":   data.get("total_raw", 0),
                "error_count": data.get("error_count", 0),
                "warn_count":  data.get("warn_count", 0),
                "top_clusters": [
                    {
                        "level":             c["level"],
                        "count":             c["count"],
                        "weight_label":      c.get("weight_label"),
                        "is_rare":           c.get("is_rare", False),
                        "cascade_suspect":   c.get("cascade_suspect", False),
                        "upstream_services": c.get("upstream_services", []),
                        "sample": (
                            c["samples"][0] if c.get("samples")
                            else c.get("fingerprint", "")
                        )[:200],
                    }
                    for c in data.get("clusters", [])[:5]
                ],
            }
            for stage, data in stages.items()
        }

    return {
        "adaptive_window": result.get("adaptive_window"),
        "anchor":          result.get("anchor"),
        "top_errors":      result.get("top_errors", [])[:8],
        "total_raw_lines": result.get("total_raw_lines", 0),
        "group_summaries": group_summaries,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Cross-instance correlation
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def correlate_instances(instance_analyses_json: str) -> dict:
    """
    Run the correlation engine across multiple instance analyses.
    Detects failure scenario (A/B/C/D), primary suspect instance,
    and classifies errors as common (shared dep) vs isolated (host-level).

    Args:
        instance_analyses_json: JSON string of {instance_id: analysis_dict}
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
# TOOL 5 — CloudTrail infrastructure events
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_infra_events(instance_id: str, region: str = "") -> dict:
    """
    Fetch CloudTrail infrastructure change events around the incident window.
    Returns high-risk events, hypotheses, and failed API calls (compact payload).

    Args:
        instance_id: EC2 instance ID
        region: AWS region
    """
    from app.processor.cloudtrail_processor import fetch_infra_context

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
            anchor={},
            severity="medium",
            issue=row.get("down_message", ""),
        )
        return {
            "total_events":     ctx.get("total_events", 0),
            "high_risk_events": ctx.get("high_risk_events", [])[:8],
            "hypotheses":       ctx.get("hypotheses", [])[:5],
            "failed_api_calls": ctx.get("failed_api_calls", [])[:5],
            "by_user":          ctx.get("by_user", {}),
        }

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — ALB target health
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_alb_target_health(alb_dns_or_arn: str, region: str = "") -> dict:
    """
    Check ALB target health for all registered targets.

    Args:
        alb_dns_or_arn: ALB DNS name or ARN
        region: AWS region
    """
    region = region or _region()
    elbv2  = _factory().get_client("elbv2", region_name=region)

    def _run():
        if not alb_dns_or_arn.startswith("arn:"):
            dns = alb_dns_or_arn.lower().replace("http://", "").replace("https://", "").strip("/")
            pag     = elbv2.get_paginator("describe_load_balancers")
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
# TOOL 7 — Logs Insights drill-down
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def query_logs_insights(log_groups: list[str], query: str,
                        lookback_minutes: int = 30, region: str = "") -> dict:
    """
    Run a CloudWatch Logs Insights query for specific patterns.

    Args:
        log_groups: list of log group names
        query: Logs Insights query string
        lookback_minutes: time window in minutes (default 30)
        region: AWS region
    """
    import time as _time
    region      = region or _region()
    logs_client = _factory().get_client("logs", region_name=region)

    row           = _incident()
    down_time_raw = row.get("incident_down_time")
    if isinstance(down_time_raw, str):
        ref = datetime.fromisoformat(down_time_raw.replace("Z", "+00:00"))
    elif isinstance(down_time_raw, datetime):
        ref = down_time_raw if down_time_raw.tzinfo else down_time_raw.replace(tzinfo=timezone.utc)
    else:
        ref = datetime.now(timezone.utc)

    end_ts   = int(ref.timestamp()) + 600
    start_ts = end_ts - lookback_minutes * 60

    def _run():
        resp = logs_client.start_query(
            logGroupNames=log_groups,
            startTime=start_ts,
            endTime=end_ts,
            queryString=query,
            limit=50,
        )
        qid = resp["queryId"]
        res = {}
        for _ in range(30):
            _time.sleep(1)
            res = logs_client.get_query_results(queryId=qid)
            if res["status"] in ("Complete", "Failed", "Cancelled"):
                break

        rows = res.get("results", [])
        return {
            "query":     query,
            "status":    res.get("status"),
            "row_count": len(rows),
            "results": [
                {field["field"]: field["value"] for field in row}
                for row in rows[:20]
            ],
        }

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# P2 — RCA signal extractor (extracts hints for the LLM)
# ═══════════════════════════════════════════════════════════════════════════════

# Signal taxonomy (P9)
# Each rule maps a pattern to:
#   type        — signal type: primary_root_cause / supporting_signal /
#                              infra_symptom / downstream_impact
#   rca_type    — machine-readable root cause class
#   description — human-readable explanation
#   confidence  — float 0-1 for this rule alone

_LOG_SIGNAL_RULES: list[dict] = [
    # ── Primary root causes (application/config) ──────────────────────────────
    {
        "pattern": re.compile(r"invalid\s+dsn", re.I),
        "type": "primary_root_cause",
        "rca_type": "database_config_error",
        "description": "Invalid DSN — database connection string is malformed or wrong.",
        "confidence": 0.93,
    },
    {
        "pattern": re.compile(r"could not connect to server|connection refused.*5432|psycopg2.*OperationalError", re.I),
        "type": "primary_root_cause",
        "rca_type": "database_unreachable",
        "description": "Database connection refused — DB host is down or port blocked.",
        "confidence": 0.88,
    },
    {
        "pattern": re.compile(r"AccessDenied|UnauthorizedOperation|not authorized to perform", re.I),
        "type": "primary_root_cause",
        "rca_type": "iam_permission_error",
        "description": "IAM AccessDenied — missing permissions for a required AWS API call.",
        "confidence": 0.91,
    },
    {
        "pattern": re.compile(r"OutOfMemoryError|OOMKilled|killed.*memory|out of memory", re.I),
        "type": "primary_root_cause",
        "rca_type": "oom_kill",
        "description": "OOM kill — process exceeded memory limits.",
        "confidence": 0.90,
    },
    {
        "pattern": re.compile(r"NoCredentialProviders|Unable to locate credentials|ExpiredTokenException", re.I),
        "type": "primary_root_cause",
        "rca_type": "aws_credentials_error",
        "description": "AWS credentials missing, expired, or not found on the instance.",
        "confidence": 0.92,
    },
    {
        "pattern": re.compile(r"ECONNREFUSED|connection reset by peer|broken pipe", re.I),
        "type": "primary_root_cause",
        "rca_type": "upstream_connection_refused",
        "description": "Upstream service refused the connection — likely crashed or overloaded.",
        "confidence": 0.80,
    },
    {
        "pattern": re.compile(r"disk quota exceeded|no space left on device|ENOSPC", re.I),
        "type": "primary_root_cause",
        "rca_type": "disk_full",
        "description": "Disk full — no space left on device.",
        "confidence": 0.95,
    },
    {
        "pattern": re.compile(r"segfault|segmentation fault|core dumped|SIGSEGV", re.I),
        "type": "primary_root_cause",
        "rca_type": "process_crash_segfault",
        "description": "Process segfault / core dump — application crash at OS level.",
        "confidence": 0.88,
    },
    # ── Supporting signals (confirm root cause, not the cause itself) ──────────
    {
        "pattern": re.compile(r"HTTP 5[0-9]{2}|status\s*5\d\d|error 50[0-9]", re.I),
        "type": "supporting_signal",
        "rca_type": "http_5xx_errors",
        "description": "HTTP 5xx errors — application returning server errors.",
        "confidence": 0.60,
    },
    {
        "pattern": re.compile(r"timeout after|read timeout|connect timed out|deadline exceeded", re.I),
        "type": "supporting_signal",
        "rca_type": "timeout_errors",
        "description": "Timeout errors — requests not completing within expected window.",
        "confidence": 0.60,
    },
    {
        "pattern": re.compile(r"retry attempt|retrying|max retries exceeded", re.I),
        "type": "supporting_signal",
        "rca_type": "retry_storm",
        "description": "Retry storm — clients or app repeatedly retrying failing calls.",
        "confidence": 0.55,
    },
    # ── Infra symptoms (infrastructure-level indicators) ──────────────────────
    {
        "pattern": re.compile(r"health check failed|healthcheck.*fail|ELB.*unhealthy", re.I),
        "type": "infra_symptom",
        "rca_type": "alb_health_check_fail",
        "description": "ALB health check failing — instance not responding on health port.",
        "confidence": 0.65,
    },
    {
        "pattern": re.compile(r"status check.*fail|instance.*impaired|system.*impaired", re.I),
        "type": "infra_symptom",
        "rca_type": "ec2_status_check_fail",
        "description": "EC2 status check failure — hardware or OS-level issue.",
        "confidence": 0.70,
    },
    # ── Downstream impact (cascade effects, not root cause) ───────────────────
    {
        "pattern": re.compile(r"circuit breaker open|circuit.*open", re.I),
        "type": "downstream_impact",
        "rca_type": "circuit_breaker_open",
        "description": "Circuit breaker opened — downstream services protecting themselves.",
        "confidence": 0.50,
    },
    {
        "pattern": re.compile(r"queue.*full|message queue.*overflow|backpressure", re.I),
        "type": "downstream_impact",
        "rca_type": "queue_overflow",
        "description": "Queue overflow — downstream consumers can't keep up.",
        "confidence": 0.55,
    },
]

# EC2 state rules (applied to instance state string)
_EC2_STATE_RULES: dict[str, dict] = {
    "stopped": {
        "type": "primary_root_cause",
        "rca_type": "instance_stopped",
        "description": "EC2 instance is stopped — not running behind the ALB.",
        "confidence": 0.97,
    },
    "terminated": {
        "type": "primary_root_cause",
        "rca_type": "instance_terminated",
        "description": "EC2 instance has been terminated.",
        "confidence": 0.97,
    },
    "stopping": {
        "type": "primary_root_cause",
        "rca_type": "instance_stopping",
        "description": "EC2 instance is in the process of stopping.",
        "confidence": 0.90,
    },
    "pending": {
        "type": "infra_symptom",
        "rca_type": "instance_pending",
        "description": "EC2 instance is still pending — not yet ready to serve traffic.",
        "confidence": 0.75,
    },
}

# ALB target reason rules
_ALB_REASON_RULES: dict[str, dict] = {
    "Target.InvalidState": {
        "type": "primary_root_cause",
        "rca_type": "alb_target_invalid_state",
        "description": "ALB target in invalid state — instance is likely stopped or terminated.",
        "confidence": 0.95,
    },
    "Target.FailedHealthChecks": {
        "type": "infra_symptom",
        "rca_type": "alb_target_health_check_fail",
        "description": "ALB target failing health checks — app not responding on health endpoint.",
        "confidence": 0.70,
    },
    "Target.NotInUse": {
        "type": "downstream_impact",
        "rca_type": "alb_target_not_in_use",
        "description": "ALB target deregistered or draining — likely a deployment event.",
        "confidence": 0.80,
    },
    "Target.DeregistrationInProgress": {
        "type": "downstream_impact",
        "rca_type": "alb_target_deregistering",
        "description": "ALB target deregistration in progress — rolling deploy or scale-in.",
        "confidence": 0.75,
    },
}


def extract_rca_signals(
    log_samples: list[str],
    ec2_state: str | None = None,
    alb_target_reasons: list[str] | None = None,
) -> dict:
    """
    RCA signal extractor (P2 + P9).

    Scans log samples, EC2 state, and ALB target reasons against rule tables.
    Returns classified signals with the highest-confidence primary root cause
    promoted to the top.

    This is called by Python (not the LLM) and the result is embedded into
    the initial investigation context so the agent starts with pre-classified
    signals.

    Returns:
    {
      "primary_root_cause": { rca_type, description, confidence, source } | None,
      "supporting_signals":  [...],
      "infra_symptoms":      [...],
      "downstream_impacts":  [...],
      "all_signals":         [...],
      "has_deterministic_rca": bool,
    }
    """
    all_signals: list[dict] = []

    # ── Scan log samples against rules ───────────────────────────────────────
    for sample in log_samples:
        for rule in _LOG_SIGNAL_RULES:
            if rule["pattern"].search(sample):
                all_signals.append({
                    "type":        rule["type"],
                    "rca_type":    rule["rca_type"],
                    "description": rule["description"],
                    "confidence":  rule["confidence"],
                    "source":      "log",
                    "evidence":    sample[:200],
                })

    # ── Check EC2 state ───────────────────────────────────────────────────────
    if ec2_state and ec2_state.lower() in _EC2_STATE_RULES:
        rule = _EC2_STATE_RULES[ec2_state.lower()]
        all_signals.append({
            "type":        rule["type"],
            "rca_type":    rule["rca_type"],
            "description": rule["description"],
            "confidence":  rule["confidence"],
            "source":      "ec2_state",
            "evidence":    f"EC2 state={ec2_state}",
        })

    # ── Check ALB target reasons ──────────────────────────────────────────────
    for reason in (alb_target_reasons or []):
        if reason in _ALB_REASON_RULES:
            rule = _ALB_REASON_RULES[reason]
            all_signals.append({
                "type":        rule["type"],
                "rca_type":    rule["rca_type"],
                "description": rule["description"],
                "confidence":  rule["confidence"],
                "source":      "alb_target_reason",
                "evidence":    f"ALB target reason={reason}",
            })

    # ── Deduplicate by rca_type (keep highest confidence) ────────────────────
    seen: dict[str, dict] = {}
    for sig in all_signals:
        key = sig["rca_type"]
        if key not in seen or sig["confidence"] > seen[key]["confidence"]:
            seen[key] = sig
    deduped = list(seen.values())

    primaries   = [s for s in deduped if s["type"] == "primary_root_cause"]
    supporting  = [s for s in deduped if s["type"] == "supporting_signal"]
    symptoms    = [s for s in deduped if s["type"] == "infra_symptom"]
    downstream  = [s for s in deduped if s["type"] == "downstream_impact"]

    # Best primary = highest confidence
    best_primary = max(primaries, key=lambda x: x["confidence"]) if primaries else None

    return {
        "primary_root_cause":    best_primary,
        "supporting_signals":    supporting,
        "infra_symptoms":        symptoms,
        "downstream_impacts":    downstream,
        "all_signals":           deduped,
        "has_strong_signal":     best_primary is not None and best_primary["confidence"] >= 0.85,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# P4 — Strict workflow state machine
# ═══════════════════════════════════════════════════════════════════════════════

# Only Python may write these states. LLM never touches them.
WORKFLOW_STATES: dict[str, int] = {
    "queued":                 0,
    "triage_started":        10,
    "infra_analysis":        30,
    "logs_analysis":         60,
    "ai_reasoning":          80,
    "remediation_generated": 90,
    "completed":            100,
    "failed":                 0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# P1 — Triage gate (high-confidence early exit)
# ═══════════════════════════════════════════════════════════════════════════════

_HIGH_CONFIDENCE_TRIAGE_MAP: dict[str, dict] = {
    "Target.InvalidState": {
        "likely_issue": "instance_stopped_or_terminated",
        "skip_metrics": True,
        "skip_logs":    True,
        "confidence":   0.95,
        "summary":      "Target.InvalidState: EC2 instance is stopped or terminated. This is a strong hypothesis, but requires validation via EC2 tool.",
    },
    "Target.NotInUse": {
        "likely_issue": "instance_deregistered_or_draining",
        "skip_metrics": True,
        "skip_logs":    True,
        "confidence":   0.80,
        "summary":      "Target.NotInUse: instance deregistered or draining — likely deployment or scale-in event.",
    },
    "Target.FailedHealthChecks": {
        "likely_issue": "app_crash_or_port_unreachable",
        "skip_metrics": False,
        "skip_logs":    False,
        "confidence":   0.65,
        "summary":      "Target.FailedHealthChecks: app not responding on health port. Needs deeper investigation.",
    },
    "Target.DeregistrationInProgress": {
        "likely_issue": "deployment_or_scale_in",
        "skip_metrics": True,
        "skip_logs":    False,
        "confidence":   0.75,
        "summary":      "Target.DeregistrationInProgress: rolling deploy or scale-in in progress.",
    },
}

# Note: ALB target health reasons are used as investigation hints only.
# Final RCA requires validation from logs, metrics, or events.

def pre_triage_targets(targets: list[dict]) -> dict:
    """
    Provide an initial investigation hypothesis based on ALB target status.

    Returns:
      {
        "triage_result": None | {
            likely_issue, skip_metrics, skip_logs,
            confidence, summary, affected_instances,
            target_reason
        }
    }
    """
    if not targets:
        return {"triage_result": None}

    unhealthy = [t for t in targets if t.get("target_health") not in ("healthy", None)]
    if not unhealthy:
        return {"triage_result": None}

    reasons = {t.get("target_reason", "") for t in unhealthy}
    if len(reasons) != 1:
        return {"triage_result": None}

    reason    = list(reasons)[0]
    blueprint = _HIGH_CONFIDENCE_TRIAGE_MAP.get(reason)
    if not blueprint:
        return {"triage_result": None}

    return {
        "triage_result": {
            **blueprint,
            "target_reason":       reason,
            "affected_instances":  [t.get("instance_id") for t in unhealthy],
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# P6 — Investigation stop conditions
# ═══════════════════════════════════════════════════════════════════════════════

def has_sufficient_evidence(signals: dict, tool_calls_made: list[str]) -> tuple[bool, str]:
    """
    Called after each tool-node execution to decide if we have enough evidence
    to stop the investigation loop without further LLM invokes.

    Returns (should_stop: bool, reason: str).

    Conditions that trigger early stop:
    1. A primary_root_cause with confidence >= 0.88 has been extracted AND
       at least ec2_analysis + one of (logs or cloudtrail) has been called.
    2. EC2 state is stopped/terminated (always sufficient on its own).
    """
    primary = signals.get("primary_root_cause")
    if not primary:
        return False, ""

    rca_type   = primary.get("rca_type", "")
    confidence = primary.get("confidence", 0)

    # Instance stopped/terminated requires EC2 tool validation
    if rca_type in ("instance_stopped", "instance_terminated"):
        if "get_ec2_analysis" in tool_calls_made:
            return True, f"EC2 instance is {rca_type.replace('instance_', '')} — corroborated by EC2 analysis."

    # High-confidence config/DB error confirmed by logs
    high_conf_types = {
        "database_config_error", "database_unreachable",
        "iam_permission_error",  "aws_credentials_error",
        "disk_full",             "oom_kill",
    }
    if rca_type in high_conf_types and confidence >= 0.88:
        have_ec2   = any("get_ec2_analysis"    in t for t in tool_calls_made)
        have_extra = any(t in tool_calls_made  for t in ("get_compressed_logs", "get_infra_events"))
        if have_ec2 and have_extra:
            return True, (
                f"High-confidence primary RCA '{rca_type}' ({confidence:.0%}) confirmed "
                f"by EC2 + {'logs' if 'get_compressed_logs' in tool_calls_made else 'CloudTrail'}."
            )

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# P7 — Temporal correlator
# ═══════════════════════════════════════════════════════════════════════════════

def correlate_timeline(
    infra_events: list[dict],
    log_anchor_ts: str | None,
    incident_down_time: str,
    window_minutes: int = 15,
) -> dict:
    """
    Links CloudTrail infrastructure events → first log error → ALB degradation
    in a causal timeline (P7).

    infra_events: list of CloudTrail high_risk_events dicts with 'event_time' key.
    log_anchor_ts: ISO timestamp of first error in logs (from log anchor).
    incident_down_time: ISO timestamp of monitor alert.
    window_minutes: how many minutes before incident_down_time to look for triggers.

    Returns:
    {
      "causal_chain": [
        { "timestamp": ..., "type": "infra_change|log_error|alb_degradation",
          "event": ..., "minutes_before_incident": float }
      ],
      "probable_trigger": { event } | None,
      "trigger_to_error_gap_minutes": float | None,
    }
    """
    try:
        down_dt = datetime.fromisoformat(incident_down_time.replace("Z", "+00:00"))
    except Exception:
        return {"error": "Invalid incident_down_time format"}

    cutoff = down_dt - timedelta(minutes=window_minutes)
    chain  = []

    # ── Infra changes ─────────────────────────────────────────────────────────
    for evt in (infra_events or []):
        raw_ts = evt.get("event_time") or evt.get("EventTime") or evt.get("time")
        if not raw_ts:
            continue
        try:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except Exception:
            continue
        if ts >= cutoff:
            chain.append({
                "timestamp":                str(ts),
                "type":                     "infra_change",
                "event":                    evt.get("event_name") or evt.get("EventName", "unknown"),
                "detail":                   evt,
                "minutes_before_incident":  round((down_dt - ts).total_seconds() / 60, 1),
            })

    # ── First log error ───────────────────────────────────────────────────────
    if log_anchor_ts:
        try:
            anchor_dt = datetime.fromisoformat(log_anchor_ts.replace("Z", "+00:00"))
            chain.append({
                "timestamp":               str(anchor_dt),
                "type":                    "first_log_error",
                "event":                   "First error detected in application logs",
                "minutes_before_incident": round((down_dt - anchor_dt).total_seconds() / 60, 1),
            })
        except Exception:
            pass

    # ── ALB degradation = incident down_time ─────────────────────────────────
    chain.append({
        "timestamp":               str(down_dt),
        "type":                    "alb_degradation",
        "event":                   "Monitor alert fired / ALB reported unhealthy",
        "minutes_before_incident": 0,
    })

    chain.sort(key=lambda x: x["timestamp"])

    # ── Identify probable trigger (infra event closest before first log error) ─
    probable_trigger          = None
    trigger_to_error_gap_mins = None

    infra_changes = [c for c in chain if c["type"] == "infra_change"]
    log_errors    = [c for c in chain if c["type"] == "first_log_error"]

    if infra_changes and log_errors:
        first_log_ts = log_errors[0]["timestamp"]
        # Latest infra change before or at first log error
        before_error = [
            c for c in infra_changes if c["timestamp"] <= first_log_ts
        ]
        if before_error:
            probable_trigger = max(before_error, key=lambda x: x["timestamp"])
            try:
                trigger_dt           = datetime.fromisoformat(probable_trigger["timestamp"].replace("Z", "+00:00"))
                first_log_dt         = datetime.fromisoformat(first_log_ts.replace("Z", "+00:00"))
                trigger_to_error_gap_mins = round(
                    (first_log_dt - trigger_dt).total_seconds() / 60, 1
                )
            except Exception:
                pass

    return {
        "causal_chain":               chain,
        "probable_trigger":           probable_trigger,
        "trigger_to_error_gap_minutes": trigger_to_error_gap_mins,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# P8 — Similar incident lookup
# ═══════════════════════════════════════════════════════════════════════════════

def find_similar_incidents(
    rca_type: str,
    monitor_type: str | None = None,
    limit: int = 3,
) -> list[dict]:
    """
    Query past incidents with the same rca_type / monitor_type for context (P8).
    Returns up to `limit` similar incidents with their root_cause and remediation.

    This is a simple keyword match — not vector similarity.
    Future: replace with pgvector cosine similarity on embedded RCA text.
    """
    try:
        from app.utils.db import get_db
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        monitor_name,
                        monitor_type,
                        down_message,
                        analysis_result,
                        remediation_steps,
                        confidence_score,
                        analysis_completed_at
                    FROM meyiconnect.insight_incidents
                    WHERE analysis_status = 'completed'
                      AND analysis_result IS NOT NULL
                      AND (
                          analysis_result::text ILIKE %s
                          OR (monitor_type = %s AND %s IS NOT NULL)
                      )
                      AND analysis_completed_at IS NOT NULL
                    ORDER BY analysis_completed_at DESC
                    LIMIT %s
                    """,
                    (
                        f"%{rca_type}%",
                        monitor_type,
                        monitor_type,
                        limit,
                    ),
                )
                rows = cur.fetchall() or []

        results = []
        for row in rows:
            try:
                rca = json.loads(row.get("analysis_result") or "{}")
                results.append({
                    "incident_id":    row.get("id"),
                    "monitor_name":   row.get("monitor_name"),
                    "monitor_type":   row.get("monitor_type"),
                    "down_message":   row.get("down_message"),
                    "summary":        rca.get("summary", ""),
                    "confidence":     row.get("confidence_score"),
                    "resolved_at":    str(row.get("analysis_completed_at", "")),
                })
            except Exception:
                pass

        return results

    except Exception as exc:
        logger.warning(f"[SimilarIncidents] Lookup failed: {exc}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Tool list exported for the agent
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    resolve_incident_targets,
    get_ec2_analysis,
    get_compressed_logs,
    correlate_instances,
    get_infra_events,
    get_alb_target_health,
    query_logs_insights,
]
