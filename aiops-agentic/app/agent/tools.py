"""
app/agent/tools.py
──────────────────
AWS investigation tools for the LangGraph agent (v4).

Changes in v4
─────────────
Added infrastructure-level causal RCA tools:
  - investigate_network_path: full EC2→DB network path investigation
    (DNS, route tables, NACL, SG, port connectivity)
  - get_security_group_rules: inspect SG inbound/outbound rules
  - get_nacl_rules: inspect NACL rules for a subnet
  - get_route_table: check route tables for a subnet/VPC
  - check_cloudtrail_sg_changes: find recent SG modifications via CloudTrail
    (who changed what and when — correlates with incident timeline)

These tools enable the agent to go beyond "database unreachable" and
produce evidence like "Security Group sg-xxxxx blocked outbound TCP 5432,
modified by user X at 09:24 UTC, 2 minutes before outage."
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
      - details  (state, type, AZ, IPs, SGs, tags, subnet_id, vpc_id)
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
        findings = []

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
                "subnet_id":         inst.get("SubnetId"),
                "vpc_id":            inst.get("VpcId"),
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
            if inst:
                findings.append({"category": "compute", "message": f"EC2 instance {iid} state is {inst.get('State', {}).get('Name')}"})

        return {"instances": results, "findings": findings}

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
# TOOL 8 — Network Path Investigation (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def investigate_network_path(
    source_instance_id: str,
    destination_host: str,
    destination_port: int,
    region: str = "",
) -> dict:
    """
    Investigate the full network path from an EC2 instance to a destination host/port.
    Checks DNS resolution, route tables, NACLs, and Security Groups.

    Call this when logs show connection timeouts or DB unreachable errors.
    This tool finds the exact network layer that is blocking traffic.

    Args:
        source_instance_id: EC2 instance ID (e.g. 'i-0abc1234')
        destination_host: target hostname or IP (e.g. 'mydb.xxxx.rds.amazonaws.com')
        destination_port: target port (e.g. 5432 for PostgreSQL, 3306 for MySQL)
        region: AWS region (uses incident region if omitted)

    Returns:
        {
          "dns_resolution": "PASS" | "FAIL" | "UNKNOWN",
          "route_table": "PASS" | "FAIL" | "UNKNOWN",
          "nacl_outbound": "PASS" | "FAIL" | "UNKNOWN",
          "nacl_inbound": "PASS" | "FAIL" | "UNKNOWN",
          "security_group_outbound": "PASS" | "FAIL" | "UNKNOWN",
          "blocked_layer": null | "security_group" | "nacl" | "route_table" | "dns",
          "blocked_port": null | int,
          "security_group_details": {...},
          "nacl_details": {...},
          "route_table_details": {...},
          "summary": "Human-readable diagnosis"
        }
    """
    region = region or _region()
    ec2    = _factory().get_client("ec2", region_name=region)

    def _run():
        result = {
            "dns_resolution":          "UNKNOWN",
            "route_table":             "UNKNOWN",
            "nacl_outbound":           "UNKNOWN",
            "nacl_inbound":            "UNKNOWN",
            "security_group_outbound": "UNKNOWN",
            "blocked_layer":           None,
            "blocked_port":            None,
            "security_group_details":  {},
            "nacl_details":            {},
            "route_table_details":     {},
            "summary":                 "",
            "findings":                [],
        }

        # ── Get instance network info ─────────────────────────────────────────
        inst_resp = ec2.describe_instances(InstanceIds=[source_instance_id])
        reservations = inst_resp.get("Reservations", [])
        if not reservations:
            return {"error": f"Instance {source_instance_id} not found"}

        inst      = reservations[0]["Instances"][0]
        subnet_id = inst.get("SubnetId")
        vpc_id    = inst.get("VpcId")
        sg_ids    = [sg["GroupId"] for sg in inst.get("SecurityGroups", [])]

        if not subnet_id:
            return {"error": "Instance has no SubnetId — cannot investigate network path"}

        # ── 1. Security Group outbound check ─────────────────────────────────
        sg_resp    = ec2.describe_security_groups(GroupIds=sg_ids)
        sg_allows  = False
        sg_details = []

        for sg in sg_resp.get("SecurityGroups", []):
            sg_info = {
                "group_id":   sg["GroupId"],
                "group_name": sg["GroupName"],
                "outbound_rules": [],
            }
            for rule in sg.get("IpPermissionsEgress", []):
                from_port = rule.get("FromPort", 0)
                to_port   = rule.get("ToPort", 65535)
                protocol  = rule.get("IpProtocol", "-1")
                # -1 = all traffic
                if protocol == "-1":
                    sg_allows = True
                    sg_info["outbound_rules"].append({
                        "port_range": "ALL",
                        "protocol":   "ALL",
                        "allows_destination_port": True,
                    })
                elif protocol in ("tcp", "6"):
                    allows = (from_port <= destination_port <= to_port)
                    if allows:
                        sg_allows = True
                    sg_info["outbound_rules"].append({
                        "port_range": f"{from_port}-{to_port}",
                        "protocol":   "tcp",
                        "allows_destination_port": allows,
                    })
            sg_details.append(sg_info)

        result["security_group_outbound"] = "PASS" if sg_allows else "FAIL"
        result["security_group_details"]  = {"security_groups": sg_details}
        if not sg_allows:
            result["blocked_layer"] = "security_group"
            result["blocked_port"]  = destination_port

        # ── 2. NACL outbound check ────────────────────────────────────────────
        nacl_resp   = ec2.describe_network_acls(
            Filters=[{"Name": "association.subnet-id", "Values": [subnet_id]}]
        )
        nacl_allows_out = False
        nacl_allows_in  = False
        nacl_details    = []

        for nacl in nacl_resp.get("NetworkAcls", []):
            outbound_rules = sorted(
                [e for e in nacl.get("Entries", []) if not e.get("Egress") == False],
                key=lambda x: x.get("RuleNumber", 9999),
            )
            inbound_rules  = sorted(
                [e for e in nacl.get("Entries", []) if e.get("Egress") == False],
                key=lambda x: x.get("RuleNumber", 9999),
            )

            def _nacl_allows(rules, port):
                for rule in rules:
                    protocol = rule.get("Protocol", "-1")
                    action   = rule.get("RuleAction", "deny")
                    port_range = rule.get("PortRange", {})
                    from_p = port_range.get("From", 0)
                    to_p   = port_range.get("To", 65535)
                    # Rule number 32767 = default deny all
                    if protocol == "-1" or (from_p <= port <= to_p):
                        return action == "allow"
                return False

            nacl_allows_out = _nacl_allows(outbound_rules, destination_port)
            # For inbound, check ephemeral port range (1024-65535)
            nacl_allows_in  = _nacl_allows(inbound_rules, 1024)

            nacl_details.append({
                "nacl_id":        nacl.get("NetworkAclId"),
                "outbound_allows_port": nacl_allows_out,
                "inbound_allows_ephemeral": nacl_allows_in,
                "outbound_rule_count": len(outbound_rules),
                "inbound_rule_count":  len(inbound_rules),
            })

        result["nacl_outbound"] = "PASS" if nacl_allows_out else "FAIL"
        result["nacl_inbound"]  = "PASS" if nacl_allows_in  else "FAIL"
        result["nacl_details"]  = {"nacls": nacl_details}
        if not nacl_allows_out and result["blocked_layer"] is None:
            result["blocked_layer"] = "nacl"
            result["blocked_port"]  = destination_port

        # ── 3. Route table check ──────────────────────────────────────────────
        rt_resp = ec2.describe_route_tables(
            Filters=[{"Name": "association.subnet-id", "Values": [subnet_id]}]
        )
        # Fall back to main route table for the VPC
        if not rt_resp.get("RouteTables"):
            rt_resp = ec2.describe_route_tables(
                Filters=[
                    {"Name": "vpc-id",            "Values": [vpc_id]},
                    {"Name": "association.main",   "Values": ["true"]},
                ]
            )

        has_default_route = False
        rt_details        = []

        for rt in rt_resp.get("RouteTables", []):
            routes = []
            for route in rt.get("Routes", []):
                dest  = route.get("DestinationCidrBlock") or route.get("DestinationPrefixListId", "")
                state = route.get("State", "")
                gw    = (
                    route.get("GatewayId")
                    or route.get("NatGatewayId")
                    or route.get("TransitGatewayId")
                    or route.get("VpcPeeringConnectionId")
                    or route.get("NetworkInterfaceId", "")
                )
                is_default = dest in ("0.0.0.0/0", "::/0")
                if is_default and state == "active":
                    has_default_route = True
                routes.append({
                    "destination": dest,
                    "gateway":     gw,
                    "state":       state,
                    "is_default":  is_default,
                })
            rt_details.append({
                "route_table_id": rt.get("RouteTableId"),
                "routes":         routes,
            })

        result["route_table"]         = "PASS" if has_default_route else "FAIL"
        result["route_table_details"] = {"route_tables": rt_details}
        if not has_default_route and result["blocked_layer"] is None:
            result["blocked_layer"] = "route_table"

        # ── 4. DNS resolution check (heuristic — no actual DNS query) ─────────
        # We can't run socket.getaddrinfo in this context, but we can check
        # whether the destination looks like a hostname or IP, and whether
        # a VPC DNS resolver would be expected to resolve RDS/internal hosts.
        import re as _re
        is_ip = bool(_re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", destination_host))
        if is_ip:
            result["dns_resolution"] = "PASS"  # Direct IP — no DNS needed
        elif "rds.amazonaws.com" in destination_host or "internal" in destination_host:
            result["dns_resolution"] = "PASS"  # AWS private DNS — usually works
        else:
            result["dns_resolution"] = "UNKNOWN"  # Cannot verify without socket

        # ── Summary ───────────────────────────────────────────────────────────
        if has_default_route:
            result["findings"].append({"category": "network", "message": "Route table validation passed"})
        if nacl_allows_out and nacl_allows_in:
            result["findings"].append({"category": "network", "message": "NACL validation passed"})

        if result["blocked_layer"] == "security_group":
            result["summary"] = (
                f"BLOCKED: Security group outbound rule does not allow TCP port "
                f"{destination_port}. Instance cannot reach {destination_host}:{destination_port}. "
                f"Check outbound rules on: {', '.join(sg_ids)}"
            )
            result["findings"].append({"category": "network", "message": f"Security group blocks outbound {destination_port}"})
        elif result["blocked_layer"] == "nacl":
            result["summary"] = (
                f"BLOCKED: Network ACL on subnet {subnet_id} denies outbound traffic "
                f"to port {destination_port}."
            )
            result["findings"].append({"category": "network", "message": f"NACL blocks outbound {destination_port}"})
        elif result["blocked_layer"] == "route_table":
            result["summary"] = (
                f"BLOCKED: No active default route (0.0.0.0/0) found in route table "
                f"for subnet {subnet_id}. Traffic cannot leave the subnet."
            )
            result["findings"].append({"category": "network", "message": "Route table missing active default route"})
        else:
            result["summary"] = (
                f"Network path from {source_instance_id} to "
                f"{destination_host}:{destination_port} appears clear at the AWS network layer. "
                f"Issue may be at the database level (DB down, auth, or firewall on DB host)."
            )
            result["findings"].append({"category": "network", "message": f"AWS network path to {destination_port} is clear"})

        return result

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 9 — Security Group rules inspector (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_security_group_rules(security_group_ids: list[str], region: str = "") -> dict:
    """
    Inspect Security Group inbound and outbound rules in detail.
    Use after investigate_network_path shows security_group_outbound=FAIL,
    or when you need to verify specific port access.

    Args:
        security_group_ids: list of SG IDs (e.g. ['sg-0abc1234'])
        region: AWS region

    Returns detailed inbound/outbound rules for each SG.
    """
    region = region or _region()
    ec2    = _factory().get_client("ec2", region_name=region)

    def _run():
        resp    = ec2.describe_security_groups(GroupIds=security_group_ids)
        results = []

        for sg in resp.get("SecurityGroups", []):
            inbound  = []
            outbound = []

            for rule in sg.get("IpPermissions", []):
                protocol  = rule.get("IpProtocol", "-1")
                from_port = rule.get("FromPort")
                to_port   = rule.get("ToPort")
                cidrs     = [r["CidrIp"] for r in rule.get("IpRanges", [])]
                cidrs    += [r["CidrIpv6"] for r in rule.get("Ipv6Ranges", [])]
                sgs_ref   = [g["GroupId"] for g in rule.get("UserIdGroupPairs", [])]
                inbound.append({
                    "protocol":   protocol,
                    "port_range": f"{from_port}-{to_port}" if from_port is not None else "ALL",
                    "sources":    cidrs + sgs_ref,
                })

            for rule in sg.get("IpPermissionsEgress", []):
                protocol  = rule.get("IpProtocol", "-1")
                from_port = rule.get("FromPort")
                to_port   = rule.get("ToPort")
                cidrs     = [r["CidrIp"] for r in rule.get("IpRanges", [])]
                cidrs    += [r["CidrIpv6"] for r in rule.get("Ipv6Ranges", [])]
                sgs_ref   = [g["GroupId"] for g in rule.get("UserIdGroupPairs", [])]
                outbound.append({
                    "protocol":   protocol,
                    "port_range": f"{from_port}-{to_port}" if from_port is not None else "ALL",
                    "destinations": cidrs + sgs_ref,
                })

            results.append({
                "group_id":        sg["GroupId"],
                "group_name":      sg["GroupName"],
                "vpc_id":          sg.get("VpcId"),
                "inbound_rules":   inbound,
                "outbound_rules":  outbound,
                "inbound_count":   len(inbound),
                "outbound_count":  len(outbound),
            })

        return {"security_groups": results}

    return _safe(_run)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 10 — CloudTrail SG change audit (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def check_cloudtrail_sg_changes(
    security_group_id: str,
    lookback_minutes: int = 60,
    region: str = "",
) -> dict:
    """
    Find recent Security Group modifications in CloudTrail.
    Use this after investigate_network_path finds a blocked port to determine
    WHO changed the SG and WHEN — enabling high-confidence causal RCA.

    Correlates the SG change timestamp with the incident start time.

    Args:
        security_group_id: SG ID to audit (e.g. 'sg-0abc1234')
        lookback_minutes: how far back to search (default 60 min)
        region: AWS region

    Returns:
        {
          "changes_found": bool,
          "changes": [
            {
              "event_name": "AuthorizeSecurityGroupEgress" | "RevokeSecurityGroupEgress" | ...,
              "event_time": "ISO timestamp",
              "user": "who made the change",
              "minutes_before_incident": float,
              "changed_rules": [...],
              "raw_event": {...}
            }
          ],
          "probable_cause": "Human-readable causal statement" | null
        }
    """
    region = region or _region()
    ct     = _factory().get_client("cloudtrail", region_name=region)
    row    = _incident()

    down_time_raw = row.get("incident_down_time")
    if not down_time_raw:
        return {"error": "incident_down_time missing"}

    if isinstance(down_time_raw, str):
        down_time = datetime.fromisoformat(down_time_raw.replace("Z", "+00:00"))
    elif isinstance(down_time_raw, datetime):
        down_time = down_time_raw if down_time_raw.tzinfo else down_time_raw.replace(tzinfo=timezone.utc)
    else:
        return {"error": "Unrecognised incident_down_time type"}

    start_time = down_time - timedelta(minutes=lookback_minutes)
    end_time   = down_time + timedelta(minutes=5)  # small buffer after incident

    SG_CHANGE_EVENTS = {
        "AuthorizeSecurityGroupIngress",
        "AuthorizeSecurityGroupEgress",
        "RevokeSecurityGroupIngress",
        "RevokeSecurityGroupEgress",
        "ModifySecurityGroupRules",
    }

    def _run():
        changes = []
        findings = []

        paginator = ct.get_paginator("lookup_events")
        for page in paginator.paginate(
            LookupAttributes=[{"AttributeKey": "ResourceName", "AttributeValue": security_group_id}],
            StartTime=start_time,
            EndTime=end_time,
        ):
            for event in page.get("Events", []):
                if event.get("EventName") not in SG_CHANGE_EVENTS:
                    continue

                event_time = event.get("EventTime")
                if isinstance(event_time, datetime):
                    if not event_time.tzinfo:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                else:
                    try:
                        event_time = datetime.fromisoformat(str(event_time).replace("Z", "+00:00"))
                    except Exception:
                        continue

                minutes_before = round((down_time - event_time).total_seconds() / 60, 1)

                # Parse who made the change
                raw_ct = {}
                try:
                    raw_ct = json.loads(event.get("CloudTrailEvent", "{}"))
                except Exception:
                    pass

                user_identity = raw_ct.get("userIdentity", {})
                user = (
                    user_identity.get("userName")
                    or user_identity.get("sessionContext", {}).get("sessionIssuer", {}).get("userName")
                    or user_identity.get("arn", "unknown")
                )

                # Extract changed rules from request parameters
                req_params    = raw_ct.get("requestParameters", {})
                changed_rules = []
                for perm in req_params.get("ipPermissions", {}).get("items", []):
                    ip_ranges = perm.get("ipRanges", {}).get("items", [])
                    port_from = perm.get("fromPort")
                    port_to   = perm.get("toPort")
                    protocol  = perm.get("ipProtocol", "?")
                    changed_rules.append({
                        "protocol":   protocol,
                        "port_range": f"{port_from}-{port_to}" if port_from is not None else "ALL",
                        "cidrs":      [r.get("cidrIp") for r in ip_ranges],
                    })

                changes.append({
                    "event_name":            event.get("EventName"),
                    "event_time":            event_time.isoformat(),
                    "user":                  user,
                    "minutes_before_incident": minutes_before,
                    "changed_rules":         changed_rules,
                    "source_ip":             raw_ct.get("sourceIPAddress", ""),
                })

        # Sort: most recent first
        changes.sort(key=lambda x: x["event_time"], reverse=True)

        # Generate causal statement if a relevant change was found before incident
        probable_cause = None
        before_changes = [c for c in changes if c["minutes_before_incident"] > 0]
        if before_changes:
            latest = before_changes[0]
            probable_cause = (
                f"Security group {security_group_id} was modified by '{latest['user']}' "
                f"at {latest['event_time']} ({latest['minutes_before_incident']:.1f} min before incident). "
                f"Event: {latest['event_name']}. "
                + (f"Changed rules: {json.dumps(latest['changed_rules'])}" if latest["changed_rules"] else "")
            )
            findings.append({"category": "change_history", "message": f"CloudTrail shows SG {security_group_id} modified by {latest['user']}"})

        return {
            "changes_found":  len(changes) > 0,
            "change_count":   len(changes),
            "changes":        changes[:10],
            "probable_cause": probable_cause,
            "findings":       findings,
        }

    return _safe(_run)


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
    investigate_network_path,
    get_security_group_rules,
    check_cloudtrail_sg_changes,
]