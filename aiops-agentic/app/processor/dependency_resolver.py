"""
processor/dependency_resolver.py
─────────────────────────────────
Resolves ANY dependency type (ec2 / alb / domain) into a normalized list of EC2 targets.

The rest of the pipeline (metrics, logs, RCA) never cares about the source —
it always receives:

    [
        {
            "instance_id": "i-xxx",
            "region":      "us-east-1",
            "log_groups":  ["group-a", "group-b"],
        },
        ...
    ]

Supported dependency types
──────────────────────────
  • ec2    — direct instance ID, passed through unchanged
  • alb    — DNS name resolved → ARN → target groups → EC2 instance IDs
  • domain — custom domain (e.g. api.customer.com) resolved via DNS CNAME
             to ALB DNS, then follows the same alb path

Auto-detect (resource_id only, no explicit type)
─────────────────────────────────────────────────
  i-...                               → ec2
  *.elb.amazonaws.com                 → alb
  anything else                       → domain

Input schema (one dependency dict)
───────────────────────────────────
{
    "type":            "ec2" | "alb" | "domain",   # optional — auto-detected if omitted
    "resource_id":     "i-01cbe..." | "internal-api-prod.us-east-1.elb.amazonaws.com"
                       | "api.customer.com",
    "region":          "us-east-1",
    "log_group_name":  ["group-a", "group-b"]   # optional list or CSV string
}
"""

import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Type auto-detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_resource_type(resource_id: str) -> str:
    """
    Infer dependency type from the resource_id value.

    Rules (checked in order):
      1. Starts with "i-"           → ec2
      2. Contains ".elb.amazonaws.com" → alb
      3. Anything else              → domain
    """
    val = (resource_id or "").strip()
    if val.startswith("i-"):
        return "ec2"
    if ".elb.amazonaws.com" in val.lower():
        return "alb"
    return "domain"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_log_groups(raw) -> list[str]:
    """Normalise log_group_name / log_group_names → plain list of strings."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [g.strip() for g in raw.split(",") if g.strip()]
    if isinstance(raw, list):
        return [g for g in raw if g]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# DNS helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_domain_to_alb_dns(domain: str) -> Optional[str]:
    """
    Resolve a custom domain (e.g. api.customer.com) to an AWS ALB DNS name
    by following CNAME chain until we reach a *.elb.amazonaws.com hostname.

    Strategy:
      1. Try dnspython (preferred — follows full CNAME chain cleanly).
      2. Fall back to socket.getaddrinfo canonical name if dnspython unavailable.

    Returns the ALB DNS name (str) or None if resolution fails or no ALB found.
    """
    clean_domain = (
        domain
        .replace("http://", "")
        .replace("https://", "")
        .strip("/")
        .lower()
    )

    # ── Attempt 1: dnspython (follows CNAME chain) ────────────────────────────
    try:
        import dns.resolver  # type: ignore

        resolver = dns.resolver.Resolver()
        resolver.timeout = 5
        resolver.lifetime = 10

        current = clean_domain
        max_hops = 10  # guard against CNAME loops

        for _ in range(max_hops):
            try:
                answers = resolver.resolve(current, "CNAME")
                cname_target = str(answers[0].target).rstrip(".")
                logger.debug(f"[DomainResolver] CNAME: {current} → {cname_target}")
                current = cname_target

                if ".elb.amazonaws.com" in current:
                    logger.info(
                        f"[DomainResolver] Resolved '{domain}' → ALB DNS '{current}'"
                    )
                    return current
            except dns.resolver.NoAnswer:
                # No more CNAMEs — check if the final name itself is an ALB
                break
            except dns.resolver.NXDOMAIN:
                logger.warning(
                    f"[DomainResolver] NXDOMAIN while resolving CNAME chain for {current}"
                )
                return None
            except Exception as exc:
                logger.warning(f"[DomainResolver] CNAME lookup error for {current}: {exc}")
                break

        # Final check: the last hop may already be an ALB DNS
        if ".elb.amazonaws.com" in current:
            return current

        logger.warning(
            f"[DomainResolver] '{domain}' CNAME chain did not terminate at an ALB: '{current}'"
        )
        return None

    except ImportError:
        logger.warning(
            "[DomainResolver] dnspython not installed — falling back to socket"
        )

    # ── Attempt 2: socket fallback (less reliable for CNAME chains) ───────────
    try:
        hostname, aliases, _ = socket.gethostbyname_ex(clean_domain)
        all_names = [hostname] + list(aliases)
        for name in all_names:
            if ".elb.amazonaws.com" in name.lower():
                logger.info(
                    f"[DomainResolver] (socket) Resolved '{domain}' → ALB DNS '{name}'"
                )
                return name.lower()

        logger.warning(
            f"[DomainResolver] (socket) No ALB DNS found for '{domain}'. "
            f"Resolved names: {all_names}"
        )
        return None

    except socket.gaierror as exc:
        logger.warning(f"[DomainResolver] socket.gethostbyname_ex failed for '{domain}': {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ALB resolver
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_alb_dns_to_arn(elbv2_client, dns_name: str) -> Optional[str]:
    """
    Resolve an ALB DNS name to its LoadBalancerArn.

    AWS doesn't have a 'lookup by DNS' API, so we page through all ALBs
    in the account/region and match on DNSName (case-insensitive).

    Also handles the "dualstack." prefix that AWS adds to some DNS names.
    """
    normalized_dns = (
        dns_name
        .replace("http://", "")
        .replace("https://", "")
        .strip("/")
        .lower()
        .removeprefix("dualstack.")   # AWS sometimes returns dualstack.<alb_dns>
    )

    paginator = elbv2_client.get_paginator("describe_load_balancers")
    for page in paginator.paginate():
        for lb in page.get("LoadBalancers", []):
            alb_dns = lb.get("DNSName", "").lower().removeprefix("dualstack.")
            if alb_dns == normalized_dns:
                arn = lb["LoadBalancerArn"]
                logger.info(
                    f"[ALB resolver] Matched DNS "
                    f"'{normalized_dns}' → ARN '{arn}'"
                )
                return arn

    logger.warning(f"[ALB resolver] No ALB found for DNS name: {dns_name}")
    return None


def _get_target_instances(elbv2_client, alb_arn: str) -> list[dict]:
    """
    ALB ARN → list of EC2 instance IDs currently registered as targets.

    Returns:
        [
            {"instance_id": "i-xxx", "health": "healthy",
             "reason": "...", "target_group_arn": "arn:..."},
            ...
        ]
    """
    instances: list[dict] = []

    try:
        tg_resp = elbv2_client.describe_target_groups(LoadBalancerArn=alb_arn)
    except Exception as exc:
        logger.error(f"[ALB resolver] describe_target_groups failed: {exc}")
        return instances

    for tg in tg_resp.get("TargetGroups", []):
        tg_arn = tg["TargetGroupArn"]
        try:
            health_resp = elbv2_client.describe_target_health(TargetGroupArn=tg_arn)
        except Exception as exc:
            logger.warning(f"[ALB resolver] describe_target_health failed for {tg_arn}: {exc}")
            continue

        for thd in health_resp.get("TargetHealthDescriptions", []):
            target = thd.get("Target", {})
            instance_id = target.get("Id")
            if not instance_id or not instance_id.startswith("i-"):
                continue  # skip IP-mode or Lambda targets

            health_info = thd.get("TargetHealth", {})
            instances.append({
                "instance_id":     instance_id,
                "health":          health_info.get("State", "unknown"),
                "reason":          health_info.get("Reason", ""),
                "description":     health_info.get("Description", ""),
                "target_group_arn": tg_arn,
            })
            logger.info(
                f"[ALB resolver] Found target {instance_id} "
                f"state={health_info.get('State')} reason={health_info.get('Reason')}"
            )

    logger.info(f"[ALB resolver] Total targets for ALB: {len(instances)}")
    return instances


def resolve_alb(dep: dict, aws_factory) -> tuple[list[dict], dict]:
    """
    Resolve an ALB dependency into normalised EC2 targets.

    Returns:
        (normalized_instances, alb_meta)

        normalized_instances — list of {instance_id, region, log_groups}
        alb_meta             — raw ALB info for the prompt / DB (target health, etc.)
    """
    region      = dep.get("region", "us-east-1")
    resource_id = dep.get("resource_id", "")      # DNS name
    log_groups  = _parse_log_groups(
        dep.get("log_group_name") or dep.get("log_group_names")
    )

    elbv2 = aws_factory.get_client("elbv2", region_name=region)

    # Step 1 — DNS → ARN
    alb_arn = _resolve_alb_dns_to_arn(elbv2, resource_id)
    if not alb_arn:
        logger.error(f"[ALB resolver] Cannot resolve ALB DNS: {resource_id}")
        return [], {}

    # Step 2 — ARN → target instances
    raw_targets = _get_target_instances(elbv2, alb_arn)

    alb_meta = {
        "alb_dns":  resource_id,
        "alb_arn":  alb_arn,
        "targets":  raw_targets,
        "total":    len(raw_targets),
        "healthy":  sum(1 for t in raw_targets if t["health"] == "healthy"),
        "unhealthy": sum(1 for t in raw_targets if t["health"] == "unhealthy"),
    }

    normalized = [
        {
            "instance_id": t["instance_id"],
            "region":      region,
            "log_groups":  log_groups,
            # carry target health for prioritisation later
            "target_health": t["health"],
            "target_reason": t.get("reason", ""),
        }
        for t in raw_targets
    ]

    return normalized, alb_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Domain resolver  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_domain(dep: dict, aws_factory) -> tuple[list[dict], dict]:
    """
    Resolve a custom domain dependency (e.g. api.customer.com) into EC2 targets.

    Flow:
      1. DNS lookup → CNAME chain → ALB DNS (*.elb.amazonaws.com)
      2. Verify that ALB DNS exists in the customer's AWS account
         (security: prevent investigating ALBs that don't belong to this account)
      3. Reuse existing ALB → targets logic

    Returns:
        (normalized_instances, alb_meta)
        alb_meta includes "original_domain" so the agent can reference it.
    """
    region      = dep.get("region", "us-east-1")
    domain      = dep.get("resource_id", "")
    log_groups  = _parse_log_groups(
        dep.get("log_group_name") or dep.get("log_group_names")
    )

    logger.info(f"[DomainResolver] Resolving domain '{domain}' in region '{region}'")

    # ── Step 1: DNS → ALB DNS ─────────────────────────────────────────────────
    alb_dns = _resolve_domain_to_alb_dns(domain)
    if not alb_dns:
        logger.error(
            f"[DomainResolver] Could not resolve '{domain}' to an ALB DNS. "
            f"Check that the domain has a CNAME pointing to an AWS ALB."
        )
        return [], {
            "error": f"Domain '{domain}' did not resolve to any AWS ALB DNS name",
            "original_domain": domain,
        }

    # ── Step 2: Verify ALB belongs to this AWS account ────────────────────────
    elbv2   = aws_factory.get_client("elbv2", region_name=region)
    alb_arn = _resolve_alb_dns_to_arn(elbv2, alb_dns)

    if not alb_arn:
        logger.error(
            f"[DomainResolver] SECURITY: Domain '{domain}' resolves to ALB DNS "
            f"'{alb_dns}' which does NOT exist in the configured AWS account/region. "
            f"Investigation blocked."
        )
        return [], {
            "error": (
                f"Domain '{domain}' resolves to ALB '{alb_dns}' "
                f"which does not belong to the configured AWS account. "
                f"Investigation blocked for security."
            ),
            "original_domain": domain,
            "resolved_alb_dns": alb_dns,
        }

    logger.info(
        f"[DomainResolver] Verified: '{domain}' → '{alb_dns}' → ARN '{alb_arn}' "
        f"(belongs to account)"
    )

    # ── Step 3: ALB ARN → targets (reuse existing logic) ──────────────────────
    raw_targets = _get_target_instances(elbv2, alb_arn)

    alb_meta = {
        "original_domain": domain,       # what the user provided
        "alb_dns":         alb_dns,      # what DNS resolved to
        "alb_arn":         alb_arn,
        "targets":         raw_targets,
        "total":           len(raw_targets),
        "healthy":         sum(1 for t in raw_targets if t["health"] == "healthy"),
        "unhealthy":       sum(1 for t in raw_targets if t["health"] == "unhealthy"),
    }

    normalized = [
        {
            "instance_id":   t["instance_id"],
            "region":        region,
            "log_groups":    log_groups,
            "target_health": t["health"],
            "target_reason": t.get("reason", ""),
        }
        for t in raw_targets
    ]

    return normalized, alb_meta


# ═══════════════════════════════════════════════════════════════════════════════
# EC2 resolver (passthrough)
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_ec2(dep: dict) -> list[dict]:
    """Direct EC2 dependency — just normalise field names."""
    log_groups = _parse_log_groups(
        dep.get("log_group_name") or dep.get("log_group_names")
    )
    return [{
        "instance_id":   dep.get("resource_id") or dep.get("instance_id", ""),
        "region":        dep.get("region", "us-east-1"),
        "log_groups":    log_groups,
        "target_health": "unknown",
        "target_reason": "",
    }]


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_dependencies(raw_deps: list[dict], aws_factory) -> tuple[list[dict], dict]:
    """
    Convert raw dependency list (from DB) into normalized EC2 targets.

    Supports three dep types: ec2, alb, domain.
    If "type" is missing, it is auto-detected from "resource_id".

    Returns:
        (normalized_instances, alb_meta)

        normalized_instances — list[{instance_id, region, log_groups, ...}]
        alb_meta             — ALB/domain-specific data if dep type is alb or domain, else {}
    """
    all_instances: list[dict] = []
    alb_meta: dict = {}

    for dep in raw_deps:
        resource_id = (dep.get("resource_id") or dep.get("instance_id") or "").strip()

        # Auto-detect type if not explicitly provided
        dep_type = (dep.get("type") or _detect_resource_type(resource_id)).lower()

        if dep_type == "alb":
            instances, meta = resolve_alb(dep, aws_factory)
            all_instances.extend(instances)
            if meta:
                alb_meta = meta   # one ALB per incident for now

        elif dep_type == "domain":
            instances, meta = resolve_domain(dep, aws_factory)
            all_instances.extend(instances)
            if meta:
                alb_meta = meta   # domain resolver returns same alb_meta shape

        else:
            # default: treat as EC2
            instances = resolve_ec2(dep)
            all_instances.extend(instances)

    # De-duplicate by instance_id (ALB may register same instance in multiple TGs)
    seen: set[str] = set()
    unique: list[dict] = []
    for inst in all_instances:
        iid = inst["instance_id"]
        if iid and iid not in seen:
            seen.add(iid)
            unique.append(inst)

    logger.info(
        f"[DependencyResolver] Resolved {len(raw_deps)} dep(s) → "
        f"{len(unique)} unique EC2 target(s): {[i['instance_id'] for i in unique]}"
    )
    return unique, alb_meta