"""
app/sop/guardrails.py
──────────────────────
Production-grade, zero-token, multi-layer input guardrail for SOP generation.

Architecture (3 layers — all pure Python, no LLM calls, no external deps):

  Layer 1 — Hard rules      (~0 ms)  Absolute disqualifiers: too short, gibberish,
                                      prompt injection attempt. Any failure → 400.

  Layer 2 — Scoring engine  (~1 ms)  4 keyword categories (service, problem, action,
                                      environment). Each category contributes points.
                                      Total score must meet the threshold (default 3).
                                      Cloud-agnostic: AWS, GCP, Azure, on-prem, OSS tools
                                      all score equally.

  Layer 3 — Domain check    (~1 ms)  Confirms the prompt has at least one signal from
                                      each of the 3 mandatory axes:
                                        ① infrastructure / technology
                                        ② problem type
                                        ③ operational objective
                                      All three must be present.

Public API:
  validate_sop_prompt(prompt: str) -> GuardrailResult

  GuardrailResult.passed  → bool
  GuardrailResult.message → str  (human-readable feedback, returned as 400 to caller)
  GuardrailResult.detail  → dict (layer-by-layer breakdown for logs/observability)

Only used for PROMPT mode — INCIDENT mode skips all layers (context comes from DB).

Design goals:
  • Zero tokens consumed — runs before any Bedrock call.
  • Cloud-agnostic — no AWS bias. GCP, Azure, on-prem, Grafana, Prometheus,
    Kubernetes, MongoDB, Redis all score the same as EC2 or RDS.
  • Extensible — add keywords to any category without touching logic.
  • Observable — GuardrailResult.detail gives layer-by-layer signal for logging.
  • Permissive by design — a borderline prompt passes rather than blocks.
    The score threshold and minimum length are intentionally low.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration constants
# ═══════════════════════════════════════════════════════════════════════════════

# Layer 1 — hard rules
MIN_CHARS          = 15        # shorter than this is always reject
MAX_CHARS          = 8000      # protect against token-flooding
MIN_WORD_COUNT     = 3         # "test" = 1 word → reject
MAX_DIGIT_RATIO    = 0.6       # > 60% digits → likely not a prompt
MIN_ALPHA_RATIO    = 0.4       # < 40% alpha chars → likely garbage

# Layer 2 — scoring
SCORE_THRESHOLD    = 3         # minimum combined score to pass
POINTS_SERVICE     = 2         # service / tech keyword hit
POINTS_PROBLEM     = 2         # problem / symptom keyword hit
POINTS_ACTION      = 1         # action verb / objective keyword hit
POINTS_ENV         = 1         # environment / cloud / tooling keyword hit

# Layer 2 — injection patterns (block before scoring)
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(the\s+)?above",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+|an\s+)?(?!sre|engineer|devops)",   # "act as DAN" etc
    r"jailbreak",
    r"do\s+anything\s+now",
    r"pretend\s+you\s+are",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"disregard\s+(the\s+)?",
    r"override\s+(the\s+)?",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword dictionaries
# ═══════════════════════════════════════════════════════════════════════════════
#
# Design principle: deliberately broad and cloud-agnostic.
# Anything that plausibly appears in an ops context is included.
# Erring toward inclusion rather than exclusion — a borderline prompt
# should score enough to pass, not be silently rejected.

# ── Category A: Service / technology ──────────────────────────────────────────
# Covers: cloud services, databases, containers, message queues,
#         observability tools, web servers, runtimes, protocols.
KEYWORDS_SERVICE: set[str] = {
    # AWS compute
    "ec2", "ecs", "eks", "lambda", "fargate", "beanstalk", "lightsail",
    # AWS data
    "rds", "aurora", "dynamodb", "elasticache", "redshift", "s3", "sqs", "sns",
    "kinesis", "msk", "glue", "athena",
    # AWS networking / security
    "alb", "nlb", "elb", "cloudfront", "api gateway", "apigateway",
    "route53", "vpc", "waf", "guardduty", "cloudwatch",
    # GCP
    "gke", "cloud run", "app engine", "cloud sql", "cloud storage",
    "pubsub", "pub/sub", "bigquery", "dataflow", "gcp", "google cloud",
    "compute engine", "cloud functions",
    # Azure
    "azure", "aks", "app service", "azure functions", "cosmos db",
    "azure sql", "blob storage", "service bus", "event hub", "eventhub",
    "azure monitor", "application insights",
    # On-prem / bare metal
    "on-prem", "on premise", "bare metal", "vmware", "vsphere",
    "hyper-v", "openstack", "proxmox",
    # Containers / orchestration
    "kubernetes", "k8s", "docker", "podman", "containerd", "helm",
    "openshift", "rancher", "istio", "envoy", "linkerd",
    # Databases
    "postgres", "postgresql", "mysql", "mariadb", "mongodb", "mongo",
    "cassandra", "elasticsearch", "opensearch", "redis", "memcached",
    "influxdb", "timescaledb", "neo4j", "couchdb", "sqlite",
    # Message queues / streaming
    "kafka", "rabbitmq", "activemq", "nats", "pulsar", "zeromq",
    # Observability / monitoring
    "grafana", "prometheus", "alertmanager", "loki", "tempo", "jaeger",
    "zipkin", "datadog", "newrelic", "dynatrace", "splunk",
    "cloudwatch", "pagerduty", "opsgenie", "victoria metrics",
    # Web servers / proxies
    "nginx", "apache", "haproxy", "traefik", "caddy", "varnish",
    # Runtimes / languages (when used as service names)
    "nodejs", "node.js", "python", "java", "golang", "go service",
    "dotnet", ".net", "ruby", "php",
    # CI/CD / infra tools
    "jenkins", "gitlab", "github actions", "argocd", "flux", "terraform",
    "ansible", "puppet", "chef", "packer",
    # Generic
    "service", "application", "app", "api", "microservice", "server",
    "cluster", "node", "pod", "container", "instance", "vm",
    "database", "db", "queue", "cache", "load balancer",
    "pipeline", "job", "worker", "daemon", "process",
}

# ── Category B: Problem / symptom ─────────────────────────────────────────────
# Covers any observable failure mode or degradation signal.
KEYWORDS_PROBLEM: set[str] = {
    # Performance
    "high cpu", "cpu utilization", "cpu spike", "cpu usage",
    "high memory", "memory leak", "oom", "out of memory",
    "high latency", "latency", "slow", "timeout", "response time",
    "throughput", "bottleneck", "degraded", "performance",
    # Availability
    "down", "outage", "unavailable", "unreachable", "not responding",
    "crash", "crashing", "restart", "restarting", "restart loop",
    "crashloopbackoff", "oomkilled", "killed", "stopped",
    # Errors
    "error", "errors", "5xx", "500", "502", "503", "504",
    "4xx", "connection refused", "connection reset", "connection timeout",
    "exception", "panic", "segfault", "core dump",
    "failed", "failure", "failing",
    # Data / storage
    "disk full", "disk usage", "disk pressure", "inode",
    "data loss", "corruption", "replication lag", "replica",
    "connection pool", "max connections", "connection exhaustion",
    # Network
    "packet loss", "network", "dns", "certificate", "cert expired",
    "ssl", "tls", "handshake", "unhealthy", "health check",
    # Queues / async
    "queue depth", "consumer lag", "backlog", "message stuck",
    "dead letter", "dlq",
    # Security / access
    "unauthorized", "403", "401", "access denied", "permission",
    "credential", "secret", "token expired",
    # Generic
    "issue", "problem", "incident", "alert", "alarm",
    "spike", "anomaly", "breach", "threshold", "saturation",
}

# ── Category C: Action / objective ────────────────────────────────────────────
# The user must be asking for something operational.
KEYWORDS_ACTION: set[str] = {
    "runbook", "sop", "playbook", "procedure",
    "troubleshoot", "troubleshooting", "debug", "debugging",
    "mitigate", "mitigation", "remediate", "remediation",
    "resolve", "resolution", "fix", "recover", "recovery",
    "investigate", "investigation", "diagnose", "diagnosis",
    "rollback", "roll back", "restore", "restart",
    "monitor", "monitoring", "alert", "alerting",
    "create", "generate", "write", "build",
    "handle", "manage", "respond", "response",
    "on-call", "oncall", "incident response",
    "how to", "steps to", "guide", "documentation",
    "detect", "prevent", "prevention", "reduce",
}

# ── Category D: Environment / cloud / tooling ─────────────────────────────────
# Extra signal that the context is an ops environment.
KEYWORDS_ENV: set[str] = {
    "production", "prod", "staging", "development", "dev",
    "aws", "gcp", "azure", "google cloud", "digitalocean", "linode",
    "cloudflare", "hetzner", "ovh", "armor cloud", "alibaba cloud",
    "on-prem", "on-premise", "data center", "datacenter",
    "namespace", "deployment", "replica", "autoscaling", "asg",
    "region", "availability zone", "az", "multi-region",
    "terraform", "helm", "ansible", "kubectl", "eksctl",
    "log", "logs", "logging", "metrics", "traces", "tracing",
    "dashboard", "panel", "query", "alert rule",
}

# ── Layer 3 domain axes (must have ≥1 from EACH) ─────────────────────────────
# These are deliberately narrow — the bare minimum to confirm this is
# an ops context, not a generic question about cooking or homework.

DOMAIN_INFRA: set[str] = {
    # any identifiable technology is enough
    "ec2", "ecs", "eks", "rds", "s3", "lambda", "alb", "nlb",
    "gke", "gcp", "azure", "aks",
    "kubernetes", "k8s", "docker", "container", "pod",
    "nginx", "redis", "kafka", "postgres", "mysql", "mongodb",
    "grafana", "prometheus", "elasticsearch",
    "server", "service", "api", "application", "cluster", "database",
    "instance", "vm", "node",
}

DOMAIN_PROBLEM: set[str] = {
    "cpu", "memory", "disk", "latency", "timeout", "crash",
    "down", "error", "fail", "slow", "outage", "spike",
    "connection", "oom", "leak", "degraded", "unavailable",
    "restart", "exception", "5xx", "503", "504", "unhealthy",
}

DOMAIN_OBJECTIVE: set[str] = {
    "runbook", "sop", "playbook", "procedure",
    "troubleshoot", "debug", "mitigate", "resolve", "fix",
    "investigate", "diagnose", "recover", "handle", "respond",
    "create", "generate", "write", "build", "guide",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Result type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailResult:
    passed:  bool
    message: str          # returned to caller on failure
    detail:  dict = field(default_factory=dict)   # for logging / observability

    def __bool__(self) -> bool:
        return self.passed

    def log(self, prompt_snippet: str = "") -> None:
        status = "PASS" if self.passed else "FAIL"
        logger.info(
            f"[Guardrail] {status} | "
            f"prompt='{prompt_snippet[:60]}...' | "
            f"detail={self.detail}"
        )
        if not self.passed:
            logger.warning(f"[Guardrail] Blocked: {self.message}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Hard rules
# ═══════════════════════════════════════════════════════════════════════════════

def _layer1_hard_rules(prompt: str) -> GuardrailResult | None:
    """
    Absolute disqualifiers. Returns a failed GuardrailResult if any rule
    triggers, otherwise None (pass through to Layer 2).
    """
    stripped = prompt.strip()
    lower    = stripped.lower()

    # ── Length checks ──────────────────────────────────────────────────────
    if len(stripped) < MIN_CHARS:
        return GuardrailResult(
            passed=False,
            message=(
                "Your prompt is too short to generate a meaningful runbook. "
                "Please describe the service, the problem, and what you need "
                "the runbook to cover. "
                "Example: 'Create a runbook for ECS high CPU in production.'"
            ),
            detail={"layer": 1, "rule": "min_chars", "value": len(stripped)},
        )

    if len(stripped) > MAX_CHARS:
        return GuardrailResult(
            passed=False,
            message=(
                f"Prompt exceeds the maximum allowed length ({MAX_CHARS} characters). "
                "Please shorten your description."
            ),
            detail={"layer": 1, "rule": "max_chars", "value": len(stripped)},
        )

    # ── Word count ─────────────────────────────────────────────────────────
    word_count = len(stripped.split())
    if word_count < MIN_WORD_COUNT:
        return GuardrailResult(
            passed=False,
            message=(
                "Please provide more context. A valid SOP request needs at least "
                "a service name, the issue you're seeing, and what the runbook "
                "should help with. "
                "Example: 'Redis connection timeout runbook for the payments service.'"
            ),
            detail={"layer": 1, "rule": "min_words", "value": word_count},
        )

    # ── Character composition ──────────────────────────────────────────────
    alpha_count = sum(1 for c in stripped if c.isalpha())
    digit_count = sum(1 for c in stripped if c.isdigit())
    total       = max(len(stripped), 1)

    if alpha_count / total < MIN_ALPHA_RATIO:
        return GuardrailResult(
            passed=False,
            message=(
                "Your prompt appears to contain mostly non-text characters. "
                "Please describe the service and problem in plain language."
            ),
            detail={
                "layer": 1,
                "rule": "min_alpha_ratio",
                "value": round(alpha_count / total, 2),
            },
        )

    if digit_count / total > MAX_DIGIT_RATIO:
        return GuardrailResult(
            passed=False,
            message=(
                "Your prompt contains too many numbers relative to text. "
                "Please describe the issue in plain language."
            ),
            detail={
                "layer": 1,
                "rule": "max_digit_ratio",
                "value": round(digit_count / total, 2),
            },
        )

    # ── Prompt injection detection ─────────────────────────────────────────
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            logger.warning(
                f"[Guardrail-L1] Injection pattern matched: '{pattern}' "
                f"in prompt: '{stripped[:80]}'"
            )
            return GuardrailResult(
                passed=False,
                message=(
                    "Your prompt contains content that cannot be processed. "
                    "Please describe the operational issue you need a runbook for."
                ),
                detail={"layer": 1, "rule": "injection", "pattern": pattern},
            )

    return None   # all hard rules passed


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Scoring engine
# ═══════════════════════════════════════════════════════════════════════════════

class _ScoreBreakdown(NamedTuple):
    service_hits:  list[str]
    problem_hits:  list[str]
    action_hits:   list[str]
    env_hits:      list[str]
    total:         int


def _score_prompt(lower: str) -> _ScoreBreakdown:
    """
    Scan the lowercased prompt against all four keyword categories.
    Returns matched keywords and total score.

    Matching is word-boundary aware so "pod" doesn't match inside "episode".
    Multi-word phrases (e.g. "api gateway") are checked with simple substring
    since they can't appear accidentally.
    """
    def _hits(keywords: set[str]) -> list[str]:
        matched = []
        for kw in keywords:
            if " " in kw:
                # multi-word phrase: substring match
                if kw in lower:
                    matched.append(kw)
            else:
                # single word: word-boundary match
                if re.search(rf"\b{re.escape(kw)}\b", lower):
                    matched.append(kw)
        return matched

    service_hits = _hits(KEYWORDS_SERVICE)
    problem_hits = _hits(KEYWORDS_PROBLEM)
    action_hits  = _hits(KEYWORDS_ACTION)
    env_hits     = _hits(KEYWORDS_ENV)

    total = (
        len(service_hits) * POINTS_SERVICE
        + len(problem_hits) * POINTS_PROBLEM
        + len(action_hits) * POINTS_ACTION
        + len(env_hits) * POINTS_ENV
    )

    return _ScoreBreakdown(service_hits, problem_hits, action_hits, env_hits, total)


def _layer2_scoring(prompt: str) -> GuardrailResult | None:
    """
    Score the prompt and reject if below the threshold.
    Returns a failed GuardrailResult or None (pass through to Layer 3).
    """
    lower = prompt.lower()
    score = _score_prompt(lower)

    detail = {
        "layer":        2,
        "score":        score.total,
        "threshold":    SCORE_THRESHOLD,
        "service_hits": score.service_hits[:5],   # cap to keep logs readable
        "problem_hits": score.problem_hits[:5],
        "action_hits":  score.action_hits[:5],
        "env_hits":     score.env_hits[:3],
    }

    if score.total < SCORE_THRESHOLD:
        # Build targeted feedback based on which categories are missing
        missing = []
        if not score.service_hits:
            missing.append(
                "service or technology (e.g. ECS, Kubernetes, Redis, Nginx, Prometheus)"
            )
        if not score.problem_hits:
            missing.append(
                "problem or symptom (e.g. high CPU, memory leak, timeout, crash)"
            )
        if not score.action_hits and not score.service_hits and not score.problem_hits:
            missing.append(
                "operational objective (e.g. troubleshoot, create runbook, mitigate)"
            )

        if missing:
            feedback = (
                "Your prompt needs more operational context. Please include: "
                + " and ".join(missing) + ". "
            )
        else:
            feedback = (
                "Your prompt needs more detail. "
            )

        feedback += (
            "Example: 'Create a runbook for Kubernetes pod CrashLoopBackOff "
            "on the orders service in production.'"
        )

        return GuardrailResult(passed=False, message=feedback, detail=detail)

    return None   # score passed


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Domain context check
# ═══════════════════════════════════════════════════════════════════════════════

def _layer3_domain(prompt: str) -> GuardrailResult | None:
    """
    Confirm the prompt has at least one keyword from each of the three
    mandatory operational axes:
      ① infrastructure / technology
      ② problem type
      ③ operational objective

    This catches prompts that scored on quantity (many generic words) but
    are missing one axis entirely (e.g. "create a guide" with no tech/problem).
    Returns a failed GuardrailResult or None (all clear).
    """
    lower = prompt.lower()

    def _any_hit(keywords: set[str]) -> bool:
        for kw in keywords:
            if " " in kw:
                if kw in lower:
                    return True
            else:
                if re.search(rf"\b{re.escape(kw)}\b", lower):
                    return True
        return False

    has_infra     = _any_hit(DOMAIN_INFRA)
    has_problem   = _any_hit(DOMAIN_PROBLEM)
    has_objective = _any_hit(DOMAIN_OBJECTIVE)

    detail = {
        "layer":         3,
        "has_infra":     has_infra,
        "has_problem":   has_problem,
        "has_objective": has_objective,
    }

    if has_infra and has_problem and has_objective:
        return None   # all three axes present

    # Build specific feedback about which axis is missing
    missing_axes = []
    if not has_infra:
        missing_axes.append(
            "a service or technology (e.g. ECS, Kubernetes, Redis, Nginx, Grafana)"
        )
    if not has_problem:
        missing_axes.append(
            "a specific problem or alert (e.g. CPU spike, memory leak, connection timeout)"
        )
    if not has_objective:
        missing_axes.append(
            "an operational goal (e.g. troubleshoot, create runbook, mitigate, recover)"
        )

    message = (
        "Your request is missing: "
        + ", and ".join(missing_axes)
        + ". All three are required for a useful runbook. "
        "Example: 'Create a runbook [objective] for Redis memory exhaustion [problem] "
        "on the checkout service [service] in production [environment].'"
    )

    return GuardrailResult(passed=False, message=message, detail=detail)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def validate_sop_prompt(prompt: str) -> GuardrailResult:
    """
    Run all three guardrail layers against a user SOP prompt.

    Pipeline (short-circuits on first failure):
      Layer 1 → Layer 2 → Layer 3 → PASS

    Args:
        prompt: The raw user prompt string from the API request.

    Returns:
        GuardrailResult
          .passed  True if all layers cleared, False on any failure.
          .message Human-readable rejection reason (return as 400 body).
          .detail  Per-layer diagnostic dict (log this, don't expose to user).

    Usage in process_sop.py (PROMPT mode only):
        result = validate_sop_prompt(user_prompt)
        result.log(prompt_snippet=user_prompt)
        if not result.passed:
            _mark_invalid_prompt(db_id, result.message)
            return

    INCIDENT mode: skip this function entirely — context comes from DB.
    """
    # Layer 1
    r = _layer1_hard_rules(prompt)
    if r is not None:
        r.log(prompt[:60])
        return r

    # Layer 2
    r = _layer2_scoring(prompt)
    if r is not None:
        r.log(prompt[:60])
        return r

    # Layer 3
    r = _layer3_domain(prompt)
    if r is not None:
        r.log(prompt[:60])
        return r

    # All layers passed
    result = GuardrailResult(
        passed=True,
        message="",
        detail={"layers_passed": [1, 2, 3]},
    )
    result.log(prompt[:60])
    return result