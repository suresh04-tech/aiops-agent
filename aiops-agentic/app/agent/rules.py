"""
app/agent/rules.py
──────────────────
Contains static dictionaries and rules used for evaluating signals and mapping triage states.
"""

import re

# Signal taxonomy (P9)
# Each rule maps a pattern to:
#   type        — signal type: primary_root_cause / supporting_signal /
#                              infra_symptom / downstream_impact
#   rca_type    — machine-readable root cause class
#   description — human-readable explanation
#   confidence  — float 0-1 for this rule alone

RCA_CATEGORIES: dict[str, str] = {
    "database_config_error": "deterministic",
    "iam_permission_error": "deterministic",
    "aws_credentials_error": "deterministic",
    "disk_full": "deterministic",
    "oom_kill": "deterministic",
    "instance_stopped": "deterministic",
    "instance_terminated": "deterministic",

    "database_connectivity_failure": "investigate",
    "upstream_connection_refused": "investigate",
    "http_5xx_errors": "investigate",
    "timeout_errors": "investigate",
}

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
        "type": "investigation_hypothesis",
        "rca_type": "database_connectivity_failure",
        "description": "Database connection refused — DB host is down or port blocked.",
        "confidence": 0.55,
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
        "type": "investigation_hypothesis",
        "rca_type": "upstream_connection_refused",
        "description": "Upstream service refused the connection — likely crashed or overloaded.",
        "confidence": 0.55,
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
