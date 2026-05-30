import boto3
import logging
from botocore.config import Config
from app.utils.db import get_db

logger = logging.getLogger(__name__)


class AWSClientFactory:
    """
    Factory to dynamically create boto3 clients using credentials stored
    in meyiconnect.insight_projects (keyed by project tag).

    Used for all AWS services EXCEPT Bedrock:
      - ec2, cloudwatch, logs, elbv2, cloudtrail, …

    Credentials are resolved at construction time from:
        insight_projects WHERE tag = <project_tag>
    """

    def __init__(self, project_tag: str):
        self.project_tag = project_tag
        self._load_project_credentials()

    def _load_project_credentials(self) -> None:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT aws_access_key_id,
                           aws_secret_access_key,
                           aws_region,
                           dependencies
                    FROM meyiconnect.insight_projects
                    WHERE tag = %s
                    LIMIT 1
                    """,
                    (self.project_tag,),
                )
                row = cur.fetchone()

        if not row:
            raise ValueError(
                f"No project found with tag '{self.project_tag}' in insight_projects"
            )

        self.access_key  = row["aws_access_key_id"]
        self.secret_key  = row["aws_secret_access_key"]
        self.region      = row["aws_region"] or "us-east-1"
        self.dependencies = row["dependencies"] or []

        if not self.access_key or not self.secret_key:
            raise ValueError(
                f"Project '{self.project_tag}' has no AWS credentials configured"
            )

        logger.info(
            f"[AWSClientFactory] Loaded credentials for project '{self.project_tag}' "
            f"region={self.region}"
        )

    def get_client(self, service_name: str, region_name: str = None,
                   config: Config = None):
        """Create a boto3 client for the given AWS service."""
        return boto3.client(
            service_name,
            region_name=region_name or self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=config,
        )


class BedrockClientFactory:
    """
    Factory to create a boto3 bedrock-runtime client using credentials from
    meyiconnect.insight_settings.

    Separate from AWSClientFactory so project-level AWS creds and
    Bedrock creds remain independent.
    """

    def __init__(self):
        self._load_settings()

    def _load_settings(self) -> None:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT aws_bedrock_access_key_id,
                           aws_bedrock_secret_access_key,
                           aws_bedrock_region
                    FROM meyiconnect.insight_settings
                    LIMIT 1
                    """
                )
                row = cur.fetchone()

        if not row:
            raise ValueError("No settings found in insight_settings table")

        self.access_key = row["aws_bedrock_access_key_id"]
        self.secret_key = row["aws_bedrock_secret_access_key"]
        self.region     = row["aws_bedrock_region"] or "us-east-1"

        if not self.access_key or not self.secret_key:
            raise ValueError(
                "Bedrock credentials (aws_bedrock_access_key_id / "
                "aws_bedrock_secret_access_key) not configured in insight_settings"
            )

        logger.info(
            f"[BedrockClientFactory] Loaded Bedrock credentials "
            f"region={self.region}"
        )

    def get_bedrock_runtime_client(self):
        """Return a boto3 bedrock-runtime client using insight_settings creds."""
        return boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
