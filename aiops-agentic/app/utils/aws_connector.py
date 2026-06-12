import boto3
import logging
import re
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError
from app.utils.db import get_db

logger = logging.getLogger(__name__)

class AWSAuthenticationError(Exception):
    pass



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
        self.use_role = False
        
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            logger.info("[AWSClientFactory] IAM role detected.")
            logger.info("[AWSClientFactory] Using IAM role credentials.")
            self.use_role = True
        else:
            logger.info("[AWSClientFactory] IAM role unavailable.")
            logger.info("[AWSClientFactory] Falling back to database credentials.")
            
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
            raise AWSAuthenticationError(
                f"No AWS credentials available for project_tag {self.project_tag}"
            )

        self.region      = row["aws_region"] or "us-east-1"
        self.dependencies = row["dependencies"] or []

        if not self.use_role:
            self.access_key  = row["aws_access_key_id"]
            self.secret_key  = row["aws_secret_access_key"]

            if not self.access_key or not self.secret_key:
                raise AWSAuthenticationError(
                    f"No AWS credentials available for project_tag {self.project_tag}"
                )

            logger.info(
                f"[AWSClientFactory] Loaded credentials for project '{self.project_tag}' "
                f"region={self.region}"
            )

    def get_client(self, service_name: str, region_name: str = None,
                   config: Config = None):
        """Create a boto3 client for the given AWS service."""
        try:
            if self.use_role:
                client = boto3.client(
                    service_name,
                    region_name=region_name or self.region,
                    config=config,
                )
            else:
                client = boto3.client(
                    service_name,
                    region_name=region_name or self.region,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=config,
                )
            
            self._validate_client(client, service_name)
            return client

        except AWSAuthenticationError:
            raise
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as e:
            if "Invalid region" in str(e) or "Illegal region" in str(e):
                raise AWSAuthenticationError(f"Invalid AWS region configured for project_tag {self.project_tag}")
            raise AWSAuthenticationError(f"Unable to reach AWS service {service_name}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'InvalidClientTokenId':
                raise AWSAuthenticationError("AWS authentication failed: InvalidClientTokenId")
            elif error_code == 'SignatureDoesNotMatch':
                raise AWSAuthenticationError("AWS authentication failed: SignatureDoesNotMatch")
            elif error_code in ['AccessDenied', 'UnauthorizedOperation']:
                if self.use_role:
                    default_actions = {
                        'cloudwatch': 'cloudwatch:ListMetrics',
                        'ec2': 'ec2:DescribeRegions',
                        'logs': 'logs:DescribeLogGroups',
                        'elbv2': 'elasticloadbalancing:DescribeLoadBalancers',
                        'elb': 'elasticloadbalancing:DescribeLoadBalancers'
                    }
                    action = self._extract_action(error_msg, default_actions.get(service_name, f"{service_name}:Unknown"))
                    raise AWSAuthenticationError(f"IAM role lacks required permission {action}")
                else:
                    raise AWSAuthenticationError(f"AWS permission denied for service {service_name}: {error_code}")
            elif error_code == 'UnrecognizedClientException':
                raise AWSAuthenticationError("AWS authentication failed: UnrecognizedClientException")
            elif error_code == 'AuthFailure':
                raise AWSAuthenticationError("AWS authentication failed: AuthFailure")
            else:
                raise AWSAuthenticationError(f"Unexpected AWS error: {error_msg}")
        except Exception as e:
            if "Invalid region" in str(e) or "Illegal region" in str(e):
                raise AWSAuthenticationError(f"Invalid AWS region configured for project_tag {self.project_tag}")
            raise AWSAuthenticationError(f"Unexpected AWS error: {str(e)}")

    def _extract_action(self, error_msg: str, default_action: str) -> str:
        match = re.search(r'perform:\s*([a-zA-Z0-9]+:[a-zA-Z0-9]+)', error_msg)
        if match:
            return match.group(1)
        return default_action

    def _validate_client(self, client, service_name: str):
        if service_name == 'cloudwatch':
            client.list_metrics(MaxResults=1)
            logger.info("[AWSClientFactory] CloudWatch permission validation succeeded.")
        elif service_name == 'ec2':
            client.describe_regions()
            logger.info("[AWSClientFactory] EC2 permission validation succeeded.")
        elif service_name == 'logs':
            client.describe_log_groups(limit=1)
            logger.info("[AWSClientFactory] Logs permission validation succeeded.")
        elif service_name in ['elbv2', 'elb']:
            client.describe_load_balancers(PageSize=1)
            logger.info("[AWSClientFactory] ELB permission validation succeeded.")


class BedrockClientFactory:
    """
    Factory to create a boto3 bedrock-runtime client using IAM role or credentials from
    meyiconnect.insight_settings.

    Separate from AWSClientFactory so project-level AWS creds and
    Bedrock creds remain independent.
    """

    def __init__(self):
        self.use_role = False
        
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            logger.info("[BedrockClientFactory] IAM role detected.")
            logger.info("[BedrockClientFactory] Using IAM role credentials.")
            self.use_role = True
        else:
            logger.info("[BedrockClientFactory] IAM role unavailable.")
            logger.info("[BedrockClientFactory] Falling back to database credentials.")
            
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
            if not self.use_role:
                raise ValueError("No settings found in insight_settings table")
            else:
                self.region = "us-east-1"
                return

        self.region = row["aws_bedrock_region"] or "us-east-1"

        if not self.use_role:
            self.access_key = row["aws_bedrock_access_key_id"]
            self.secret_key = row["aws_bedrock_secret_access_key"]

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
        """Return a boto3 bedrock-runtime client using insight_settings creds or IAM role."""
        if self.use_role:
            return boto3.client(
                "bedrock-runtime",
                region_name=self.region,
            )
        else:
            return boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
