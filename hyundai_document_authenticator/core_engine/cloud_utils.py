"""
cloud_utils.py

Production-ready AWS S3 utility functions and S3Client class for cloud storage operations.

Features:
- Download/upload files and folders to/from S3.
- Check if a file exists in S3.
- Generate presigned URLs.
- Retrieve S3 object metadata.
- Robust error handling, logging, and retry logic.

Dependencies:
    pip install boto3 tenacity

Note:
    Do not configure logging handlers in this module. Applications/CLIs should
    configure handlers; this module only defines a module-level logger.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
import concurrent.futures

# Configure module-level logger
logger = logging.getLogger(__name__)

class S3Client:
    """
    AWS S3 Client wrapper for robust, production-grade S3 operations.
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize the S3 client.

        Args:
            aws_access_key_id (Optional[str]): AWS access key ID.
            aws_secret_access_key (Optional[str]): AWS secret access key.
            region_name (Optional[str]): AWS region.
            max_retries (int): Max retry attempts for operations.
        """
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.s3 = self.session.client('s3')
        self.max_retries = max_retries

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_file(self, bucket: str, s3_key: str, local_path: Union[str, Path]) -> str:
        """
        Download a single file from S3.

        Args:
            bucket (str): S3 bucket name.
            s3_key (str): S3 object key.
            local_path (Union[str, Path]): Local file path to save.

        Returns:
            str: Path to the downloaded file.

        Raises:
            ClientError: If download fails.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.s3.download_file(bucket, s3_key, str(local_path))
            logger.info("Downloaded s3://%s/%s to %s", bucket, s3_key, local_path)
            return str(local_path)
        except ClientError as e:
            logger.error("Failed to download s3://%s/%s: %s", bucket, s3_key, e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upload_file(self, local_path: Union[str, Path], bucket: str, s3_key: str, extra_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a file to S3.

        Args:
            local_path (Union[str, Path]): Local file path.
            bucket (str): S3 bucket name.
            s3_key (str): S3 object key.
            extra_args (Optional[Dict[str, Any]]): Extra upload args (e.g., ACL, ContentType).

        Returns:
            str: S3 URI of the uploaded file.

        Raises:
            ClientError: If upload fails.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        try:
            self.s3.upload_file(str(local_path), bucket, s3_key, ExtraArgs=extra_args or {})
            logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, s3_key)
            return f"s3://{bucket}/{s3_key}"
        except ClientError as e:
            logger.error("Failed to upload %s to s3://%s/%s: %s", local_path, bucket, s3_key, e)
            raise

    def file_exists(self, bucket: str, s3_key: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            bucket (str): S3 bucket name.
            s3_key (str): S3 object key.

        Returns:
            bool: True if file exists, False otherwise.
        """
        try:
            self.s3.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return False
            logger.error("Error checking existence of s3://%s/%s: %s", bucket, s3_key, e)
            raise

    def get_metadata(self, bucket: str, s3_key: str) -> Dict[str, Any]:
        """
        Get metadata for an S3 object.

        Args:
            bucket (str): S3 bucket name.
            s3_key (str): S3 object key.

        Returns:
            Dict[str, Any]: Metadata dictionary.

        Raises:
            ClientError: If retrieval fails.
        """
        try:
            response = self.s3.head_object(Bucket=bucket, Key=s3_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            logger.error("Failed to get metadata for s3://%s/%s: %s", bucket, s3_key, e)
            raise

    def generate_presigned_url(self, bucket: str, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            bucket (str): S3 bucket name.
            s3_key (str): S3 object key.
            expiration (int): Expiration in seconds.

        Returns:
            str: Presigned URL.

        Raises:
            ClientError: If generation fails.
        """
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            logger.info("Generated presigned URL for s3://%s/%s", bucket, s3_key)
            return url
        except ClientError as e:
            logger.error("Failed to generate presigned URL for s3://%s/%s: %s", bucket, s3_key, e)
            raise

    def sync_folder_from_s3(self, bucket: str, s3_prefix: str, local_dir: Union[str, Path], max_workers: int = 8) -> List[str]:
        """
        Download all files from an S3 prefix to a local directory (sync).

        Args:
            bucket (str): S3 bucket name.
            s3_prefix (str): S3 prefix (folder path).
            local_dir (Union[str, Path]): Local directory to sync to.
            max_workers (int): Number of parallel download threads.

        Returns:
            List[str]: List of downloaded file paths.

        Raises:
            ClientError: If listing or download fails.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []

        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            files_to_download = []
            for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
                for obj in page.get('Contents', []) or []:
                    s3_key = obj['Key']
                    rel_path = Path(s3_key[len(s3_prefix):].lstrip('/\\'))
                    local_path = local_dir / rel_path
                    files_to_download.append((s3_key, local_path))

            def _download(args: tuple[str, Path]) -> Optional[str]:
                """Worker to download a single object to a local path.

                Args:
                    args: Tuple of (s3_key, local_path) where local_path is a Path.

                Returns:
                    Optional[str]: The string path to the downloaded file on success; None on failure.
                """
                s3_key, local_path = args
                try:
                    self.download_file(bucket, s3_key, local_path)
                    return str(local_path)
                except Exception as e:
                    # TODO: Narrow exceptions to ClientError and IO-related errors.
                    logger.error("Failed to download %s: %s", s3_key, e)
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_download, files_to_download))
                downloaded_files = [r for r in results if r]

            logger.info("Synced %d files from s3://%s/%s to %s", len(downloaded_files), bucket, s3_prefix, local_dir)
            return downloaded_files

        except ClientError as e:
            logger.error("Failed to sync from s3://%s/%s: %s", bucket, s3_prefix, e)
            raise

# --- Module-level default client and convenience functions ---

# TODO: Consider lazy initialization to avoid AWS client creation at import time.
_default_client = S3Client()

def download_from_s3(bucket: str, s3_key: str, local_path: Union[str, Path]) -> str:
    """Download a file from S3 using the default client."""
    return _default_client.download_file(bucket, s3_key, local_path)

def upload_to_s3(local_path: Union[str, Path], bucket: str, s3_key: str, extra_args: Optional[Dict[str, Any]] = None) -> str:
    """Upload a file to S3 using the default client."""
    return _default_client.upload_file(local_path, bucket, s3_key, extra_args)

def check_s3_file_exists(bucket: str, s3_key: str) -> bool:
    """Check if a file exists in S3 using the default client."""
    return _default_client.file_exists(bucket, s3_key)

def get_s3_file_metadata(bucket: str, s3_key: str) -> Dict[str, Any]:
    """Get S3 file metadata using the default client."""
    return _default_client.get_metadata(bucket, s3_key)

def generate_presigned_url(bucket: str, s3_key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL using the default client."""
    return _default_client.generate_presigned_url(bucket, s3_key, expiration)

def sync_s3_to_local(bucket: str, s3_prefix: str, local_dir: Union[str, Path], max_workers: int = 8) -> List[str]:
    """Sync an S3 prefix to a local directory using the default client."""
    return _default_client.sync_folder_from_s3(bucket, s3_prefix, local_dir, max_workers)
