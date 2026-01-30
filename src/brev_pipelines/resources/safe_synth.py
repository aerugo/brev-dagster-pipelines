"""NVIDIA Safe Synthesizer resource for Dagster.

Manages Safe Synthesizer API calls for synthetic data generation.
Safe Synthesizer runs always-on alongside nim-llm via KAI fractional GPU allocation.

Uses NVIDIA's official NeMo Microservices Helm Chart (nemo-safe-synthesizer).

How it works:
1. Safe Synthesizer service is always running (replicaCount=1)
2. KAI Scheduler allocates fractional GPU: safe-synth 40Gi + nim-llm 25Gi = 65Gi << 141Gi H200
3. Dagster calls the synthesis API directly (no scaling needed)
4. nim-llm stays running throughout for PII classification
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests
from dagster import ConfigurableResource
from pydantic import Field

from brev_pipelines.resources.safe_synth_retry import (
    SafeSynthError,
    SafeSynthJobFailedError,
    SafeSynthServerError,
    SafeSynthTimeoutError,
)
from brev_pipelines.types import SafeSynthConfig, SafeSynthEvaluationResult, SafeSynthJobStatus

if TYPE_CHECKING:
    from datetime import datetime

    from brev_pipelines.types import K8sBatchV1Api, K8sCoreV1Api


class SafeSynthesizerResource(ConfigurableResource):
    """NVIDIA Safe Synthesizer resource using Kubernetes Jobs.

    Supports two modes:
    1. API mode: Calls Safe Synthesizer service API (requires NDS data upload)
    2. Job mode: Creates Kubernetes Jobs for on-demand synthesis

    Attributes:
        namespace: Kubernetes namespace for Safe Synthesizer jobs.
        image: Safe Synthesizer container image.
        service_endpoint: Safe Synthesizer service endpoint for API calls.
        nds_endpoint: NeMo Data Store endpoint for HuggingFace-compatible uploads.
        nds_token: Access token for NDS authentication.
        nds_repo: Repository name in NDS for data uploads.
        poll_interval: Job status polling interval in seconds.
        max_wait_time: Maximum wait time for job completion in seconds.
        gpu_memory: GPU memory allocation for KAI Scheduler.
        priority_class: Kubernetes priority class for preemption.
    """

    namespace: str = Field(
        default="nvidia-ai",
        description="Kubernetes namespace for Safe Synthesizer",
    )
    image: str = Field(
        default="nvcr.io/nvidia/nemo-microservices/safe-synthesizer:25.12",
        description="Safe Synthesizer container image",
    )
    service_endpoint: str = Field(
        default="http://nemo-safe-synthesizer.nvidia-ai.svc.cluster.local:8000",
        description="Safe Synthesizer API endpoint",
    )
    nds_endpoint: str = Field(
        default="http://nemo-data-store.nvidia-ai.svc.cluster.local:3000",
        description="NeMo Data Store endpoint (Gitea-based HuggingFace-compatible)",
    )
    nds_token: str = Field(
        default="",
        description="NDS access token (from nds-credentials secret)",
    )
    nds_repo: str = Field(
        default="admin/central-bank-speeches",
        description="NDS repository for data uploads",
    )
    poll_interval: int = Field(
        default=30,
        description="Job status poll interval in seconds",
    )
    max_wait_time: int = Field(
        default=7200,
        description="Max job wait time in seconds (default 2 hours)",
    )
    gpu_memory: str = Field(
        default="40Gi",
        description="GPU memory allocation for KAI Scheduler",
    )
    priority_class: str = Field(
        default="build-preemptible",
        description="Kubernetes priority class for batch jobs",
    )
    use_mock_fallback: bool = Field(
        default=True,
        description="Use mock synthesis if Safe Synthesizer unavailable (for local dev)",
    )

    def _get_nds_token(self) -> str:
        """Get NDS token from environment at runtime.

        This reads the token from the NDS_TOKEN environment variable at runtime,
        rather than at definition time, to handle cases where the secret is
        mounted after the module is imported.
        """
        import os

        token = os.environ.get("NDS_TOKEN", "") or self.nds_token
        if not token:
            raise ValueError(
                "NDS_TOKEN environment variable is not set. "
                "Ensure the nds-credentials secret is mounted."
            )
        return token

    def _raise_for_nds_status(self, response: requests.Response) -> None:
        """Raise retryable SafeSynthServerError for NDS auth and server errors.

        Converts HTTP 401/403/5xx errors from NDS (Gitea) into SafeSynthServerError,
        which is in RETRYABLE_ERRORS and will be retried by retry_safe_synth_call.
        This handles transient NDS failures (e.g., must-change-password bug after pod
        restart invalidates API tokens with 403).
        """
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and (
                e.response.status_code in (401, 403)
                or e.response.status_code >= 500
            ):
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "NDS returned %d: %s",
                    e.response.status_code,
                    e.response.text[:500],
                )
                raise SafeSynthServerError(
                    e.response.status_code, e.response.text
                ) from e
            raise

    def _get_k8s_batch_client(self) -> K8sBatchV1Api:
        """Get Kubernetes batch API client."""
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.BatchV1Api()  # type: ignore[no-any-return]

    def _get_k8s_core_client(self) -> K8sCoreV1Api:
        """Get Kubernetes core API client."""
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.CoreV1Api()  # type: ignore[no-any-return]

    def create_synthesis_job(
        self,
        job_name: str,
        input_data_path: str,
        output_data_path: str,
        synth_config: SafeSynthConfig | None = None,
    ) -> str:
        """Create a Kubernetes Job for synthetic data generation.

        Args:
            job_name: Unique name for the job.
            input_data_path: Path to input data in MinIO/LakeFS.
            output_data_path: Path for output data.
            synth_config: Synthesis configuration.

        Returns:
            Job name for tracking.
        """
        from kubernetes import client

        batch_api = self._get_k8s_batch_client()

        # Default synthesis config
        config_defaults: dict[str, object] = {
            "epsilon": 1.0,
            "delta": 1e-5,
            "piiReplacement": True,
            "temperature": 0.7,
            "runMiaEvaluation": True,
            "runAiaEvaluation": True,
        }
        if synth_config:
            config_defaults.update(synth_config)

        # Create Job spec
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={
                    "app": "safe-synth-job",
                    "kai.scheduler/queue": "batch-queue",
                },
            ),
            spec=client.V1JobSpec(
                backoff_limit=0,
                ttl_seconds_after_finished=3600,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "safe-synth-job",
                            "kai.scheduler/queue": "batch-queue",
                        },
                        annotations={
                            "kai.scheduler.nvidia.com/gpu-memory": self.gpu_memory,
                        },
                    ),
                    spec=client.V1PodSpec(
                        scheduler_name="kai-scheduler",
                        priority_class_name=self.priority_class,
                        restart_policy="Never",
                        runtime_class_name="nvidia",
                        image_pull_secrets=[client.V1LocalObjectReference(name="ngc-image-pull")],
                        tolerations=[
                            client.V1Toleration(
                                key="nvidia.com/gpu",
                                operator="Exists",
                                effect="NoSchedule",
                            )
                        ],
                        containers=[
                            client.V1Container(
                                name="safe-synth",
                                image=self.image,
                                env=[
                                    client.V1EnvVar(
                                        name="NGC_API_KEY",
                                        value_from=client.V1EnvVarSource(
                                            secret_key_ref=client.V1SecretKeySelector(
                                                name="ngc-credentials",
                                                key="api-key",
                                            )
                                        ),
                                    ),
                                    client.V1EnvVar(name="LOG_LEVEL", value="INFO"),
                                    client.V1EnvVar(name="INPUT_PATH", value=input_data_path),
                                    client.V1EnvVar(name="OUTPUT_PATH", value=output_data_path),
                                    client.V1EnvVar(
                                        name="SYNTH_CONFIG",
                                        value=json.dumps(config_defaults),
                                    ),
                                ],
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        "memory": "32Gi",
                                        "cpu": "4",
                                        "nvidia.com/gpu": "1",
                                    },
                                    limits={
                                        "memory": "64Gi",
                                        "cpu": "8",
                                        "nvidia.com/gpu": "1",
                                    },
                                ),
                            )
                        ],
                    ),
                ),
            ),
        )

        # Create the job
        batch_api.create_namespaced_job(namespace=self.namespace, body=job)
        return job_name

    def wait_for_job(self, job_name: str) -> SafeSynthJobStatus:
        """Wait for a job to complete.

        Args:
            job_name: Job name to wait for.

        Returns:
            Job status information.

        Raises:
            SafeSynthTimeoutError: If job doesn't complete in max_wait_time.
            SafeSynthJobFailedError: If job fails.
            SafeSynthError: If job not found or other errors.
        """
        from kubernetes.client.rest import ApiException

        batch_api = self._get_k8s_batch_client()
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            try:
                job = batch_api.read_namespaced_job_status(
                    name=job_name,
                    namespace=self.namespace,
                )

                if job.status.succeeded and job.status.succeeded > 0:
                    # Get completion time (Kubernetes returns datetime)
                    completion_time = job.status.completion_time
                    completion_str = ""
                    if completion_time is not None:
                        # Kubernetes returns datetime, cast for type safety
                        completion_str = cast("datetime", completion_time).isoformat()
                    return {
                        "state": "completed",
                        "succeeded": job.status.succeeded,
                        "completion_time": completion_str,
                    }

                if job.status.failed and job.status.failed > 0:
                    # Get pod logs for error details
                    error_msg = self._get_job_logs(job_name)
                    raise SafeSynthJobFailedError(job_name, error_msg)

            except ApiException as e:
                if e.status == 404:
                    raise SafeSynthError(f"Job {job_name} not found") from e
                raise

            time.sleep(self.poll_interval)

        raise SafeSynthTimeoutError(
            f"Job {job_name} did not complete in {self.max_wait_time} seconds"
        )

    def _get_job_logs(self, job_name: str) -> str:
        """Get logs from a job's pod."""
        core_api = self._get_k8s_core_client()

        try:
            pods = core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}",
            )

            if pods.items:
                pod_name = pods.items[0].metadata.name
                logs = core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.namespace,
                    tail_lines=100,
                )
                return logs
        except Exception as e:
            return f"Could not retrieve logs: {e}"

        return "No pods found for job"

    def delete_job(self, job_name: str) -> bool:
        """Delete a job and its pods."""
        from kubernetes.client.rest import ApiException

        batch_api = self._get_k8s_batch_client()

        try:
            batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Foreground",
            )
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            raise

    def synthesize(
        self,
        input_data: list[dict[str, Any]],
        run_id: str,
        config: SafeSynthConfig | None = None,
    ) -> tuple[list[dict[str, Any]], SafeSynthEvaluationResult]:
        """Run full synthesis pipeline via always-on Safe Synthesizer service.

        Safe Synthesizer runs always-on alongside nim-llm via KAI fractional
        GPU allocation. No manual scaling needed.

        If use_mock_fallback is True and the service is unavailable (local dev),
        generates mock synthetic data instead.

        Args:
            input_data: Input data records.
            run_id: Unique run identifier.
            config: Optional synthesis configuration.

        Returns:
            Tuple of (synthetic_data, evaluation_report).
        """
        if not self.health_check() and self.use_mock_fallback:
            return self._generate_mock_synthetic_data(input_data, run_id)

        return self._synthesize_via_api(input_data, config, run_id)

    def _ensure_hf_dataset_exists(self) -> str:
        """Ensure HuggingFace-compatible dataset exists and is functional in NDS.

        Creates the dataset via HF API if it doesn't exist. Also verifies the
        underlying Git repo is accessible (not just present in the DB), handling
        the case where the data-store PVC was recreated but PostgreSQL retains
        stale repo metadata.

        Returns:
            The dataset repo ID (e.g., 'default/speeches-data').
        """
        import logging

        logger = logging.getLogger(__name__)

        repo_name = self.nds_repo.split("/")[-1]
        repo_id = f"default/{repo_name}"
        token = self._get_nds_token()

        # Try to create dataset via HF API
        create_url = f"{self.nds_endpoint}/v1/hf/api/repos/create"
        create_payload = {"type": "dataset", "name": repo_name, "private": False}

        try:
            response = requests.post(
                create_url,
                json=create_payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if response.status_code == 200:
                result: dict[str, str] = response.json()
                url: str = result.get("url", f"default/{repo_name}")
                return url.replace("datasets/", "")
            elif response.status_code == 409:
                # DB says repo exists — verify Git directory is functional
                if self._verify_repo_functional(repo_id, token):
                    return repo_id
                # Stale repo: DB has metadata but Git dir is missing
                logger.warning(
                    "NDS repo %s exists in DB but Git directory is broken. "
                    "Deleting and recreating.",
                    repo_id,
                )
                self._delete_nds_repo(repo_id, token)
                # Retry creation
                retry_resp = requests.post(
                    create_url,
                    json=create_payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
                retry_resp.raise_for_status()
                return repo_id
            else:
                response.raise_for_status()
        except requests.exceptions.HTTPError:
            pass

        return repo_id

    def _verify_repo_functional(self, repo_id: str, token: str) -> bool:
        """Check if a NDS repo's Git directory is actually accessible.

        The Gitea API may report a repo as existing (in PostgreSQL) even when
        the underlying bare Git repository is missing from disk (e.g., after
        PVC recreation). This verifies the repo can serve Git operations.

        Args:
            repo_id: Repository ID (e.g., 'default/central-bank-speeches').
            token: NDS API token.

        Returns:
            True if the repo is functional, False if Git dir is missing.
        """
        try:
            resp = requests.get(
                f"{self.nds_endpoint}/api/v1/repos/{repo_id}/git/refs",
                headers={"Authorization": f"token {token}"},
                timeout=10,
            )
            # 500 = broken Git dir; 200 or 404 (empty repo) = functional
            return resp.status_code != 500
        except Exception:
            return True  # Network error — assume functional, let upload fail with retries

    def _delete_nds_repo(self, repo_id: str, token: str) -> None:
        """Delete a stale NDS repo so it can be recreated with a fresh Git dir.

        Args:
            repo_id: Repository ID (e.g., 'default/central-bank-speeches').
            token: NDS API token.
        """
        import logging

        logger = logging.getLogger(__name__)
        resp = requests.delete(
            f"{self.nds_endpoint}/api/v1/repos/{repo_id}",
            headers={"Authorization": f"token {token}"},
            timeout=30,
        )
        if resp.status_code in (200, 204):
            logger.info("Deleted stale NDS repo %s", repo_id)
        else:
            logger.error(
                "Failed to delete stale NDS repo %s: %d %s",
                repo_id,
                resp.status_code,
                resp.text[:200],
            )

    def _upload_to_nds(self, data: list[dict[str, Any]], run_id: str) -> str:
        """Upload data to NDS (NeMo Data Store) for Safe Synthesizer.

        Uses Git LFS protocol for uploading parquet files, as required by NDS.

        Args:
            data: Input data records to upload.
            run_id: Unique run identifier for the file name.

        Returns:
            HuggingFace-style URL for Safe Synthesizer (hf://datasets/repo/file.parquet).
        """
        import base64
        import hashlib
        import io

        import pandas as pd

        # Ensure HF dataset exists
        repo_id = self._ensure_hf_dataset_exists()

        # Convert data to parquet bytes
        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        parquet_bytes = buffer.getvalue()

        # Calculate SHA256 hash and size for LFS
        file_hash = hashlib.sha256(parquet_bytes).hexdigest()
        file_size = len(parquet_bytes)
        filename = f"input_{run_id}.parquet"

        # Step 1: LFS batch request to get upload URL
        lfs_batch_url = f"{self.nds_endpoint}/{repo_id}.git/info/lfs/objects/batch"
        lfs_payload = {
            "operation": "upload",
            "transfers": ["basic"],
            "objects": [{"oid": file_hash, "size": file_size}],
        }

        batch_response = requests.post(
            lfs_batch_url,
            json=lfs_payload,
            headers={
                "Accept": "application/vnd.git-lfs+json",
                "Content-Type": "application/vnd.git-lfs+json",
                "Authorization": f"token {self._get_nds_token()}",
            },
            timeout=30,
        )
        self._raise_for_nds_status(batch_response)
        batch_result = batch_response.json()

        # Step 2: Upload file via LFS (if upload action is provided)
        obj_result = batch_result["objects"][0]
        actions = obj_result.get("actions", {})

        if "upload" in actions:
            upload_info = actions["upload"]
            upload_headers = {"Content-Type": "application/octet-stream"}
            upload_headers.update(upload_info.get("header", {}))

            upload_response = requests.put(
                upload_info["href"],
                data=parquet_bytes,
                headers=upload_headers,
                timeout=120,
            )
            self._raise_for_nds_status(upload_response)

            # Step 3: Verify upload if endpoint provided
            verify_info = actions.get("verify")
            if verify_info:
                verify_payload = {"oid": file_hash, "size": file_size}
                verify_headers = {"Content-Type": "application/vnd.git-lfs+json"}
                verify_headers.update(verify_info.get("header", {}))
                requests.post(
                    verify_info["href"],
                    json=verify_payload,
                    headers=verify_headers,
                    timeout=30,
                )

        # Step 4: Create Git commit with LFS pointer
        lfs_pointer = f"""version https://git-lfs.github.com/spec/v1
oid sha256:{file_hash}
size {file_size}
"""
        # Check if file exists (for update)
        contents_url = f"{self.nds_endpoint}/api/v1/repos/{repo_id}/contents/{filename}"
        existing_sha = None
        try:
            get_response = requests.get(
                contents_url,
                headers={"Authorization": f"token {self._get_nds_token()}"},
                timeout=10,
            )
            if get_response.status_code == 200:
                existing_sha = get_response.json().get("sha")
        except Exception:
            pass

        # Commit the LFS pointer file
        commit_payload: dict[str, Any] = {
            "message": f"Add input data for run {run_id}",
            "content": base64.b64encode(lfs_pointer.encode()).decode("utf-8"),
        }
        if existing_sha:
            commit_payload["sha"] = existing_sha

        commit_response = requests.request(
            "PUT" if existing_sha else "POST",
            contents_url,
            json=commit_payload,
            headers={"Authorization": f"token {self._get_nds_token()}"},
            timeout=30,
        )
        self._raise_for_nds_status(commit_response)

        # Return HuggingFace-style URL as expected by Safe Synthesizer
        return f"hf://datasets/{repo_id}/{filename}"

    def _get_sdk_client(self) -> Any:
        """Get NeMo Microservices SDK client.

        Returns:
            NeMoMicroservices client configured for our Safe Synthesizer endpoint.
        """
        from nemo_microservices import NeMoMicroservices

        return NeMoMicroservices(base_url=self.service_endpoint)

    def _synthesize_via_api(
        self,
        data: list[dict[str, Any]],
        config: SafeSynthConfig | None = None,
        run_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], SafeSynthEvaluationResult]:
        """Synthesize data via NeMo SDK with Gitea-compatible NDS upload.

        Uploads data to NDS via Gitea API (our NDS is Gitea-based, not HF Hub),
        then uses the SDK's low-level jobs.create() and SafeSynthesizerJob for
        polling and result download.

        Args:
            data: Input data records.
            config: Optional synthesis configuration.
            run_id: Optional run identifier for tracking.

        Returns:
            Tuple of (synthetic_data, evaluation_report).
        """
        import tempfile
        import uuid

        from nemo_microservices.beta.safe_synthesizer.config.parameters import (
            SafeSynthesizerParameters,
        )
        from nemo_microservices.beta.safe_synthesizer.sdk.job_builder import (
            SafeSynthesizerJob,
            SafeSynthesizerJobConfig,
        )

        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        # Step 1: Upload data to NDS via Gitea API (SDK assumes HF Hub, ours is Gitea)
        data_source_url = self._upload_to_nds(data, run_id)

        # Step 2: Build job config using SDK models
        nss_params = SafeSynthesizerParameters(
            enable_replace_pii=True,
            enable_synthesis=True,
            generate={"num_records": len(data)},
        )
        job_config = SafeSynthesizerJobConfig(
            data_source=data_source_url,
            config=nss_params,
        )

        # Step 3: Create job via SDK low-level API
        client = self._get_sdk_client()
        response = client.beta.safe_synthesizer.jobs.create(
            spec=job_config.model_dump(),
            name=f"dagster-synth-{run_id}",
        )
        job = SafeSynthesizerJob(response.id, client)

        # Step 4: Wait for completion using SDK polling
        job.wait_for_completion(
            poll_interval=self.poll_interval,
            verbose=False,
        )

        status = job.fetch_status()
        if status != "completed":
            raise SafeSynthJobFailedError(
                job.job_id, f"Job finished with status: {status}"
            )

        # Step 5: Fetch results via SDK (returns DataFrame directly)
        synth_df = job.fetch_data()
        synthetic_data: list[dict[str, Any]] = synth_df.to_dict("records")

        # Fetch evaluation summary
        summary = job.fetch_summary()

        # Save HTML report to temp file and read bytes
        # Note: save_report() writes to disk, so we must use delete=False
        # and read the file after save_report() completes.
        html_report_bytes: bytes | None = None
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                tmp_path = tmp.name
            job.save_report(tmp_path)
            with open(tmp_path, "rb") as f:
                html_report_bytes = f.read()
        except Exception:
            pass
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        evaluation: SafeSynthEvaluationResult = {
            "job_id": job.job_id,
            "mia_score": getattr(summary, "membership_inference_protection_score", 0.0) or 0.0,
            "aia_score": getattr(summary, "attribute_inference_protection_score", 0.0) or 0.0,
            "privacy_passed": (getattr(summary, "data_privacy_score", 0) or 0) > 0.7,
            "quality_score": getattr(summary, "synthetic_data_quality_score", 0.0) or 0.0,
            "html_report_bytes": html_report_bytes,
        }

        return synthetic_data, evaluation

    def health_check(self) -> bool:
        """Check if Safe Synthesizer service is healthy."""
        try:
            response = requests.get(f"{self.service_endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _generate_mock_synthetic_data(
        self,
        input_data: list[dict[str, Any]],
        run_id: str,
    ) -> tuple[list[dict[str, Any]], SafeSynthEvaluationResult]:
        """Generate mock synthetic data for local development/testing.

        This method creates a privacy-preserving mock by:
        1. Shuffling categorical fields across records
        2. Perturbing numeric fields by ±1 (clamped to valid ranges)
        3. Flipping binary fields with 20% probability
        4. Generating new SYNTH-* reference IDs

        Args:
            input_data: Input data records to mock-synthesize.
            run_id: Unique run identifier.

        Returns:
            Tuple of (synthetic_data, mock_evaluation_result).
        """
        import random
        import uuid

        if not input_data:
            return [], {
                "job_id": f"mock-{run_id}",
                "mia_score": 0.0,
                "aia_score": 0.0,
                "privacy_passed": True,
                "quality_score": 0.0,
                "html_report_bytes": None,
            }

        # Get field names from first record
        sample = input_data[0]
        categorical_fields = []
        numeric_fields = []
        binary_fields = []

        for key, value in sample.items():
            if key in ("reference_id", "source_reference_id"):
                continue  # Skip ID fields - we'll generate new ones
            if isinstance(value, bool):
                binary_fields.append(key)
            elif isinstance(value, (int, float)):
                numeric_fields.append(key)
            elif isinstance(value, str):
                categorical_fields.append(key)

        # Create lists of values for shuffling categorical fields
        categorical_values: dict[str, list[Any]] = {}
        for field in categorical_fields:
            categorical_values[field] = [record.get(field) for record in input_data]
            random.shuffle(categorical_values[field])

        # Generate synthetic records
        synthetic_data: list[dict[str, Any]] = []
        for i, record in enumerate(input_data):
            synth_record: dict[str, Any] = {}

            # Generate new synthetic reference ID
            synth_record["reference_id"] = f"SYNTH-{uuid.uuid4().hex[:8].upper()}"
            if "source_reference_id" in record:
                synth_record["source_reference_id"] = record.get("reference_id", "")

            # Shuffle categorical fields
            for field in categorical_fields:
                synth_record[field] = categorical_values[field][i]

            # Perturb numeric fields
            for field in numeric_fields:
                original_value = record.get(field)
                if original_value is not None:
                    # Add small random perturbation
                    perturbation = random.choice([-1, 0, 1])
                    new_value = original_value + perturbation
                    # Clamp to reasonable ranges
                    if field in ("monetary_stance", "sentiment_score"):
                        new_value = max(1, min(5, new_value))
                    elif field.endswith("_score"):
                        new_value = max(0.0, min(1.0, new_value))
                    synth_record[field] = new_value
                else:
                    synth_record[field] = original_value

            # Flip binary fields with 20% probability
            for field in binary_fields:
                original_value = record.get(field)
                if random.random() < 0.2:
                    synth_record[field] = not original_value
                else:
                    synth_record[field] = original_value

            # Copy any remaining fields that weren't processed
            for key, value in record.items():
                if key not in synth_record:
                    synth_record[key] = value

            # Mark as mock-synthesized
            synth_record["_mock_synthesized"] = True

            synthetic_data.append(synth_record)

        # Return mock evaluation results
        evaluation: SafeSynthEvaluationResult = {
            "job_id": f"mock-{run_id}",
            "mia_score": 0.85,  # Mock: High membership inference protection
            "aia_score": 0.82,  # Mock: High attribute inference protection
            "privacy_passed": True,
            "quality_score": 0.78,  # Mock: Good quality score
            "html_report_bytes": None,  # No HTML report for mock
        }

        return synthetic_data, evaluation
