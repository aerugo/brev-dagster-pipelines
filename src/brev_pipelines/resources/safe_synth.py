"""NVIDIA Safe Synthesizer resource for Dagster.

Manages Safe Synthesizer deployment with automatic GPU time-sharing.
Uses KAI Scheduler priority-based preemption for GPU orchestration.

Uses NVIDIA's official NeMo Microservices Helm Chart (nemo-safe-synthesizer).

How it works:
1. Dagster scales nemo-safe-synthesizer deployment from 0 to 1 replica
2. KAI Scheduler preempts nim-llm (priority 125) for safe-synth (priority 130)
3. Safe Synthesizer runs the synthesis job
4. Dagster scales nemo-safe-synthesizer back to 0 replicas
5. nim-llm automatically restarts
6. No manual intervention required!
"""

import json
import time
from typing import Any

import requests
from dagster import ConfigurableResource
from pydantic import Field


class SafeSynthesizerResource(ConfigurableResource):
    """NVIDIA Safe Synthesizer resource using Kubernetes Jobs.

    Supports three modes:
    1. API mode: Calls Safe Synthesizer service API (requires NDS data upload)
    2. Job mode: Creates Kubernetes Jobs for on-demand synthesis
    3. Mock mode: Generates mock synthetic data for testing (no GPU required)

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
        mock_mode: Use mock synthesis for testing (no GPU/API required).
    """

    namespace: str = Field(
        default="nvidia-ai",
        description="Kubernetes namespace for Safe Synthesizer",
    )
    deployment_name: str = Field(
        default="nvidia-safe-synth-safe-synthesizer",
        description="Kubernetes deployment name for Safe Synthesizer",
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
        default="80Gi",
        description="GPU memory allocation for KAI Scheduler",
    )
    priority_class: str = Field(
        default="batch-high",
        description="Kubernetes priority class for preemption",
    )
    mock_mode: bool = Field(
        default=True,
        description="Use mock synthesis for testing (no GPU/API required)",
    )

    def _get_k8s_batch_client(self) -> Any:
        """Get Kubernetes batch API client."""
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.BatchV1Api()

    def _get_k8s_core_client(self) -> Any:
        """Get Kubernetes core API client."""
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.CoreV1Api()

    def _get_k8s_apps_client(self) -> Any:
        """Get Kubernetes apps API client."""
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.AppsV1Api()

    def _scale_deployment(self, replicas: int) -> None:
        """Scale the nemo-safe-synthesizer deployment.

        Args:
            replicas: Target replica count (0 or 1).
        """
        apps_api = self._get_k8s_apps_client()
        apps_api.patch_namespaced_deployment_scale(
            name=self.deployment_name,
            namespace=self.namespace,
            body={"spec": {"replicas": replicas}},
        )

    def _wait_for_ready(self, timeout: int = 600) -> bool:
        """Wait for nemo-safe-synthesizer deployment to be ready.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            True if ready, raises TimeoutError otherwise.
        """
        apps_api = self._get_k8s_apps_client()
        start_time = time.time()

        while time.time() - start_time < timeout:
            deployment = apps_api.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
            )
            if (
                deployment.status.ready_replicas
                and deployment.status.ready_replicas >= 1
            ):
                # Also verify the service is responding
                try:
                    response = requests.get(
                        f"{self.service_endpoint}/health", timeout=10
                    )
                    if response.status_code == 200:
                        return True
                except Exception:
                    pass
            time.sleep(10)

        raise TimeoutError(
            f"{self.deployment_name} deployment not ready after {timeout} seconds"
        )

    def create_synthesis_job(
        self,
        job_name: str,
        input_data_path: str,
        output_data_path: str,
        synth_config: dict[str, Any] | None = None,
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
        config_defaults = {
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
                        image_pull_secrets=[
                            client.V1LocalObjectReference(name="ngc-image-pull")
                        ],
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
                                    client.V1EnvVar(
                                        name="INPUT_PATH", value=input_data_path
                                    ),
                                    client.V1EnvVar(
                                        name="OUTPUT_PATH", value=output_data_path
                                    ),
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

    def wait_for_job(self, job_name: str) -> dict[str, Any]:
        """Wait for a job to complete.

        Args:
            job_name: Job name to wait for.

        Returns:
            Job status information.

        Raises:
            TimeoutError: If job doesn't complete in max_wait_time.
            RuntimeError: If job fails.
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
                    return {
                        "state": "completed",
                        "succeeded": job.status.succeeded,
                        "completion_time": (
                            job.status.completion_time.isoformat()
                            if job.status.completion_time
                            else None
                        ),
                    }

                if job.status.failed and job.status.failed > 0:
                    # Get pod logs for error details
                    error_msg = self._get_job_logs(job_name)
                    raise RuntimeError(f"Job {job_name} failed: {error_msg}")

            except ApiException as e:
                if e.status == 404:
                    raise RuntimeError(f"Job {job_name} not found")
                raise

            time.sleep(self.poll_interval)

        raise TimeoutError(
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
        config: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run full synthesis pipeline with automatic GPU orchestration.

        This method automatically:
        1. Scales up the safe-synth deployment (0 -> 1 replica)
        2. Waits for the service to be ready (KAI preempts nim-llm)
        3. Runs the synthesis via API
        4. Scales down the deployment (1 -> 0 replica)
        5. nim-llm automatically restarts

        No manual intervention required!

        If mock_mode is True, generates mock synthetic data without GPU/API.

        Args:
            input_data: Input data records.
            run_id: Unique run identifier.
            config: Optional synthesis configuration.

        Returns:
            Tuple of (synthetic_data, evaluation_report).
        """
        # Use mock mode for testing without GPU/API
        if self.mock_mode:
            return self._synthesize_mock(input_data, config)

        # Check if already running
        already_running = self.health_check()

        if not already_running:
            # Scale up deployment - KAI will preempt nim-llm
            self._scale_deployment(replicas=1)
            self._wait_for_ready(timeout=600)  # 10 min for model loading

        try:
            return self._synthesize_via_api(input_data, config, run_id)
        finally:
            if not already_running:
                # Scale down - nim-llm will auto-restart
                self._scale_deployment(replicas=0)

    def _upload_to_nds(self, data: list[dict[str, Any]], run_id: str) -> str:
        """Upload data to NDS (NeMo Data Store) for Safe Synthesizer.

        Uploads data as a parquet file using Gitea's REST API.

        Args:
            data: Input data records to upload.
            run_id: Unique run identifier for the file name.

        Returns:
            The repo path for the uploaded data (e.g., admin/central-bank-speeches/input_xxx.parquet).
        """
        import base64
        import io

        import pandas as pd

        # Convert data to parquet bytes
        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        parquet_bytes = buffer.getvalue()

        # Upload file via Gitea API
        filename = f"input_{run_id}.parquet"
        owner, repo = self.nds_repo.split("/")

        # First, try to get existing file (for update)
        get_url = f"{self.nds_endpoint}/api/v1/repos/{owner}/{repo}/contents/{filename}"
        sha = None
        try:
            get_response = requests.get(
                get_url,
                headers={"Authorization": f"token {self.nds_token}"},
                timeout=30,
            )
            if get_response.status_code == 200:
                sha = get_response.json().get("sha")
        except Exception:
            pass

        # Upload/update file via Gitea contents API
        upload_url = f"{self.nds_endpoint}/api/v1/repos/{owner}/{repo}/contents/{filename}"
        payload: dict[str, Any] = {
            "message": f"Add input data for run {run_id}",
            "content": base64.b64encode(parquet_bytes).decode("utf-8"),
        }
        if sha:
            payload["sha"] = sha

        response = requests.post(
            upload_url,
            json=payload,
            headers={"Authorization": f"token {self.nds_token}"},
            timeout=120,
        )
        response.raise_for_status()

        return f"{self.nds_repo}/{filename}"

    def _synthesize_via_api(
        self,
        data: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Synthesize data via the Safe Synthesizer API.

        This method:
        1. Uploads data to NDS (HuggingFace-compatible data store)
        2. Creates a Safe Synthesizer job via /v1beta1/safe-synthesizer/jobs
        3. Polls for completion
        4. Downloads and returns synthetic data

        Args:
            data: Input data records.
            config: Optional synthesis configuration.
            run_id: Optional run identifier for tracking.

        Returns:
            Tuple of (synthetic_data, evaluation_report).
        """
        import uuid

        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        # Upload data to NDS
        data_source = self._upload_to_nds(data, run_id)

        # Build Safe Synthesizer job config
        synth_config: dict[str, Any] = {
            "enable_synthesis": True,
            "enable_replace_pii": config.get("piiReplacement", True) if config else True,
            "data": {
                "holdout": 0.05,
                "max_holdout": min(len(data) // 10, 2000),
            },
            "evaluation": {
                "mia_enabled": config.get("runMiaEvaluation", True) if config else True,
                "aia_enabled": config.get("runAiaEvaluation", True) if config else True,
                "enabled": True,
            },
            "training": {
                "pretrained_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "num_input_records_to_sample": "auto",
            },
            "generation": {
                "num_records": len(data),
                "temperature": config.get("temperature", 0.9) if config else 0.9,
            },
        }

        # Add differential privacy if epsilon is specified
        if config and config.get("epsilon"):
            synth_config["privacy"] = {
                "dp_enabled": True,
                "epsilon": config["epsilon"],
                "delta": config.get("delta", "auto"),
            }

        # Create job payload
        payload = {
            "name": f"dagster-synth-{run_id}",
            "spec": {
                "data_source": data_source,
                "config": synth_config,
            },
        }

        # Create job via Safe Synthesizer API v1beta1
        response = requests.post(
            f"{self.service_endpoint}/v1beta1/safe-synthesizer/jobs",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        job_response = response.json()
        job_id = job_response.get("id")

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < self.max_wait_time:
            status_response = requests.get(
                f"{self.service_endpoint}/v1beta1/safe-synthesizer/jobs/{job_id}/status",
                timeout=60,
            )
            status_response.raise_for_status()
            status = status_response.json()

            job_status = status.get("status")
            if job_status == "completed":
                # Download synthetic data
                synth_response = requests.get(
                    f"{self.service_endpoint}/v1beta1/safe-synthesizer/jobs/{job_id}/results/synthetic_data/download",
                    timeout=120,
                )
                synth_response.raise_for_status()

                # Parse parquet response
                import io

                import pandas as pd

                synth_df = pd.read_parquet(io.BytesIO(synth_response.content))
                synthetic_data = synth_df.to_dict("records")

                # Get evaluation summary
                summary_response = requests.get(
                    f"{self.service_endpoint}/v1beta1/safe-synthesizer/jobs/{job_id}/results/summary/download",
                    timeout=60,
                )
                summary = {}
                if summary_response.status_code == 200:
                    summary = summary_response.json()

                evaluation = {
                    "job_id": job_id,
                    "mia_score": summary.get("membership_inference_protection_score"),
                    "aia_score": summary.get("attribute_inference_protection_score"),
                    "privacy_passed": summary.get("data_privacy_score", 0) > 0.7,
                    "quality_score": summary.get("synthetic_data_quality_score"),
                }

                return synthetic_data, evaluation

            elif job_status in ("error", "cancelled"):
                error_details = status.get("error_details", {})
                raise RuntimeError(
                    f"Job {job_id} failed with status {job_status}: {error_details}"
                )

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Job {job_id} did not complete in {self.max_wait_time} seconds"
        )

    def _synthesize_mock(
        self,
        data: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate mock synthetic data for testing.

        Creates synthetic copies of input records with minor text modifications
        to simulate privacy-preserving synthesis. Useful for testing the pipeline
        without requiring GPU or Safe Synthesizer API access.

        Args:
            data: Input data records.
            config: Optional configuration (ignored in mock mode).

        Returns:
            Tuple of (synthetic_data, mock_evaluation_report).
        """
        import hashlib
        import random

        synthetic_data: list[dict[str, Any]] = []

        for i, record in enumerate(data):
            # Create a synthetic copy with modified text
            synthetic_record = {}

            for key, value in record.items():
                if key == "text" and isinstance(value, str):
                    # Add synthetic prefix and slight text modification
                    words = value.split()
                    # Shuffle some words to simulate synthesis
                    if len(words) > 10:
                        random.seed(i)  # Reproducible for testing
                        mid = len(words) // 2
                        shuffled = words[:5] + words[mid : mid + 5] + words[5:mid] + words[mid + 5 :]
                        synthetic_record[key] = " ".join(shuffled)
                    else:
                        synthetic_record[key] = f"[SYNTHETIC] {value}"
                elif key == "title" and isinstance(value, str):
                    synthetic_record[key] = f"{value} (Synthetic)"
                elif key == "speaker" and isinstance(value, str):
                    # Replace speaker name with pseudonym
                    name_hash = hashlib.md5(value.encode()).hexdigest()[:6]
                    synthetic_record[key] = f"Speaker_{name_hash.upper()}"
                else:
                    synthetic_record[key] = value

            synthetic_data.append(synthetic_record)

        # Mock evaluation report with good scores
        evaluation = {
            "job_id": f"mock-{random.randint(1000, 9999)}",
            "mia_score": 0.95,  # High = good privacy (low attack success)
            "aia_score": 0.92,  # High = good privacy
            "privacy_passed": True,
            "mock_mode": True,
            "records_processed": len(data),
        }

        return synthetic_data, evaluation

    def health_check(self) -> bool:
        """Check if Safe Synthesizer service is healthy."""
        try:
            response = requests.get(f"{self.service_endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
