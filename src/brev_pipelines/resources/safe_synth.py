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

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast

import requests
from dagster import ConfigurableResource
from pydantic import Field

from brev_pipelines.types import SafeSynthConfig, SafeSynthEvaluationResult, SafeSynthJobStatus

if TYPE_CHECKING:
    from datetime import datetime

    from brev_pipelines.types import K8sAppsV1Api, K8sBatchV1Api, K8sCoreV1Api


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

    def _get_k8s_apps_client(self) -> K8sAppsV1Api:
        """Get Kubernetes apps API client."""
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.AppsV1Api()  # type: ignore[no-any-return]

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
            if deployment.status.ready_replicas and deployment.status.ready_replicas >= 1:
                # Also verify the service is responding
                try:
                    response = requests.get(f"{self.service_endpoint}/health", timeout=10)
                    if response.status_code == 200:
                        return True
                except Exception:
                    pass
            time.sleep(10)

        raise TimeoutError(f"{self.deployment_name} deployment not ready after {timeout} seconds")

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
                    raise RuntimeError(f"Job {job_name} failed: {error_msg}")

            except ApiException as e:
                if e.status == 404:
                    raise RuntimeError(f"Job {job_name} not found") from e
                raise

            time.sleep(self.poll_interval)

        raise TimeoutError(f"Job {job_name} did not complete in {self.max_wait_time} seconds")

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
        """Run full synthesis pipeline with automatic GPU orchestration.

        This method automatically:
        1. Scales up the safe-synth deployment (0 -> 1 replica)
        2. Waits for the service to be ready (KAI preempts nim-llm)
        3. Runs the synthesis via API
        4. Scales down the deployment (1 -> 0 replica)
        5. nim-llm automatically restarts

        No manual intervention required!

        Args:
            input_data: Input data records.
            run_id: Unique run identifier.
            config: Optional synthesis configuration.

        Returns:
            Tuple of (synthetic_data, evaluation_report).
        """
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

    def _ensure_hf_dataset_exists(self) -> str:
        """Ensure HuggingFace-compatible dataset exists in NDS.

        Creates the dataset via HF API if it doesn't exist.

        Returns:
            The dataset repo ID (e.g., 'default/speeches-data').
        """
        # Extract repo name from nds_repo (e.g., 'admin/central-bank-speeches' -> 'speeches-data')
        repo_name = self.nds_repo.split("/")[-1]

        # Try to create dataset via HF API
        create_url = f"{self.nds_endpoint}/v1/hf/api/repos/create"
        create_payload = {"type": "dataset", "name": repo_name, "private": False}

        try:
            response = requests.post(
                create_url,
                json=create_payload,
                headers={"Authorization": f"Bearer {self.nds_token}"},
                timeout=30,
            )
            if response.status_code == 200:
                result: dict[str, str] = response.json()
                # Returns {'url': 'datasets/default/repo-name'}
                url: str = result.get("url", f"default/{repo_name}")
                return url.replace("datasets/", "")
            elif response.status_code == 409:
                # Already exists
                return f"default/{repo_name}"
            else:
                response.raise_for_status()
        except requests.exceptions.HTTPError:
            # Assume it exists if we get an error
            pass

        return f"default/{repo_name}"

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
                "Authorization": f"token {self.nds_token}",
            },
            timeout=30,
        )
        batch_response.raise_for_status()
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
            upload_response.raise_for_status()

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
                headers={"Authorization": f"token {self.nds_token}"},
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
            headers={"Authorization": f"token {self.nds_token}"},
            timeout=30,
        )
        commit_response.raise_for_status()

        # Return HuggingFace-style URL as expected by Safe Synthesizer
        return f"hf://datasets/{repo_id}/{filename}"

    def _synthesize_via_api(
        self,
        data: list[dict[str, Any]],
        config: SafeSynthConfig | None = None,
        run_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], SafeSynthEvaluationResult]:
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

        # Build Safe Synthesizer job config with memory-optimized settings
        # Lower max_vram_fraction leaves room for evaluation models (sentence_transformers)
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
                # Reduce evaluation rows to prevent CUDA OOM
                "sqs_report_rows": 1000,
            },
            "training": {
                "pretrained_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                # RoPE scaling extends context from 2K to ~12K tokens (factor 6 = max)
                # Allows longer text fields (~10K chars) at cost of increased runtime
                "rope_scaling_factor": 6,
                # Use more training samples to prevent underfitting
                "num_input_records_to_sample": min(len(data), 5000),
                # Memory optimization: leave 40% VRAM for evaluation
                "max_vram_fraction": 0.6,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
            },
            "generation": {
                "num_records": len(data),
                # Lower temperature for more deterministic (valid JSON) output
                "temperature": config.get("temperature", 0.5) if config else 0.5,
                # Increase patience to allow more retry attempts
                "patience": 5,
                "invalid_fraction_threshold": 0.9,
                # Enable structured generation to force valid JSON output
                "use_structured_generation": True,
                "structured_generation_backend": "auto",
            },
        }

        # Add differential privacy if epsilon is specified
        epsilon = config.get("epsilon") if config else None
        if epsilon is not None:
            synth_config["privacy"] = {
                "dp_enabled": True,
                "epsilon": epsilon,
                "delta": config.get("delta", "auto") if config else "auto",
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
                synthetic_data: list[dict[str, Any]] = synth_df.to_dict("records")

                # Get evaluation summary
                summary_response = requests.get(
                    f"{self.service_endpoint}/v1beta1/safe-synthesizer/jobs/{job_id}/results/summary/download",
                    timeout=60,
                )
                summary = {}
                if summary_response.status_code == 200:
                    summary = summary_response.json()

                # Get HTML evaluation report (contains SQS/DPS graphs and metrics)
                html_report_bytes: bytes | None = None
                try:
                    report_response = requests.get(
                        f"{self.service_endpoint}/v1beta1/safe-synthesizer/jobs/{job_id}/results/report/download",
                        timeout=60,
                    )
                    if report_response.status_code == 200:
                        html_report_bytes = report_response.content
                except Exception:
                    # HTML report is optional - don't fail if unavailable
                    pass

                evaluation: SafeSynthEvaluationResult = {
                    "job_id": job_id,
                    "mia_score": summary.get("membership_inference_protection_score") or 0.0,
                    "aia_score": summary.get("attribute_inference_protection_score") or 0.0,
                    "privacy_passed": (summary.get("data_privacy_score") or 0) > 0.7,
                    "quality_score": summary.get("synthetic_data_quality_score") or 0.0,
                    "html_report_bytes": html_report_bytes,
                }

                return synthetic_data, evaluation

            elif job_status in ("error", "cancelled"):
                error_details = status.get("error_details", {})
                raise RuntimeError(f"Job {job_id} failed with status {job_status}: {error_details}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete in {self.max_wait_time} seconds")

    def health_check(self) -> bool:
        """Check if Safe Synthesizer service is healthy."""
        try:
            response = requests.get(f"{self.service_endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
