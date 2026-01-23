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
            return self._synthesize_via_api(input_data, config)
        finally:
            if not already_running:
                # Scale down - nim-llm will auto-restart
                self._scale_deployment(replicas=0)

    def _synthesize_via_api(
        self,
        data: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Synthesize data via the API endpoint."""
        default_config = {
            "epsilon": 1.0,
            "delta": 1e-5,
            "piiReplacement": True,
            "temperature": 0.7,
            "runMiaEvaluation": True,
            "runAiaEvaluation": True,
        }
        if config:
            default_config.update(config)

        payload = {"data": data, "config": default_config}

        # Create job via NeMo Core API v1
        response = requests.post(
            f"{self.service_endpoint}/v1/jobs",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        job_response = response.json()
        job_id = job_response.get("job_id") or job_response.get("id")

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < self.max_wait_time:
            status_response = requests.get(
                f"{self.service_endpoint}/v1/jobs/{job_id}",
                timeout=60,
            )
            status_response.raise_for_status()
            status = status_response.json()

            if status.get("state") == "completed" or status.get("status") == "completed":
                # Download results
                results = status.get("results", [])
                result_id = results[0].get("id") or results[0].get("name") if results else "output"
                result_response = requests.get(
                    f"{self.service_endpoint}/v1/jobs/{job_id}/results/{result_id}/download",
                    timeout=60,
                )
                result_response.raise_for_status()
                synthetic_data = result_response.json()

                evaluation = {
                    "job_id": job_id,
                    "mia_score": status.get("evaluation", {}).get("mia_score"),
                    "aia_score": status.get("evaluation", {}).get("aia_score"),
                    "privacy_passed": status.get("evaluation", {}).get(
                        "privacy_passed", False
                    ),
                }

                return synthetic_data, evaluation

            elif status.get("state") == "failed" or status.get("status") == "failed":
                raise RuntimeError(f"Job {job_id} failed: {status.get('error') or status.get('message')}")

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
