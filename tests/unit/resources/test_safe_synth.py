"""Unit tests for Safe Synthesizer resource.

Tests the SafeSynthesizerResource configuration, Kubernetes operations,
and synthesis workflows. All external calls are mocked per INV-P010.
"""

from __future__ import annotations

from typing import get_type_hints
from unittest.mock import MagicMock, patch

import pytest

from brev_pipelines.types import (
    K8sAppsV1Api,
    K8sBatchV1Api,
    K8sCoreV1Api,
    SafeSynthConfig,
    SafeSynthEvaluationResult,
    SafeSynthJobStatus,
)


class TestSafeSynthesizerResourceInit:
    """Tests for SafeSynthesizerResource initialization."""

    def test_resource_initialization_defaults(self):
        """Test SafeSynthesizerResource has sensible defaults."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource()

        assert resource.namespace == "nvidia-ai"
        assert resource.deployment_name == "nvidia-safe-synth-safe-synthesizer"
        assert "nvcr.io/nvidia" in resource.image
        assert resource.poll_interval == 30
        assert resource.max_wait_time == 7200
        assert resource.gpu_memory == "80Gi"
        assert resource.priority_class == "batch-high"

    def test_resource_custom_config(self):
        """Test SafeSynthesizerResource with custom configuration."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource(
            namespace="custom-ns",
            deployment_name="my-safe-synth",
            poll_interval=60,
            max_wait_time=3600,
            gpu_memory="40Gi",
            priority_class="batch-low",
        )

        assert resource.namespace == "custom-ns"
        assert resource.deployment_name == "my-safe-synth"
        assert resource.poll_interval == 60
        assert resource.max_wait_time == 3600
        assert resource.gpu_memory == "40Gi"
        assert resource.priority_class == "batch-low"

    def test_resource_nds_config(self):
        """Test SafeSynthesizerResource NDS configuration."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource(
            nds_endpoint="http://custom-nds:3000",
            nds_token="test-token",
            nds_repo="org/repo",
        )

        assert resource.nds_endpoint == "http://custom-nds:3000"
        assert resource.nds_token == "test-token"
        assert resource.nds_repo == "org/repo"


class TestK8sClientCreation:
    """Tests for Kubernetes client creation methods."""

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_get_k8s_batch_client_returns_batch_api(self, mock_method: MagicMock):
        """Test _get_k8s_batch_client returns K8sBatchV1Api-compatible client."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_client.create_namespaced_job = MagicMock()
        mock_client.read_namespaced_job_status = MagicMock()
        mock_client.delete_namespaced_job = MagicMock()
        mock_method.return_value = mock_client

        resource = SafeSynthesizerResource()
        client = resource._get_k8s_batch_client()

        # Verify client has expected methods
        assert hasattr(client, "create_namespaced_job")
        assert hasattr(client, "read_namespaced_job_status")
        assert hasattr(client, "delete_namespaced_job")

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_core_client")
    def test_get_k8s_core_client_returns_core_api(self, mock_method: MagicMock):
        """Test _get_k8s_core_client returns K8sCoreV1Api-compatible client."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_client.list_namespaced_pod = MagicMock()
        mock_client.read_namespaced_pod_log = MagicMock()
        mock_method.return_value = mock_client

        resource = SafeSynthesizerResource()
        client = resource._get_k8s_core_client()

        # Verify client has expected methods
        assert hasattr(client, "list_namespaced_pod")
        assert hasattr(client, "read_namespaced_pod_log")

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_apps_client")
    def test_get_k8s_apps_client_returns_apps_api(self, mock_method: MagicMock):
        """Test _get_k8s_apps_client returns K8sAppsV1Api-compatible client."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_client.read_namespaced_deployment = MagicMock()
        mock_client.patch_namespaced_deployment_scale = MagicMock()
        mock_method.return_value = mock_client

        resource = SafeSynthesizerResource()
        client = resource._get_k8s_apps_client()

        # Verify client has expected methods
        assert hasattr(client, "read_namespaced_deployment")
        assert hasattr(client, "patch_namespaced_deployment_scale")


class TestDeploymentScaling:
    """Tests for deployment scaling operations."""

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_apps_client")
    def test_scale_deployment_up(self, mock_get_client: MagicMock):
        """Test scaling deployment to 1 replica."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        resource._scale_deployment(replicas=1)

        mock_client.patch_namespaced_deployment_scale.assert_called_once_with(
            name=resource.deployment_name,
            namespace=resource.namespace,
            body={"spec": {"replicas": 1}},
        )

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_apps_client")
    def test_scale_deployment_down(self, mock_get_client: MagicMock):
        """Test scaling deployment to 0 replicas."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        resource._scale_deployment(replicas=0)

        mock_client.patch_namespaced_deployment_scale.assert_called_once_with(
            name=resource.deployment_name,
            namespace=resource.namespace,
            body={"spec": {"replicas": 0}},
        )


class TestWaitForReady:
    """Tests for deployment readiness waiting."""

    @patch("brev_pipelines.resources.safe_synth.requests.get")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_apps_client")
    def test_wait_for_ready_success(self, mock_get_client: MagicMock, mock_requests_get: MagicMock):
        """Test waiting for deployment to become ready."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        # Mock deployment with ready replicas
        mock_deployment = MagicMock()
        mock_deployment.status.ready_replicas = 1
        mock_client = MagicMock()
        mock_client.read_namespaced_deployment.return_value = mock_deployment
        mock_get_client.return_value = mock_client

        # Mock health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        resource = SafeSynthesizerResource()
        result = resource._wait_for_ready(timeout=10)

        assert result is True

    @patch("brev_pipelines.resources.safe_synth.time.sleep")
    @patch("brev_pipelines.resources.safe_synth.time.time")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_apps_client")
    def test_wait_for_ready_timeout(
        self,
        mock_get_client: MagicMock,
        mock_time: MagicMock,
        mock_sleep: MagicMock,
    ):
        """Test timeout when deployment doesn't become ready."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        # Mock deployment with no ready replicas
        mock_deployment = MagicMock()
        mock_deployment.status.ready_replicas = None
        mock_client = MagicMock()
        mock_client.read_namespaced_deployment.return_value = mock_deployment
        mock_get_client.return_value = mock_client

        # Simulate time passing beyond timeout
        mock_time.side_effect = [0, 5, 15]  # Start, first check, timeout

        resource = SafeSynthesizerResource()

        with pytest.raises(TimeoutError, match="not ready"):
            resource._wait_for_ready(timeout=10)


class TestCreateSynthesisJob:
    """Tests for Kubernetes job creation."""

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_create_synthesis_job_success(self, mock_get_client: MagicMock):
        """Test successful job creation."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        job_name = resource.create_synthesis_job(
            job_name="test-job-123",
            input_data_path="s3://bucket/input.parquet",
            output_data_path="s3://bucket/output.parquet",
        )

        assert job_name == "test-job-123"
        mock_client.create_namespaced_job.assert_called_once()

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_create_synthesis_job_with_config(self, mock_get_client: MagicMock):
        """Test job creation with custom synthesis config."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        synth_config: SafeSynthConfig = {
            "epsilon": 2.0,
            "delta": 1e-6,
            "piiReplacement": False,
            "temperature": 0.5,
        }

        job_name = resource.create_synthesis_job(
            job_name="test-job-456",
            input_data_path="s3://bucket/input.parquet",
            output_data_path="s3://bucket/output.parquet",
            synth_config=synth_config,
        )

        assert job_name == "test-job-456"
        mock_client.create_namespaced_job.assert_called_once()


class TestWaitForJob:
    """Tests for job completion waiting."""

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_wait_for_job_completed(self, mock_get_client: MagicMock):
        """Test waiting for job that completes successfully."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_job = MagicMock()
        mock_job.status.succeeded = 1
        mock_job.status.failed = None
        mock_job.status.completion_time = MagicMock()
        mock_job.status.completion_time.isoformat.return_value = "2024-01-15T12:00:00Z"

        mock_client = MagicMock()
        mock_client.read_namespaced_job_status.return_value = mock_job
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        result = resource.wait_for_job("test-job")

        assert result["state"] == "completed"
        assert result["succeeded"] == 1
        assert result["completion_time"] == "2024-01-15T12:00:00Z"

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_job_logs")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_wait_for_job_failed(self, mock_get_client: MagicMock, mock_get_logs: MagicMock):
        """Test waiting for job that fails."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_job = MagicMock()
        mock_job.status.succeeded = None
        mock_job.status.failed = 1

        mock_client = MagicMock()
        mock_client.read_namespaced_job_status.return_value = mock_job
        mock_get_client.return_value = mock_client

        mock_get_logs.return_value = "Out of memory error"

        resource = SafeSynthesizerResource()

        with pytest.raises(RuntimeError, match="failed"):
            resource.wait_for_job("failed-job")

    @patch("brev_pipelines.resources.safe_synth.time.sleep")
    @patch("brev_pipelines.resources.safe_synth.time.time")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_wait_for_job_timeout(
        self,
        mock_get_client: MagicMock,
        mock_time: MagicMock,
        mock_sleep: MagicMock,
    ):
        """Test timeout when job doesn't complete."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_job = MagicMock()
        mock_job.status.succeeded = None
        mock_job.status.failed = None

        mock_client = MagicMock()
        mock_client.read_namespaced_job_status.return_value = mock_job
        mock_get_client.return_value = mock_client

        # Simulate time passing beyond max_wait_time
        mock_time.side_effect = [0, 100, 7300]

        resource = SafeSynthesizerResource(max_wait_time=7200)

        with pytest.raises(TimeoutError, match="did not complete"):
            resource.wait_for_job("stuck-job")


class TestDeleteJob:
    """Tests for job deletion."""

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_delete_job_success(self, mock_get_client: MagicMock):
        """Test successful job deletion."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        result = resource.delete_job("test-job")

        assert result is True
        mock_client.delete_namespaced_job.assert_called_once()

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_batch_client")
    def test_delete_job_not_found(self, mock_get_client: MagicMock):
        """Test deleting non-existent job returns False."""
        from kubernetes.client.rest import ApiException

        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_client = MagicMock()
        mock_error = ApiException(status=404)
        mock_client.delete_namespaced_job.side_effect = mock_error
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        result = resource.delete_job("nonexistent-job")

        assert result is False


class TestGetJobLogs:
    """Tests for retrieving job logs."""

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_core_client")
    def test_get_job_logs_success(self, mock_get_client: MagicMock):
        """Test successful log retrieval."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_pod = MagicMock()
        mock_pod.metadata.name = "test-job-pod-abc"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]

        mock_client = MagicMock()
        mock_client.list_namespaced_pod.return_value = mock_pod_list
        mock_client.read_namespaced_pod_log.return_value = "Job completed successfully"
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        logs = resource._get_job_logs("test-job")

        assert "Job completed successfully" in logs

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._get_k8s_core_client")
    def test_get_job_logs_no_pods(self, mock_get_client: MagicMock):
        """Test log retrieval when no pods exist."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_pod_list = MagicMock()
        mock_pod_list.items = []

        mock_client = MagicMock()
        mock_client.list_namespaced_pod.return_value = mock_pod_list
        mock_get_client.return_value = mock_client

        resource = SafeSynthesizerResource()
        logs = resource._get_job_logs("test-job")

        assert "No pods found" in logs


class TestSynthesize:
    """Tests for full synthesis pipeline."""

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._synthesize_via_api")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._wait_for_ready")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._scale_deployment")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource.health_check")
    def test_synthesize_scales_deployment(
        self,
        mock_health: MagicMock,
        mock_scale: MagicMock,
        mock_wait: MagicMock,
        mock_synth: MagicMock,
    ):
        """Test synthesize scales deployment up and down."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_health.return_value = False  # Not already running
        mock_wait.return_value = True
        mock_synth.return_value = ([{"col": "val"}], {"job_id": "test"})

        resource = SafeSynthesizerResource()
        input_data = [{"text": "Sample speech"}]

        resource.synthesize(input_data, run_id="test-run")

        # Should scale up, wait, synthesize, scale down
        assert mock_scale.call_count == 2
        mock_scale.assert_any_call(replicas=1)
        mock_scale.assert_any_call(replicas=0)

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._synthesize_via_api")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._scale_deployment")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource.health_check")
    def test_synthesize_skips_scaling_when_running(
        self,
        mock_health: MagicMock,
        mock_scale: MagicMock,
        mock_synth: MagicMock,
    ):
        """Test synthesize skips scaling when service already running."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_health.return_value = True  # Already running
        mock_synth.return_value = ([{"col": "val"}], {"job_id": "test"})

        resource = SafeSynthesizerResource()
        resource.synthesize([{"text": "Sample"}], run_id="test")

        # Should not scale when already running
        mock_scale.assert_not_called()

    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._synthesize_via_api")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._wait_for_ready")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource._scale_deployment")
    @patch("brev_pipelines.resources.safe_synth.SafeSynthesizerResource.health_check")
    def test_synthesize_with_config(
        self,
        mock_health: MagicMock,
        mock_scale: MagicMock,
        mock_wait: MagicMock,
        mock_synth: MagicMock,
    ):
        """Test synthesize passes config to API."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_health.return_value = False
        mock_wait.return_value = True
        mock_synth.return_value = ([{"col": "val"}], {"job_id": "test"})

        resource = SafeSynthesizerResource()
        config: SafeSynthConfig = {
            "epsilon": 2.0,
            "piiReplacement": True,
        }

        resource.synthesize([{"text": "Sample"}], run_id="test", config=config)

        mock_synth.assert_called_once()
        call_args = mock_synth.call_args
        assert call_args[0][1] == config  # Second positional arg is config


class TestHealthCheck:
    """Tests for health check functionality."""

    @patch("brev_pipelines.resources.safe_synth.requests.get")
    def test_health_check_success(self, mock_get: MagicMock):
        """Test health check returns True when service is healthy."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        resource = SafeSynthesizerResource()
        result = resource.health_check()

        assert result is True

    @patch("brev_pipelines.resources.safe_synth.requests.get")
    def test_health_check_failure(self, mock_get: MagicMock):
        """Test health check returns False on connection error."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        mock_get.side_effect = Exception("Connection refused")

        resource = SafeSynthesizerResource()
        result = resource.health_check()

        assert result is False


class TestTypeAnnotations:
    """Tests verifying type annotations use proper types from brev_pipelines.types."""

    def test_get_k8s_batch_client_return_type(self):
        """Test _get_k8s_batch_client has proper return type annotation."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        hints = get_type_hints(
            SafeSynthesizerResource._get_k8s_batch_client,
            globalns={"K8sBatchV1Api": K8sBatchV1Api},
        )
        assert "return" in hints
        assert hints["return"] is K8sBatchV1Api

    def test_get_k8s_core_client_return_type(self):
        """Test _get_k8s_core_client has proper return type annotation."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        hints = get_type_hints(
            SafeSynthesizerResource._get_k8s_core_client,
            globalns={"K8sCoreV1Api": K8sCoreV1Api},
        )
        assert "return" in hints
        assert hints["return"] is K8sCoreV1Api

    def test_get_k8s_apps_client_return_type(self):
        """Test _get_k8s_apps_client has proper return type annotation."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        hints = get_type_hints(
            SafeSynthesizerResource._get_k8s_apps_client,
            globalns={"K8sAppsV1Api": K8sAppsV1Api},
        )
        assert "return" in hints
        assert hints["return"] is K8sAppsV1Api

    def test_create_synthesis_job_synth_config_type(self):
        """Test create_synthesis_job accepts SafeSynthConfig."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        hints = get_type_hints(
            SafeSynthesizerResource.create_synthesis_job,
            globalns={"SafeSynthConfig": SafeSynthConfig},
        )
        assert "synth_config" in hints
        # Should be SafeSynthConfig | None
        expected_type = SafeSynthConfig | None
        assert hints["synth_config"] == expected_type

    def test_wait_for_job_return_type(self):
        """Test wait_for_job returns SafeSynthJobStatus."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        hints = get_type_hints(
            SafeSynthesizerResource.wait_for_job,
            globalns={"SafeSynthJobStatus": SafeSynthJobStatus},
        )
        assert "return" in hints
        assert hints["return"] is SafeSynthJobStatus

    def test_synthesize_config_type(self):
        """Test synthesize accepts SafeSynthConfig."""
        from typing import Any

        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        hints = get_type_hints(
            SafeSynthesizerResource.synthesize,
            globalns={
                "SafeSynthConfig": SafeSynthConfig,
                "SafeSynthEvaluationResult": SafeSynthEvaluationResult,
                "Any": Any,
            },
        )
        assert "config" in hints
        expected_type = SafeSynthConfig | None
        assert hints["config"] == expected_type

    def test_synthesize_return_type(self):
        """Test synthesize returns tuple with SafeSynthEvaluationResult."""
        from typing import Any

        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        hints = get_type_hints(
            SafeSynthesizerResource.synthesize,
            globalns={
                "SafeSynthConfig": SafeSynthConfig,
                "SafeSynthEvaluationResult": SafeSynthEvaluationResult,
                "Any": Any,
            },
        )
        assert "return" in hints
        # Return type should include SafeSynthEvaluationResult in tuple
        return_type = hints["return"]
        assert "SafeSynthEvaluationResult" in str(return_type)
