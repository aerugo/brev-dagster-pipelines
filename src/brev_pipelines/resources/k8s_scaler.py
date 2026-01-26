"""Kubernetes deployment scaler resource for GPU workload management.

Provides scaling operations for NIM deployments to manage GPU resource contention.
Used to scale down large models (nim-reasoning) when running embedding jobs,
then scale back up after completion.

Usage:
    @asset
    def my_asset(k8s_scaler: K8sScalerResource):
        with k8s_scaler.temporarily_scale("nim-reasoning", "nvidia-ai", replicas=0):
            # Run GPU-intensive work while nim-reasoning is scaled down
            ...
        # nim-reasoning automatically scaled back up

Follows invariants:
- INV-P004: Complete type annotations
- INV-P006: Modern Python 3.11+ syntax
- INV-P007: Pydantic v2 for configuration
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

from dagster import ConfigurableResource, get_dagster_logger
from kubernetes import client, config
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Generator


class K8sScalerResource(ConfigurableResource):
    """Kubernetes deployment scaler for managing GPU resources.

    Provides methods to scale deployments up/down and wait for readiness.
    Includes context manager for temporary scaling with automatic restoration.

    Attributes:
        in_cluster: Whether running inside a Kubernetes cluster.
        scale_timeout: Timeout in seconds for scale operations.
        ready_timeout: Timeout in seconds waiting for pods to be ready.
    """

    in_cluster: bool = Field(
        default=True,
        description="Whether to use in-cluster config (True) or local kubeconfig (False)",
    )
    scale_timeout: int = Field(
        default=30,
        description="Timeout in seconds for scale API calls",
    )
    ready_timeout: int = Field(
        default=600,
        description="Timeout in seconds waiting for pods to become ready",
    )

    def _get_apps_api(self) -> client.AppsV1Api:
        """Get Kubernetes AppsV1 API client."""
        if self.in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config()
        return client.AppsV1Api()

    def get_replicas(self, deployment: str, namespace: str) -> int:
        """Get current replica count for a deployment.

        Args:
            deployment: Deployment name.
            namespace: Kubernetes namespace.

        Returns:
            Current replica count.

        Raises:
            ApiException: If the deployment doesn't exist or API call fails.
        """
        api = self._get_apps_api()
        dep = api.read_namespaced_deployment(deployment, namespace)
        return dep.spec.replicas or 0

    def scale(
        self,
        deployment: str,
        namespace: str,
        replicas: int,
        wait_ready: bool = True,
    ) -> int:
        """Scale a deployment to the specified replica count.

        Args:
            deployment: Deployment name.
            namespace: Kubernetes namespace.
            replicas: Target replica count.
            wait_ready: Whether to wait for pods to be ready (for scale up).

        Returns:
            Previous replica count.

        Raises:
            ApiException: If the scale operation fails.
            TimeoutError: If wait_ready is True and pods don't become ready.
        """
        log = get_dagster_logger()
        api = self._get_apps_api()

        # Get current state
        dep = api.read_namespaced_deployment(deployment, namespace)
        previous_replicas = dep.spec.replicas or 0

        if previous_replicas == replicas:
            log.info(f"{namespace}/{deployment} already at {replicas} replicas")
            return previous_replicas

        # Scale the deployment
        log.info(f"Scaling {namespace}/{deployment}: {previous_replicas} -> {replicas}")
        body = {"spec": {"replicas": replicas}}
        api.patch_namespaced_deployment(deployment, namespace, body)

        # Wait for scale to complete
        if wait_ready and replicas > 0:
            self._wait_for_ready(deployment, namespace, replicas)
        elif replicas == 0:
            self._wait_for_scale_down(deployment, namespace)

        log.info(f"Scaled {namespace}/{deployment} to {replicas} replicas")
        return previous_replicas

    def _wait_for_ready(self, deployment: str, namespace: str, replicas: int) -> None:
        """Wait for deployment pods to be ready.

        Args:
            deployment: Deployment name.
            namespace: Kubernetes namespace.
            replicas: Expected replica count.

        Raises:
            TimeoutError: If pods don't become ready within timeout.
        """
        log = get_dagster_logger()
        api = self._get_apps_api()
        start = time.time()

        while time.time() - start < self.ready_timeout:
            dep = api.read_namespaced_deployment(deployment, namespace)
            ready = dep.status.ready_replicas or 0
            if ready >= replicas:
                return
            log.debug(f"Waiting for {namespace}/{deployment}: {ready}/{replicas} ready")
            time.sleep(10)

        raise TimeoutError(
            f"{namespace}/{deployment} did not reach {replicas} ready replicas "
            f"within {self.ready_timeout}s"
        )

    def _wait_for_scale_down(self, deployment: str, namespace: str) -> None:
        """Wait for deployment to scale down to 0.

        Args:
            deployment: Deployment name.
            namespace: Kubernetes namespace.
        """
        log = get_dagster_logger()
        api = self._get_apps_api()
        start = time.time()

        while time.time() - start < self.scale_timeout:
            dep = api.read_namespaced_deployment(deployment, namespace)
            available = dep.status.available_replicas or 0
            if available == 0:
                return
            log.debug(f"Waiting for {namespace}/{deployment} to scale down: {available} remaining")
            time.sleep(2)

        # Don't fail on scale down timeout, just log warning
        log.warning(f"{namespace}/{deployment} scale down may not be complete")

    @contextlib.contextmanager
    def temporarily_scale(
        self,
        deployment: str,
        namespace: str,
        replicas: int = 0,
        restore_wait_ready: bool = False,
    ) -> Generator[None, None, None]:
        """Context manager to temporarily scale a deployment.

        Scales the deployment to the target replicas, yields control,
        then restores the original replica count on exit (even on exception).

        Args:
            deployment: Deployment name.
            namespace: Kubernetes namespace.
            replicas: Target replica count during the context.
            restore_wait_ready: Whether to wait for ready when restoring.

        Yields:
            None

        Example:
            with k8s_scaler.temporarily_scale("nim-reasoning", "nvidia-ai", 0):
                # nim-reasoning is scaled to 0
                run_embedding_job()
            # nim-reasoning is restored to original replicas
        """
        log = get_dagster_logger()
        original_replicas: int | None = None

        try:
            original_replicas = self.scale(
                deployment, namespace, replicas, wait_ready=False
            )
            yield
        finally:
            if original_replicas is not None and original_replicas != replicas:
                log.info(f"Restoring {namespace}/{deployment} to {original_replicas} replicas")
                try:
                    self.scale(
                        deployment,
                        namespace,
                        original_replicas,
                        wait_ready=restore_wait_ready,
                    )
                except Exception as e:
                    log.error(f"Failed to restore {namespace}/{deployment}: {e}")
                    # Don't raise - we don't want to mask the original exception
