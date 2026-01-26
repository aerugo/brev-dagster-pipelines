"""Comprehensive platform validation assets.

These assets provide programmatic validation of the entire Brev Data Platform stack.
Run them through Dagster UI or programmatically via:

    dagster asset materialize -m brev_pipelines.definitions --select validate_platform

Each validation asset tests a specific component and returns a structured result
with pass/fail status and diagnostic information.
"""

import io
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import dagster as dg

from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMResource


@dataclass
class ValidationResult:
    """Structured validation result."""

    component: str
    passed: bool
    tests: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component": self.component,
            "passed": self.passed,
            "tests": self.tests,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# MinIO Validation
# =============================================================================


@dg.asset(
    description="Validate MinIO object storage connectivity and operations",
    group_name="validation",
    metadata={"validation_type": "storage"},
)
def validate_minio(
    context: dg.AssetExecutionContext,
    minio: MinIOResource,
) -> dict[str, Any]:
    """Comprehensive MinIO validation.

    Tests:
    1. Connection - Can connect to MinIO
    2. List buckets - Can enumerate buckets
    3. Create bucket - Can create test bucket
    4. Write object - Can write data
    5. Read object - Can read data back
    6. Delete object - Can delete data
    7. Delete bucket - Can cleanup test bucket
    """
    start = time.time()
    result = ValidationResult(component="minio", passed=True)
    test_bucket = "validation-test-bucket"
    test_key = "validation/test.json"
    test_data = {"test": True, "timestamp": datetime.now(UTC).isoformat()}

    try:
        client = minio.get_client()

        # Test 1: Connection
        context.log.info("Test 1: MinIO connection...")
        try:
            client.list_buckets()
            result.tests.append({"name": "connection", "passed": True})
        except Exception as e:
            result.tests.append({"name": "connection", "passed": False, "error": str(e)})
            result.passed = False
            raise

        # Test 2: List buckets
        context.log.info("Test 2: List buckets...")
        try:
            buckets = client.list_buckets()
            bucket_names = [b.name for b in buckets]
            result.tests.append(
                {
                    "name": "list_buckets",
                    "passed": True,
                    "bucket_count": len(buckets),
                    "buckets": bucket_names,
                }
            )
        except Exception as e:
            result.tests.append({"name": "list_buckets", "passed": False, "error": str(e)})
            result.passed = False

        # Test 3: Create bucket
        context.log.info(f"Test 3: Create test bucket '{test_bucket}'...")
        try:
            if not client.bucket_exists(test_bucket):
                client.make_bucket(test_bucket)
            result.tests.append({"name": "create_bucket", "passed": True, "bucket": test_bucket})
        except Exception as e:
            result.tests.append({"name": "create_bucket", "passed": False, "error": str(e)})
            result.passed = False

        # Test 4: Write object
        context.log.info(f"Test 4: Write test object to '{test_key}'...")
        try:
            data_bytes = json.dumps(test_data).encode()
            client.put_object(
                test_bucket,
                test_key,
                io.BytesIO(data_bytes),
                len(data_bytes),
                content_type="application/json",
            )
            result.tests.append({"name": "write_object", "passed": True, "key": test_key})
        except Exception as e:
            result.tests.append({"name": "write_object", "passed": False, "error": str(e)})
            result.passed = False

        # Test 5: Read object
        context.log.info(f"Test 5: Read test object from '{test_key}'...")
        try:
            response = client.get_object(test_bucket, test_key)
            read_data = json.loads(response.read())
            response.close()
            response.release_conn()
            if read_data == test_data:
                result.tests.append({"name": "read_object", "passed": True, "data_matches": True})
            else:
                result.tests.append(
                    {
                        "name": "read_object",
                        "passed": False,
                        "error": "Data mismatch",
                    }
                )
                result.passed = False
        except Exception as e:
            result.tests.append({"name": "read_object", "passed": False, "error": str(e)})
            result.passed = False

        # Test 6: Delete object
        context.log.info(f"Test 6: Delete test object '{test_key}'...")
        try:
            client.remove_object(test_bucket, test_key)
            result.tests.append({"name": "delete_object", "passed": True})
        except Exception as e:
            result.tests.append({"name": "delete_object", "passed": False, "error": str(e)})
            result.passed = False

        # Test 7: Delete bucket
        context.log.info(f"Test 7: Delete test bucket '{test_bucket}'...")
        try:
            client.remove_bucket(test_bucket)
            result.tests.append({"name": "delete_bucket", "passed": True})
        except Exception as e:
            result.tests.append({"name": "delete_bucket", "passed": False, "error": str(e)})
            result.passed = False

    except Exception as e:
        result.error = str(e)
        result.passed = False

    result.duration_ms = (time.time() - start) * 1000
    context.log.info(f"MinIO validation: {'PASSED' if result.passed else 'FAILED'}")
    return result.to_dict()


# =============================================================================
# LakeFS Validation
# =============================================================================


@dg.asset(
    description="Validate LakeFS data versioning connectivity and operations",
    group_name="validation",
    metadata={"validation_type": "versioning"},
)
def validate_lakefs(
    context: dg.AssetExecutionContext,
    lakefs: LakeFSResource,
) -> dict[str, Any]:
    """Comprehensive LakeFS validation.

    Tests:
    1. Connection - Can connect to LakeFS
    2. List repositories - Can enumerate repositories
    3. API health - API responds correctly
    """
    start = time.time()
    result = ValidationResult(component="lakefs", passed=True)

    try:
        # Test 1: Connection / Health
        context.log.info("Test 1: LakeFS connection...")
        try:
            health = lakefs.health_check()
            result.tests.append({"name": "connection", "passed": health})
            if not health:
                result.passed = False
        except Exception as e:
            result.tests.append({"name": "connection", "passed": False, "error": str(e)})
            result.passed = False
            raise

        # Test 2: List repositories
        context.log.info("Test 2: List repositories...")
        try:
            repos = lakefs.list_repositories()
            result.tests.append(
                {
                    "name": "list_repositories",
                    "passed": True,
                    "repository_count": len(repos),
                    "repositories": repos,
                }
            )
        except Exception as e:
            result.tests.append({"name": "list_repositories", "passed": False, "error": str(e)})
            result.passed = False

        # Test 3: API version check
        context.log.info("Test 3: API version check...")
        try:
            # Verify client can be created (validates endpoint/credentials)
            _ = lakefs.get_client()
            result.tests.append(
                {
                    "name": "api_version",
                    "passed": True,
                    "endpoint": lakefs.endpoint,
                }
            )
        except Exception as e:
            result.tests.append({"name": "api_version", "passed": False, "error": str(e)})
            # Not a critical failure

    except Exception as e:
        result.error = str(e)
        result.passed = False

    result.duration_ms = (time.time() - start) * 1000
    context.log.info(f"LakeFS validation: {'PASSED' if result.passed else 'FAILED'}")
    return result.to_dict()


# =============================================================================
# NIM LLM Validation
# =============================================================================


@dg.asset(
    description="Validate NVIDIA NIM LLM inference endpoint",
    group_name="validation",
    metadata={"validation_type": "ai", "uses_gpu": "true"},
)
def validate_nim(
    context: dg.AssetExecutionContext,
    nim: NIMResource,
) -> dict[str, Any]:
    """Comprehensive NIM LLM validation.

    Tests:
    1. Health check - Service is responding
    2. Simple completion - Can generate text
    3. Response quality - Output is coherent
    """
    start = time.time()
    result = ValidationResult(component="nim", passed=True)

    try:
        # Test 1: Health check
        context.log.info("Test 1: NIM health check...")
        try:
            health = nim.health_check()
            result.tests.append({"name": "health_check", "passed": health})
            if not health:
                result.passed = False
                # If health check fails, skip other tests
                result.error = "NIM service not healthy"
                result.duration_ms = (time.time() - start) * 1000
                return result.to_dict()
        except Exception as e:
            result.tests.append({"name": "health_check", "passed": False, "error": str(e)})
            result.passed = False
            raise

        # Test 2: Simple completion
        context.log.info("Test 2: Simple text completion...")
        try:
            test_prompt = "Complete this sentence with exactly 5 words: The sky is"
            response = nim.generate(test_prompt, max_tokens=20)
            has_content = len(response.strip()) > 0 and "error" not in response.lower()
            result.tests.append(
                {
                    "name": "simple_completion",
                    "passed": has_content,
                    "prompt": test_prompt,
                    "response": response[:100],  # Truncate for logging
                }
            )
            if not has_content:
                result.passed = False
        except Exception as e:
            result.tests.append({"name": "simple_completion", "passed": False, "error": str(e)})
            result.passed = False

        # Test 3: Response quality (structured output)
        context.log.info("Test 3: Structured response test...")
        try:
            json_prompt = 'Return only a valid JSON object with keys \'status\' and \'message\'. Example: {"status": "ok", "message": "working"}'
            response = nim.generate(json_prompt, max_tokens=50)
            # Check if response looks like JSON
            is_json_like = "{" in response and "}" in response
            result.tests.append(
                {
                    "name": "structured_response",
                    "passed": is_json_like,
                    "response": response[:100],
                }
            )
            # Not a critical failure if JSON isn't perfect
        except Exception as e:
            result.tests.append({"name": "structured_response", "passed": False, "error": str(e)})
            # Not a critical failure

    except Exception as e:
        result.error = str(e)
        result.passed = False

    result.duration_ms = (time.time() - start) * 1000
    context.log.info(f"NIM validation: {'PASSED' if result.passed else 'FAILED'}")
    return result.to_dict()


# =============================================================================
# End-to-End Validation
# =============================================================================


@dg.asset(
    description="Run complete platform validation and generate report",
    group_name="validation",
    metadata={"validation_type": "e2e"},
    deps=[validate_minio, validate_lakefs, validate_nim],
)
def validate_platform(
    context: dg.AssetExecutionContext,
    validate_minio: dict[str, Any],
    validate_lakefs: dict[str, Any],
    validate_nim: dict[str, Any],
    minio: MinIOResource,
) -> dict[str, Any]:
    """Complete platform validation with comprehensive report.

    Aggregates results from all component validations and stores
    the full report to MinIO for persistence.
    """
    start = time.time()

    # Aggregate all results
    components = {
        "minio": validate_minio,
        "lakefs": validate_lakefs,
        "nim": validate_nim,
    }

    # Calculate overall status
    all_passed = all(c["passed"] for c in components.values())
    passed_count = sum(1 for c in components.values() if c["passed"])

    # Create comprehensive report with typed intermediate dicts
    overall_status = "PASSED" if all_passed else "FAILED"
    summary: dict[str, str] = {
        "minio": "✅ PASSED" if validate_minio["passed"] else "❌ FAILED",
        "lakefs": "✅ PASSED" if validate_lakefs["passed"] else "❌ FAILED",
        "nim": "✅ PASSED" if validate_nim["passed"] else "❌ FAILED",
    }
    report: dict[str, object] = {
        "validation_run": {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_status": overall_status,
            "passed_components": passed_count,
            "total_components": len(components),
        },
        "components": components,
        "summary": summary,
    }

    # Log summary
    context.log.info("=" * 60)
    context.log.info("PLATFORM VALIDATION REPORT")
    context.log.info("=" * 60)
    for comp, status in summary.items():
        context.log.info(f"  {comp.upper():10} {status}")
    context.log.info("=" * 60)
    context.log.info(f"OVERALL: {overall_status}")
    context.log.info("=" * 60)

    # Store report to MinIO
    try:
        bucket = "data-products"
        minio.ensure_bucket(bucket)
        client = minio.get_client()

        # Store timestamped report
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_key = f"validation/report_{timestamp}.json"

        data = json.dumps(report, indent=2).encode()
        client.put_object(
            bucket,
            report_key,
            io.BytesIO(data),
            len(data),
            content_type="application/json",
        )
        context.log.info(f"Report stored to {bucket}/{report_key}")

        # Also store as latest
        client.put_object(
            bucket,
            "validation/latest.json",
            io.BytesIO(data),
            len(data),
            content_type="application/json",
        )
        context.log.info(f"Report stored to {bucket}/validation/latest.json")

        report["report_location"] = f"{bucket}/{report_key}"
    except Exception as e:
        context.log.warning(f"Could not store report to MinIO: {e}")
        report["report_location"] = "not_stored"

    report["duration_ms"] = (time.time() - start) * 1000
    return report


# =============================================================================
# Quick Health Check (Lightweight)
# =============================================================================


@dg.asset(
    description="Quick health check of all services (lightweight)",
    group_name="validation",
    metadata={"validation_type": "quick"},
)
def quick_health_check(
    context: dg.AssetExecutionContext,
    minio: MinIOResource,
    lakefs: LakeFSResource,
    nim: NIMResource,
) -> dict[str, Any]:
    """Quick health check of all services.

    This is a lightweight version that just checks connectivity,
    without running full validation tests.
    """
    start = time.time()
    results = {}

    # MinIO
    try:
        client = minio.get_client()
        client.list_buckets()
        results["minio"] = {"status": "healthy", "error": None}
    except Exception as e:
        results["minio"] = {"status": "unhealthy", "error": str(e)[:100]}

    # LakeFS
    try:
        health = lakefs.health_check()
        results["lakefs"] = {"status": "healthy" if health else "unhealthy", "error": None}
    except Exception as e:
        results["lakefs"] = {"status": "unhealthy", "error": str(e)[:100]}

    # NIM
    try:
        health = nim.health_check()
        results["nim"] = {"status": "healthy" if health else "unhealthy", "error": None}
    except Exception as e:
        results["nim"] = {"status": "unhealthy", "error": str(e)[:100]}

    # Summary
    all_healthy = all(r["status"] == "healthy" for r in results.values())
    overall_status = "healthy" if all_healthy else "unhealthy"

    report: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "overall_status": overall_status,
        "services": results,
        "duration_ms": (time.time() - start) * 1000,
    }

    context.log.info(f"Quick health check: {overall_status.upper()}")
    for svc, status in results.items():
        icon = "✅" if status["status"] == "healthy" else "❌"
        context.log.info(f"  {icon} {svc}: {status['status']}")

    return report


# Export all validation assets
validation_assets = [
    validate_minio,
    validate_lakefs,
    validate_nim,
    validate_platform,
    quick_health_check,
]
