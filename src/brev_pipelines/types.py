"""Type definitions for Brev Pipelines.

Provides TypedDict definitions and Protocol types to replace dict[str, Any]
throughout the codebase, ensuring strict typing per INV-P005.

Usage:
    from brev_pipelines.types import WeaviatePropertyDef, SafeSynthConfig

All types follow:
- INV-P005: No Any types - use TypedDict for structured dicts
- INV-P006: Modern Python 3.11+ typing syntax
- INV-P011: No bare generics
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

# =============================================================================
# Weaviate Types
# =============================================================================


class _WeaviatePropertyDefRequired(TypedDict):
    """Required fields for WeaviatePropertyDef."""

    name: str


class WeaviatePropertyDef(_WeaviatePropertyDefRequired, total=False):
    """Schema definition for a Weaviate collection property.

    Used by WeaviateResource.ensure_collection() to define collection schema.

    Required:
        name: Property name (snake_case recommended).

    Optional:
        type: Data type ('text', 'int', 'boolean', 'number', 'date').
              Defaults to 'text' if not specified.
        description: Human-readable description for documentation.
    """

    type: str
    description: str


class WeaviateSearchResult(TypedDict):
    """Result from Weaviate vector similarity search.

    Contains the matched object's properties plus distance/certainty metadata.

    Attributes:
        properties: Dictionary of object properties (varies by collection).
        _distance: Cosine distance from query vector (lower = more similar).
        _certainty: Confidence score (higher = more certain match).
    """

    properties: dict[str, str | int | float | bool]
    _distance: float
    _certainty: float


class WeaviateObject(TypedDict):
    """Object to insert into Weaviate collection.

    Used by WeaviateResource.insert_objects() for batch insertion.
    """

    reference: str
    date: str
    central_bank: str
    speaker: str
    title: str
    text: str
    monetary_stance: int
    trade_stance: int
    tariff_mention: bool
    economic_outlook: int
    is_governor: bool


# =============================================================================
# Safe Synthesizer Types
# =============================================================================


class SafeSynthTrainingConfig(TypedDict, total=False):
    """Training configuration for Safe Synthesizer."""

    rope_scaling_factor: int


class SafeSynthGenerationConfig(TypedDict, total=False):
    """Generation configuration for Safe Synthesizer."""

    temperature: float
    use_structured_generation: bool


class SafeSynthDataConfig(TypedDict, total=False):
    """Data configuration for Safe Synthesizer."""

    holdout: int


class SafeSynthConfig(TypedDict, total=False):
    """Configuration for NVIDIA Safe Synthesizer synthetic data generation.

    All fields are optional - defaults are applied by the resource.

    Differential Privacy:
        epsilon: Privacy budget (lower = more private). Default: 1.0
        delta: Privacy failure probability. Default: 1e-5

    Data Processing:
        piiReplacement: Enable automatic PII replacement. Default: True

    Evaluation:
        runMiaEvaluation: Run Membership Inference Attack evaluation. Default: True
        runAiaEvaluation: Run Attribute Inference Attack evaluation. Default: True

    Nested Config:
        training: Training configuration (rope_scaling_factor, etc.).
        generation: Generation configuration (temperature, structured generation).
        data: Data configuration (holdout, etc.).
    """

    epsilon: float
    delta: float
    piiReplacement: bool
    runMiaEvaluation: bool
    runAiaEvaluation: bool
    training: SafeSynthTrainingConfig
    generation: SafeSynthGenerationConfig
    data: SafeSynthDataConfig


class SafeSynthJobStatus(TypedDict, total=False):
    """Status information from a Kubernetes synthesis job.

    Returned by SafeSynthesizerResource.wait_for_job().

    Attributes:
        state: Job state ('completed', 'failed', 'running').
        succeeded: Number of succeeded pods (typically 1 when complete).
        failed: Number of failed pods (non-zero indicates failure).
        completion_time: ISO-format timestamp of job completion.
        error: Error message if job failed.
    """

    state: str  # Required
    succeeded: int
    failed: int
    completion_time: str
    error: str


class SafeSynthEvaluationResult(TypedDict, total=False):
    """Evaluation metrics from synthetic data generation.

    Returned as part of SafeSynthesizerResource.synthesize() tuple.

    Privacy Metrics:
        mia_score: Membership Inference Attack resistance (0.0-1.0, higher = safer).
        aia_score: Attribute Inference Attack resistance (0.0-1.0, higher = safer).
        privacy_passed: Whether privacy thresholds were met.

    Quality Metrics:
        quality_score: Synthetic data quality score (0.0-1.0, higher = better).

    Statistics:
        input_records: Number of input records processed.
        output_records: Number of synthetic records generated.
        job_id: Kubernetes job identifier for tracing.

    Optional:
        html_report_bytes: Raw HTML bytes of evaluation report (for visualization).
    """

    mia_score: float
    aia_score: float
    privacy_passed: bool
    quality_score: float
    input_records: int
    output_records: int
    job_id: str
    html_report_bytes: bytes | None


# =============================================================================
# Asset Output Types - Central Bank Speeches
# =============================================================================


class DataProductMetadata(TypedDict, total=False):
    """Metadata returned from speeches_data_product asset.

    Contains information about the stored Parquet file in LakeFS.

    Attributes:
        path: LakeFS URI (e.g., 'lakefs://data/main/central-bank-speeches/speeches.parquet').
        commit_id: LakeFS commit ID (None if no changes committed).
        num_records: Total number of speech records.
        tariff_mentions: Count of speeches mentioning tariffs.
    """

    path: str  # Required
    commit_id: str
    num_records: int
    tariff_mentions: int


class WeaviateIndexMetadata(TypedDict):
    """Metadata returned from weaviate_index asset.

    Contains information about the Weaviate vector index.

    Attributes:
        collection: Weaviate collection name.
        object_count: Number of indexed objects.
        vector_dimensions: Dimensionality of embedding vectors.
    """

    collection: str
    object_count: int
    vector_dimensions: int


class SnapshotMetadata(TypedDict, total=False):
    """Metadata returned from intermediate snapshot assets.

    Used by: classification_snapshot, summaries_snapshot, embeddings_snapshot.

    Attributes:
        path: LakeFS URI for the snapshot file.
        commit_id: LakeFS commit ID (None if no changes).
        num_records: Number of records in snapshot.
    """

    path: str  # Required
    commit_id: str
    num_records: int


# =============================================================================
# Validation Types
# =============================================================================


class ValidationTestResult(TypedDict, total=False):
    """Result of a single validation test.

    Used within ValidationResult.tests list.

    Attributes:
        name: Test name identifier.
        passed: Whether test passed.
        error: Error message if test failed.
        Additional fields may be present based on test type.
    """

    name: str  # Required
    passed: bool  # Required
    error: str
    # Additional fields allowed for test-specific metadata
    bucket_count: int
    buckets: list[str]
    repositories: list[str]
    model: str
    response: str


class ValidationReportDict(TypedDict, total=False):
    """Dictionary representation of full validation report.

    Returned by validate_platform and individual validation assets.

    Attributes:
        component: Component being validated (minio, lakefs, nim, etc.).
        passed: Overall validation result.
        tests: List of individual test results.
        error: Overall error message if validation failed.
        duration_ms: Validation duration in milliseconds.
    """

    component: str  # Required
    passed: bool  # Required
    tests: list[ValidationTestResult]
    error: str
    duration_ms: float


# =============================================================================
# Kubernetes Protocol Types
# =============================================================================


@runtime_checkable
class K8sV1ObjectMeta(Protocol):
    """Protocol for Kubernetes V1ObjectMeta."""

    name: str
    namespace: str | None


@runtime_checkable
class K8sV1JobStatus(Protocol):
    """Protocol for Kubernetes V1JobStatus."""

    succeeded: int | None
    failed: int | None
    completion_time: object | None  # datetime or None


@runtime_checkable
class K8sV1Job(Protocol):
    """Protocol for Kubernetes V1Job."""

    metadata: K8sV1ObjectMeta
    status: K8sV1JobStatus


@runtime_checkable
class K8sV1Pod(Protocol):
    """Protocol for Kubernetes V1Pod."""

    metadata: K8sV1ObjectMeta


@runtime_checkable
class K8sV1PodList(Protocol):
    """Protocol for Kubernetes V1PodList."""

    items: list[K8sV1Pod]


@runtime_checkable
class K8sV1DeploymentSpec(Protocol):
    """Protocol for Kubernetes V1DeploymentSpec."""

    replicas: int


@runtime_checkable
class K8sV1DeploymentStatus(Protocol):
    """Protocol for Kubernetes V1DeploymentStatus."""

    ready_replicas: int | None


@runtime_checkable
class K8sV1Deployment(Protocol):
    """Protocol for Kubernetes V1Deployment."""

    spec: K8sV1DeploymentSpec
    status: K8sV1DeploymentStatus


@runtime_checkable
class K8sBatchV1Api(Protocol):
    """Protocol for Kubernetes BatchV1Api client.

    Used by SafeSynthesizerResource for job management.
    """

    def create_namespaced_job(
        self,
        namespace: str,
        body: object,
    ) -> K8sV1Job:
        """Create a namespaced job."""
        ...

    def read_namespaced_job_status(
        self,
        name: str,
        namespace: str,
    ) -> K8sV1Job:
        """Read job status."""
        ...

    def delete_namespaced_job(
        self,
        name: str,
        namespace: str,
        *,
        propagation_policy: str = "Background",
    ) -> object:
        """Delete a job."""
        ...


@runtime_checkable
class K8sCoreV1Api(Protocol):
    """Protocol for Kubernetes CoreV1Api client.

    Used by SafeSynthesizerResource for pod operations.
    """

    def list_namespaced_pod(
        self,
        namespace: str,
        *,
        label_selector: str | None = None,
    ) -> K8sV1PodList:
        """List pods in namespace."""
        ...

    def read_namespaced_pod_log(
        self,
        name: str,
        namespace: str,
        *,
        container: str | None = None,
        tail_lines: int | None = None,
    ) -> str:
        """Read pod logs."""
        ...


@runtime_checkable
class K8sAppsV1Api(Protocol):
    """Protocol for Kubernetes AppsV1Api client.

    Used by SafeSynthesizerResource for deployment scaling.
    """

    def read_namespaced_deployment(
        self,
        name: str,
        namespace: str,
    ) -> K8sV1Deployment:
        """Read deployment."""
        ...

    def patch_namespaced_deployment_scale(
        self,
        name: str,
        namespace: str,
        body: dict[str, object],
    ) -> object:
        """Patch deployment scale."""
        ...


# =============================================================================
# LakeFS Types
# =============================================================================


class LakeFSCommitMetadata(TypedDict, total=False):
    """Metadata for LakeFS commit operations.

    Attributes:
        dagster_run_id: Dagster run identifier for tracing.
        num_records: Number of records in committed data.
        tariff_mentions: Count of tariff mentions (for CBS pipeline).
    """

    dagster_run_id: str
    num_records: str  # LakeFS requires string metadata values
    tariff_mentions: str


# =============================================================================
# NIM Types
# =============================================================================


class NIMChatMessage(TypedDict):
    """Message in NIM chat completion format.

    Follows OpenAI-compatible API format.
    """

    role: str  # 'system', 'user', 'assistant'
    content: str


class NIMCompletionChoice(TypedDict):
    """Choice in NIM completion response."""

    message: NIMChatMessage
    finish_reason: str


class NIMUsage(TypedDict):
    """Token usage statistics from NIM."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class NIMCompletionResponse(TypedDict):
    """Response from NIM chat completion API."""

    choices: list[NIMCompletionChoice]
    usage: NIMUsage


class NIMEmbeddingData(TypedDict):
    """Single embedding in NIM response."""

    embedding: list[float]
    index: int


class NIMEmbeddingResponse(TypedDict):
    """Response from NIM embedding API."""

    data: list[NIMEmbeddingData]
    model: str
    usage: NIMUsage


# =============================================================================
# Classification Types
# =============================================================================


class SpeechClassification(TypedDict):
    """Multi-dimensional classification result for a speech.

    Generated by NIM LLM classification prompt.
    """

    monetary_stance: int  # 1-5 scale: very_dovish to very_hawkish
    trade_stance: int  # 1-5 scale: very_protectionist to very_globalist
    tariff_mention: int  # 0 or 1: boolean as int
    economic_outlook: int  # 1-5 scale: very_negative to very_positive


# Classification scale mappings (shared between assets and validation)
MONETARY_STANCE_SCALE: dict[str, int] = {
    "very_dovish": 1,
    "somewhat_dovish": 2,
    "dovish": 2,  # Alias for somewhat_dovish
    "neutral": 3,
    "somewhat_hawkish": 4,
    "hawkish": 4,  # Alias for somewhat_hawkish
    "very_hawkish": 5,
}

TRADE_STANCE_SCALE: dict[str, int] = {
    "very_protectionist": 1,
    "somewhat_protectionist": 2,
    "protectionist": 2,  # Alias for somewhat_protectionist
    "neutral": 3,
    "somewhat_globalist": 4,
    "globalist": 4,  # Alias for somewhat_globalist
    "very_globalist": 5,
}

OUTLOOK_SCALE: dict[str, int] = {
    "very_negative": 1,
    "somewhat_negative": 2,
    "negative": 2,  # Alias for somewhat_negative
    "neutral": 3,
    "somewhat_positive": 4,
    "positive": 4,  # Alias for somewhat_positive
    "very_positive": 5,
}


# =============================================================================
# LLM Observability Types
# =============================================================================


class LLMFailureBreakdown(TypedDict):
    """Breakdown of LLM failures by error type.

    Used for tracking failure statistics in Dagster asset metadata.

    Attributes:
        ValidationError: Count of validation failures (invalid JSON, missing fields).
        LLMTimeoutError: Count of timeout failures.
        LLMRateLimitError: Count of rate limit (429) failures.
        LLMServerError: Count of server (5xx) failures.
        unexpected_error: Count of unexpected/unclassified failures.
    """

    ValidationError: int
    LLMTimeoutError: int
    LLMRateLimitError: int
    LLMServerError: int
    unexpected_error: int


class LLMAssetMetadata(TypedDict):
    """Metadata for LLM-powered assets with failure tracking.

    Used for Dagster asset metadata output via context.add_output_metadata().
    Provides observability into LLM call success rates and failure patterns.

    Attributes:
        total_processed: Total number of records processed.
        successful: Number of successful LLM calls.
        failed: Number of failed LLM calls (using fallback).
        success_rate: Percentage as formatted string (e.g., "95.0%").
        failed_references: List of record IDs that failed (limited to 100).
        failure_breakdown: Counts by error type.
        avg_attempts: Average number of attempts per record.
        total_duration_ms: Total processing time in milliseconds.
    """

    total_processed: int
    successful: int
    failed: int
    success_rate: str
    failed_references: list[str]
    failure_breakdown: LLMFailureBreakdown
    avg_attempts: float
    total_duration_ms: int


# =============================================================================
# LLM Checkpoint Types
# =============================================================================


class CheckpointRecordBase(TypedDict, total=False):
    """Base fields for all checkpoint records.

    All checkpoint records must have a reference field for identification.
    LLM status fields track processing results.
    """

    reference: str  # Required - unique record identifier
    _llm_status: str  # 'success' or 'failed'
    _llm_error: str | None  # Error message if failed
    _llm_attempts: int  # Number of retry attempts
    _llm_fallback_used: bool  # Whether fallback value was used


class ClassificationCheckpointRecord(CheckpointRecordBase):
    """Checkpoint record for classification LLM results.

    Used by speech_classifications asset to checkpoint classification progress.
    """

    monetary_stance: int  # 1-5 scale
    trade_stance: int  # 1-5 scale
    tariff_mention: int  # 0 or 1
    economic_outlook: int  # 1-5 scale


class SummaryCheckpointRecord(CheckpointRecordBase):
    """Checkpoint record for summary LLM results.

    Used by speech_summaries asset to checkpoint summary progress.
    """

    summary: str  # Generated summary text


class EmbeddingCheckpointRecord(TypedDict, total=False):
    """Checkpoint record for embedding results.

    Note: Does not extend CheckpointRecordBase since embeddings
    don't go through LLM retry wrapper.
    """

    reference: str  # Required - unique record identifier
    embedding: list[float]  # Embedding vector


# =============================================================================
# Type Aliases
# =============================================================================

# For property definitions that accept various structures
PropertyDefList = list[WeaviatePropertyDef]

# For embedding vectors
EmbeddingVector = list[float]
EmbeddingBatch = list[EmbeddingVector]

# For Weaviate objects
WeaviateObjectList = list[WeaviateObject]
