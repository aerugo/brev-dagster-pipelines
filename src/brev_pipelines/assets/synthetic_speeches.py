"""Synthetic Central Bank Speeches Pipeline - Summary-Based Synthesis.

Generates privacy-preserving synthetic twin of the speeches dataset using
NVIDIA Safe Synthesizer for metadata + classification + summary synthesis.

What we synthesize:
- Categorical metadata: date, central_bank, speaker, title
- Classification scores: monetary_stance, trade_stance, economic_outlook (1-5 scales)
- Binary flags: tariff_mention, is_governor
- Compact summary: ~1000 char bullet-point summary with specific details

The synthetic dataset contains summaries (not full text) because:
1. Compact summaries (~1000 chars) fit within Safe Synthesizer's context window
2. Summaries capture semantic nuance: specific metrics, geographic focus, sector
   commentary, forward guidance language, risk factors, policy tools
3. Full speech text (~20000+ chars) is too long for TinyLlama to reproduce

Pipeline is DECOUPLED from the ETL pipeline - loads enriched data from LakeFS.

GPU Orchestration (Automatic via KAI):
1. Safe Synthesizer runs with 'batch-high' priority (130)
2. KAI preempts NIM (priority 125) to free the GPU
3. After Safe Synth completes, NIM Deployment restarts its pod
4. No manual kubectl commands required!
"""

import io
import json
from datetime import UTC, datetime
from typing import Any

import dagster as dg
import polars as pl

from brev_pipelines.config import PipelineConfig
from brev_pipelines.io_managers.checkpoint import LLMCheckpointManager
from brev_pipelines.resources.lakefs import (
    LakeFSConnectionError,
    LakeFSError,
    LakeFSNotFoundError,
    LakeFSResource,
)
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource
from brev_pipelines.resources.safe_synth import SafeSynthesizerResource
from brev_pipelines.resources.safe_synth_retry import (
    SafeSynthRetryConfig,
    retry_safe_synth_call,
)
from brev_pipelines.resources.weaviate import (
    WeaviateCollectionError,
    WeaviateConnectionError,
    WeaviateResource,
)
from brev_pipelines.types import SafeSynthConfig, SafeSynthEvaluationResult, WeaviatePropertyDef

# Weaviate schema for synthetic speeches (summary-based, no full text)
SYNTHETIC_SCHEMA: list[WeaviatePropertyDef] = [
    WeaviatePropertyDef(
        name="reference", type="text", description="Unique identifier (SYNTH-XXXXXX)"
    ),
    WeaviatePropertyDef(name="date", type="text", description="Speech date (ISO format)"),
    WeaviatePropertyDef(name="central_bank", type="text", description="Issuing institution"),
    WeaviatePropertyDef(name="speaker", type="text", description="Speaker name"),
    WeaviatePropertyDef(name="title", type="text", description="Speech title"),
    WeaviatePropertyDef(
        name="summary", type="text", description="Compact summary (~1000 chars, synthesized)"
    ),
    WeaviatePropertyDef(
        name="monetary_stance", type="int", description="1=very_dovish to 5=very_hawkish"
    ),
    WeaviatePropertyDef(
        name="trade_stance", type="int", description="1=very_protectionist to 5=very_globalist"
    ),
    WeaviatePropertyDef(
        name="tariff_mention", type="boolean", description="Contains tariff discussion"
    ),
    WeaviatePropertyDef(
        name="economic_outlook", type="int", description="1=very_negative to 5=very_positive"
    ),
    WeaviatePropertyDef(
        name="is_governor", type="boolean", description="Speaker is central bank governor"
    ),
    WeaviatePropertyDef(name="is_synthetic", type="boolean", description="Synthetic data marker"),
]


@dg.asset(
    description="Load enriched speeches data product from LakeFS for synthesis",
    group_name="synthetic_speeches",
    metadata={
        "layer": "input",
        "source": "lakefs",
    },
)
def enriched_data_for_synthesis(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    lakefs: LakeFSResource,
) -> pl.DataFrame:
    """Load enriched speeches data product from LakeFS.

    This DECOUPLES the synthetic pipeline from the ETL pipeline,
    allowing them to run independently. The ETL pipeline must complete
    first and store data in LakeFS before running synthesis.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (is_trial for path selection).
        lakefs: LakeFS resource for data versioning.

    Returns:
        DataFrame with enriched speeches including summaries.
    """
    # Determine path based on trial mode
    if config.is_trial:
        path = "central-bank-speeches/trial/speeches.parquet"
        context.log.info("TRIAL RUN: Loading from trial-specific LakeFS path")
    else:
        path = "central-bank-speeches/speeches.parquet"

    context.log.info(f"Loading enriched speeches from lakefs://data/main/{path}")

    # Download from LakeFS with proper exception handling
    try:
        lakefs_client = lakefs.get_client()
        response = lakefs_client.objects_api.get_object(
            repository="data",
            ref="main",
            path=path,
        )
    except LakeFSNotFoundError:
        raise ValueError(
            f"Enriched data not found at {path}. "
            "Run the ETL pipeline first to generate enriched speeches with summaries."
        ) from None
    except LakeFSConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to LakeFS. Ensure LakeFS is running and accessible. Details: {e}"
        ) from e
    except LakeFSError as e:
        raise RuntimeError(f"Failed to load enriched data from LakeFS: {e}") from e

    # Load as DataFrame (response is bytes directly)
    df = pl.read_parquet(io.BytesIO(response))
    context.log.info(f"Loaded {len(df)} enriched speeches from LakeFS")

    # Verify required columns exist
    required_columns = [
        "reference",
        "summary",
        "monetary_stance",
        "trade_stance",
        "economic_outlook",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in LakeFS data: {missing}. "
            "Run the ETL pipeline first to generate summaries and classifications."
        )

    # Validate dead letter columns if present
    _validate_input_data_quality(df, context)

    context.log.info(f"Loaded columns: {df.columns}")

    return df


def _validate_input_data_quality(
    df: pl.DataFrame,
    context: dg.AssetExecutionContext,
) -> None:
    """Validate input data quality by checking dead letter columns.

    Logs warnings when failed records are detected. Does NOT filter
    records - the synthesis will process all records including those
    with fallback values.

    Args:
        df: Input DataFrame to validate.
        context: Dagster context for logging.
    """
    total_records = len(df)

    # Check classification dead letter columns
    class_status_col = "_llm_status_class"
    summary_status_col = "_llm_status_summary"

    failed_classification = 0
    failed_summary = 0

    if class_status_col in df.columns:
        failed_classification = df.filter(pl.col(class_status_col) == "failed").height
        if failed_classification > 0:
            context.log.warning(
                f"Input data contains {failed_classification} records with failed "
                f"classification ({100 * failed_classification / total_records:.1f}%). "
                "These records use fallback values."
            )

    if summary_status_col in df.columns:
        failed_summary = df.filter(pl.col(summary_status_col) == "failed").height
        if failed_summary > 0:
            context.log.warning(
                f"Input data contains {failed_summary} records with failed "
                f"summaries ({100 * failed_summary / total_records:.1f}%). "
                "These records use fallback values."
            )

    # Calculate total unique failed records
    if class_status_col in df.columns or summary_status_col in df.columns:
        # Build filter for any failure
        failure_conditions = []
        if class_status_col in df.columns:
            failure_conditions.append(pl.col(class_status_col) == "failed")
        if summary_status_col in df.columns:
            failure_conditions.append(pl.col(summary_status_col) == "failed")

        if failure_conditions:
            combined_filter = failure_conditions[0]
            for cond in failure_conditions[1:]:
                combined_filter = combined_filter | cond

            total_failed = df.filter(combined_filter).height

            if total_failed > 0:
                failure_rate = 100 * total_failed / total_records
                context.log.info(
                    f"Input data quality: {total_failed}/{total_records} records "
                    f"({failure_rate:.1f}%) have at least one LLM failure"
                )

                # Warn if failure rate is high (>10%)
                if failure_rate > 10:
                    context.log.warning(
                        f"High failure rate ({failure_rate:.1f}%) in input data. "
                        "Consider reprocessing failed records before synthesis."
                    )
            else:
                context.log.info(
                    f"Input data quality: All {total_records} records have successful LLM results"
                )
    else:
        context.log.info(
            "Input data does not contain dead letter columns (legacy data or pre-retry pattern)"
        )


@dg.asset(
    description="Synthetic metadata + summaries generated by NVIDIA Safe Synthesizer",
    group_name="synthetic_speeches",
    metadata={
        "layer": "synthetic",
        "uses_gpu": "true",
        "gpu_orchestration": "KAI priority-based preemption",
    },
)
def synthetic_summaries(
    context: dg.AssetExecutionContext,
    enriched_data_for_synthesis: pl.DataFrame,
    safe_synth: SafeSynthesizerResource,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Generate synthetic speech metadata + summaries using Safe Synthesizer.

    Synthesizes METADATA + CLASSIFICATIONS + COMPACT SUMMARIES. Safe Synthesizer
    uses TinyLlama with limited context (~12K tokens with rope_scaling_factor=6),
    but our compact summaries (~1000 chars) fit within this budget.

    What we synthesize:
    - Categorical metadata: date, central_bank, speaker, title
    - Classification scores: monetary_stance, trade_stance, economic_outlook (1-5 scales)
    - Binary flags: tariff_mention, is_governor
    - Compact summary: ~1000 char bullet-point summary with specific details

    Why this works without underfitting:
    - Metadata fields are short strings or numeric values
    - Classifications capture overall sentiment/stance numerically
    - Compact summaries (~1000 chars) capture semantic nuance that numeric
      classifications miss: specific metrics, geographic focus, sector commentary,
      forward guidance language, risk factors, policy tools
    - Total context per record: ~1500 chars, well within TinyLlama's capacity

    What the compact summary captures (beyond numeric classifications):
    - Specific metrics cited (inflation %, GDP growth, unemployment rate)
    - Geographic focus (which regions/countries mentioned)
    - Sector commentary (housing, energy, labor, finance)
    - Forward guidance language (next quarter, 2024, medium-term)
    - Risk factors (supply chain, geopolitical, banking)
    - Policy tools (rates, QE, reserves, forward guidance)

    Why we exclude full text:
    - Full speeches are ~20000+ chars - way too long for TinyLlama
    - The synthetic dataset contains summaries only (no full text)

    GPU orchestration is handled automatically by KAI Scheduler:
    - Safe Synthesizer job runs with 'batch-high' priority (130)
    - KAI preempts NIM (priority 125) to free the GPU
    - After job completion, NIM Deployment restarts automatically

    Args:
        context: Dagster execution context for logging.
        enriched_data_for_synthesis: Enriched speeches from LakeFS (with classifications).
        safe_synth: Safe Synthesizer resource for privacy-preserving synthesis.

    Returns:
        Tuple of (synthetic metadata+summary DataFrame, evaluation report).
    """
    df = enriched_data_for_synthesis
    run_id = context.run_id or datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    context.log.info("Starting synthetic data generation with KAI GPU orchestration...")
    context.log.info("Training on METADATA + CLASSIFICATIONS + COMPACT SUMMARIES")
    context.log.info("KAI Scheduler will automatically preempt NIM to free the GPU")

    # Columns for synthesis - METADATA + CLASSIFICATIONS + COMPACT SUMMARY
    # Compact summaries (~1000 chars) capture semantic nuance beyond numeric classifications.
    # Full text is excluded (GPT-OSS expands summaries during speech generation step).
    synthesis_columns = [
        "reference",  # Unique identifier
        "date",  # Speech date
        "central_bank",  # Institution name (short string)
        "speaker",  # Speaker name (short string)
        "title",  # Speech title (short string)
        # Classifications from ETL pipeline - capture overall stance numerically
        "monetary_stance",  # 1-5 scale (dovish to hawkish)
        "trade_stance",  # 1-5 scale (protectionist to globalist)
        "economic_outlook",  # 1-5 scale (negative to positive)
        "tariff_mention",  # Binary 0/1
        "is_governor",  # Binary 0/1
        # Compact summary - captures specific details beyond numeric classifications
        "summary",  # ~1000 chars, bullet-point format (fits in context)
        # EXCLUDED: "text" - full speech text, way too long (~20000+ chars)
    ]

    # Select columns that exist
    available_columns = [c for c in synthesis_columns if c in df.columns]
    context.log.info(f"Synthesis columns: {available_columns}")

    # Log what we're preserving
    if "monetary_stance" in available_columns:
        context.log.info(f"Monetary stance distribution: {df['monetary_stance'].value_counts()}")
    if "trade_stance" in available_columns:
        context.log.info(f"Trade stance distribution: {df['trade_stance'].value_counts()}")
    if "economic_outlook" in available_columns:
        context.log.info(f"Economic outlook distribution: {df['economic_outlook'].value_counts()}")

    # Log summary statistics if included
    if "summary" in available_columns:
        summary_lengths = df["summary"].str.len_chars().to_list()
        avg_len = sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0
        max_len = max(summary_lengths) if summary_lengths else 0
        context.log.info(f"Summary lengths: avg={avg_len:.0f}, max={max_len} chars")

    df_for_synthesis = df.select(available_columns)

    # Convert to list of dicts for Safe Synthesizer
    data_for_synthesis = df_for_synthesis.to_dicts()

    context.log.info(f"Training on {len(data_for_synthesis)} records")
    context.log.info(
        "Summaries + classifications provide semantic content for embedding and search"
    )

    # Build Safe Synthesizer config based on dataset size
    # Per documentation: holdout requires 200+ records, recommended to disable for <500
    num_records = len(data_for_synthesis)
    synth_config: SafeSynthConfig = {
        "epsilon": 6.0,  # Recommended 4-12 for large datasets
        "piiReplacement": True,
        "runMiaEvaluation": True,
        "runAiaEvaluation": True,
        "training": {
            "rope_scaling_factor": 4,  # Extend context window for ~1000 char summaries
        },
        "generation": {
            "temperature": 0.9,  # Safe Synthesizer default for diversity
            "use_structured_generation": True,  # Better tabular output quality
        },
    }

    # Disable holdout for small datasets to avoid "Dataset must have at least 200 records" error
    if num_records < 500:
        synth_config["data"] = {"holdout": 0}
        context.log.warning(
            f"Dataset has {num_records} records (<500) - disabling holdout for synthesis"
        )

    # Single synthesis call with ALL data - wrapped with retry logic
    context.log.info("Starting Safe Synthesizer with retry support...")

    def do_synthesis() -> tuple[list[dict[str, Any]], SafeSynthEvaluationResult]:
        return safe_synth.synthesize(
            input_data=data_for_synthesis,
            run_id=run_id,
            config=synth_config,
        )

    synthetic_data, evaluation = retry_safe_synth_call(
        do_synthesis,
        run_id=run_id,
        config=SafeSynthRetryConfig(
            max_retries=3,
            initial_delay=30.0,  # Safe Synth jobs are slow to recover
            max_delay=300.0,  # Max 5 minutes between retries
        ),
        logger=context.log,
    )

    # Convert to DataFrame
    synthetic_df = pl.DataFrame(synthetic_data)

    # Add synthetic marker and regenerate IDs (replace original reference with synthetic ID)
    synthetic_df = synthetic_df.with_row_index("_row_idx")
    synthetic_df = synthetic_df.with_columns(
        [
            (pl.lit("SYNTH-") + pl.col("_row_idx").cast(pl.Utf8).str.zfill(6)).alias("reference"),
            pl.lit(True).alias("is_synthetic"),
        ]
    )
    synthetic_df = synthetic_df.drop("_row_idx")

    # Log synthetic classification distributions
    if "monetary_stance" in synthetic_df.columns:
        context.log.info(
            f"Synthetic monetary stance: {synthetic_df['monetary_stance'].value_counts()}"
        )
    if "trade_stance" in synthetic_df.columns:
        context.log.info(f"Synthetic trade stance: {synthetic_df['trade_stance'].value_counts()}")

    combined_evaluation = {
        "total_records": len(synthetic_df),
        "input_records": num_records,
        "mia_score": evaluation.get("mia_score") or 0,
        "aia_score": evaluation.get("aia_score") or 0,
        "quality_score": evaluation.get("quality_score") or 0,
        "privacy_passed": evaluation.get("privacy_passed", False),
        "job_id": evaluation.get("job_id", ""),
        "generated_at": datetime.now(UTC).isoformat(),
        "gpu_orchestration": "KAI priority-based preemption",
        "synthesis_type": "metadata-classification-summary-based",
        "synthesis_columns": available_columns,
        "includes_summary": "summary" in available_columns,
        # Config parameters for traceability
        "config": {
            "epsilon": synth_config["epsilon"],
            "rope_scaling_factor": synth_config["training"]["rope_scaling_factor"],
            "temperature": synth_config["generation"]["temperature"],
            "use_structured_generation": synth_config["generation"]["use_structured_generation"],
            "holdout_disabled": num_records < 500,
        },
    }

    context.log.info(
        f"Generated {len(synthetic_df)} synthetic records (metadata + classifications + summaries)"
    )
    context.log.info(f"Privacy passed: {combined_evaluation['privacy_passed']}")
    context.log.info(
        f"MIA score: {combined_evaluation['mia_score']}, AIA score: {combined_evaluation['aia_score']}"
    )

    return (synthetic_df, combined_evaluation)


@dg.asset(
    description="Privacy validation report and HTML evaluation report stored in LakeFS",
    group_name="synthetic_speeches",
    metadata={
        "layer": "validation",
        "destination": "lakefs",
    },
)
def synthetic_validation_report(
    context: dg.AssetExecutionContext,
    synthetic_summaries: tuple[pl.DataFrame, dict[str, Any]],
    lakefs: LakeFSResource,
) -> dict[str, Any]:
    """Store privacy validation report and HTML evaluation report in LakeFS.

    Contains MIA (Membership Inference Attack) and AIA (Attribute Inference Attack)
    evaluation scores to verify synthetic data privacy. Also saves the rich HTML
    evaluation report with SQS/DPS graphs and detailed metrics.

    Args:
        context: Dagster execution context for logging.
        synthetic_summaries: Tuple of (synthetic DataFrame, evaluation report).
        lakefs: LakeFS resource for data versioning.

    Returns:
        Validation report dictionary with privacy metrics.
    """
    from lakefs_sdk.models import CommitCreation  # type: ignore[attr-defined]

    _, evaluation = synthetic_summaries

    # Extract HTML report bytes (don't include in JSON report)
    html_report_bytes: bytes | None = evaluation.pop("html_report_bytes", None)

    # Add metadata
    report = {
        **evaluation,
        "report_version": "3.0",  # Version 3.0 for summary-based synthesis (no expansion)
        "report_type": "safe-synthesizer-evaluation",
        "pipeline_type": "summary-based-synthesis",
        "html_report_available": html_report_bytes is not None,
    }

    # Store in LakeFS with proper exception handling
    try:
        lakefs_client = lakefs.get_client()
    except LakeFSConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to LakeFS to store validation report. "
            f"Ensure LakeFS is running and accessible. Details: {e}"
        ) from e

    # Upload JSON report
    report_path = "central-bank-speeches/synthetic/validation_report.json"
    report_bytes = json.dumps(report, indent=2).encode()

    try:
        lakefs_client.objects_api.upload_object(
            repository="data",
            branch="main",
            path=report_path,
            content=report_bytes,
        )
        context.log.info(f"Stored validation report to lakefs://data/main/{report_path}")

        # Upload HTML evaluation report if available
        html_report_path = "central-bank-speeches/synthetic/evaluation_report.html"
        if html_report_bytes:
            lakefs_client.objects_api.upload_object(
                repository="data",
                branch="main",
                path=html_report_path,
                content=html_report_bytes,
            )
            context.log.info(
                f"Stored HTML evaluation report to lakefs://data/main/{html_report_path}"
            )
        else:
            context.log.warning("HTML evaluation report not available from Safe Synthesizer")
    except LakeFSError as e:
        raise RuntimeError(f"Failed to upload validation report to LakeFS: {e}") from e

    # Commit (skip if no changes)
    try:
        lakefs_client.commits_api.commit(
            repository="data",
            branch="main",
            commit_creation=CommitCreation(
                message="Add synthetic data validation report and HTML evaluation (summary-based synthesis)",
                metadata={
                    "dagster_run_id": context.run_id or "",
                    "mia_score": str(report.get("mia_score", "")),
                    "aia_score": str(report.get("aia_score", "")),
                    "privacy_passed": str(report.get("privacy_passed", "")),
                    "synthesis_type": "summary-based",
                    "html_report_available": str(report.get("html_report_available", False)),
                },
                date=None,
                allow_empty=False,
            ),
        )
    except Exception as e:
        if "no changes" in str(e).lower():
            context.log.info("No changes to commit (report already exists in LakeFS)")
        else:
            raise

    return report


@dg.asset(
    description="Embeddings for synthetic summaries",
    group_name="synthetic_speeches",
    metadata={
        "layer": "enriched",
        "uses_nim_embedding": "true",
    },
)
def synthetic_embeddings(
    context: dg.AssetExecutionContext,
    synthetic_summaries: tuple[pl.DataFrame, dict[str, Any]],
    nim_embedding: NIMEmbeddingResource,
    minio: MinIOResource,
) -> tuple[pl.DataFrame, list[list[float]]]:
    """Generate embeddings for synthetic summaries.

    Uses local NIM embedding model. Embeds the compact summary which captures
    the semantic content of each synthetic record.

    Uses checkpointing to save progress every 32 rows, allowing recovery
    from failures without reprocessing completed embeddings.

    Args:
        context: Dagster execution context for logging.
        synthetic_summaries: Tuple of (synthetic DataFrame, evaluation).
        nim_embedding: NIM embedding resource for vector generation.
        minio: MinIO resource for checkpoint storage.

    Returns:
        Tuple of (DataFrame, list of 1024-dim embedding vectors).
    """
    df, _ = synthetic_summaries

    # Create checkpoint manager for embeddings
    checkpoint_mgr = LLMCheckpointManager(
        minio=minio,
        asset_name="synthetic_embeddings",
        run_id=context.run_id,
        checkpoint_interval=32,  # Match embedding batch size
    )

    # Load existing checkpoint
    existing_checkpoint = checkpoint_mgr.load()
    processed_refs: set[str] = set()
    if existing_checkpoint is not None:
        processed_refs = set(existing_checkpoint["reference"].to_list())
        context.log.info(f"Loaded checkpoint with {len(processed_refs)} embeddings")

    # Filter to unprocessed rows
    to_process = df.filter(~pl.col("reference").is_in(list(processed_refs)))
    context.log.info(
        f"Processing {len(to_process)} remaining rows (skipping {len(processed_refs)} already done)"
    )

    # Process in batches with checkpointing
    batch_size = 32
    rows = to_process.to_dicts()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]

        # Prepare texts for this batch
        texts: list[str] = []
        for row in batch:
            title = row.get("title", "") or ""
            summary = row.get("summary", "") or ""
            combined = f"{title}\n\n{summary}"
            texts.append(combined)

        # Generate embeddings for batch
        batch_embeddings = nim_embedding.embed_texts(texts, batch_size=batch_size)

        # Save to checkpoint
        for j, row in enumerate(batch):
            checkpoint_mgr.save_batch(
                [
                    {
                        "reference": row["reference"],
                        "embedding": batch_embeddings[j],
                    }
                ],
                force=(j == len(batch) - 1),
            )  # Force save at end of batch

        context.log.info(f"Checkpoint saved: {checkpoint_mgr.processed_count} embeddings complete")

    # Finalize and get all results
    final_checkpoint = checkpoint_mgr.finalize()

    # Clean up checkpoint on success
    checkpoint_mgr.cleanup()

    # Build embeddings list in DataFrame order
    embedding_map: dict[str, list[float]] = {}
    if final_checkpoint is not None:
        for row in final_checkpoint.to_dicts():
            embedding_map[row["reference"]] = row["embedding"]

    embeddings: list[list[float]] = []
    for row in df.iter_rows(named=True):
        ref = row["reference"]
        if ref in embedding_map:
            embeddings.append(embedding_map[ref])
        else:
            # This shouldn't happen, but fallback to generating
            title = row.get("title", "") or ""
            summary = row.get("summary", "") or ""
            combined = f"{title}\n\n{summary}"
            embeddings.append(nim_embedding.embed_text(combined))

    context.log.info(f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")

    return (df, embeddings)


@dg.asset(
    description="Synthetic summaries data product in LakeFS",
    group_name="synthetic_speeches",
    metadata={
        "layer": "output",
        "destination": "lakefs",
    },
)
def synthetic_data_product(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    synthetic_summaries: tuple[pl.DataFrame, dict[str, Any]],
    lakefs: LakeFSResource,
) -> dict[str, Any]:
    """Store synthetic summaries as versioned data product in LakeFS.

    The synthetic dataset contains metadata, classifications, and compact summaries
    (no full speech text). Uses trial-specific path when is_trial=True.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (is_trial for path selection).
        synthetic_summaries: Tuple of (synthetic DataFrame, evaluation).
        lakefs: LakeFS resource for data versioning.

    Returns:
        Dictionary with storage metadata (path, commit_id, counts).
    """
    from lakefs_sdk.models import CommitCreation  # type: ignore[attr-defined]

    df, _ = synthetic_summaries

    # Add timestamp
    df = df.with_columns(pl.lit(datetime.now(UTC).isoformat()).alias("generated_at"))

    # Serialize to Parquet
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    # Store in LakeFS - use trial path if is_trial
    try:
        lakefs_client = lakefs.get_client()
    except LakeFSConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to LakeFS to store synthetic data product. "
            f"Ensure LakeFS is running and accessible. Details: {e}"
        ) from e

    if config.is_trial:
        path = "central-bank-speeches/synthetic/trial/speeches.parquet"
        context.log.info("TRIAL RUN: Using trial-specific LakeFS path for synthetic data")
    else:
        path = "central-bank-speeches/synthetic/speeches.parquet"

    try:
        lakefs_client.objects_api.upload_object(
            repository="data",
            branch="main",
            path=path,
            content=parquet_bytes,
        )
    except LakeFSError as e:
        raise RuntimeError(f"Failed to upload synthetic data product to LakeFS: {e}") from e

    # Note: Direct SDK calls raise standard exceptions, not LakeFSError
    commit_id = None
    try:
        commit = lakefs_client.commits_api.commit(
            repository="data",
            branch="main",
            commit_creation=CommitCreation(
                message=f"Update synthetic summaries data product ({len(df)} records, summary-based synthesis)",
                metadata={
                    "dagster_run_id": context.run_id or "",
                    "num_records": str(len(df)),
                    "is_synthetic": "true",
                    "synthesis_type": "summary-based",
                },
                date=None,
                allow_empty=False,
            ),
        )
        commit_id = commit.id
        context.log.info(f"Committed synthetic data to LakeFS: {commit_id}")
    except Exception as e:
        if "no changes" in str(e).lower():
            context.log.info("No changes to commit (data already exists in LakeFS)")
        else:
            raise

    return {
        "path": f"lakefs://data/main/{path}",
        "commit_id": commit_id,
        "num_records": len(df),
    }


@dg.asset(
    description="Synthetic summaries indexed in Weaviate",
    group_name="synthetic_speeches",
    metadata={
        "layer": "output",
        "destination": "weaviate",
    },
)
def synthetic_weaviate_index(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    synthetic_embeddings: tuple[pl.DataFrame, list[list[float]]],
    weaviate: WeaviateResource,
) -> dict[str, Any]:
    """Index synthetic summaries in separate Weaviate collection.

    Creates SyntheticSpeeches collection (separate from CentralBankSpeeches)
    per INV-P004 (Synthetic Data Isolation). Contains metadata, classifications,
    and compact summaries (no full speech text).
    Uses trial-specific collection when is_trial=True.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (is_trial for collection selection).
        synthetic_embeddings: Tuple of (DataFrame, embeddings) from embedding step.
        weaviate: Weaviate resource for vector storage.

    Returns:
        Dictionary with indexing metadata (collection, count, dimensions).
    """
    df, embeddings = synthetic_embeddings

    # Use trial collection if is_trial
    if config.is_trial:
        collection_name = "SyntheticSpeechesTrial"
        context.log.info("TRIAL RUN: Using trial-specific Weaviate collection for synthetic data")
    else:
        collection_name = "SyntheticSpeeches"

    # Ensure collection exists with updated schema
    try:
        weaviate.ensure_collection(
            name=collection_name,
            properties=SYNTHETIC_SCHEMA,
            vector_dimensions=len(embeddings[0]),
        )
    except WeaviateConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to Weaviate to create collection {collection_name}. "
            f"Ensure Weaviate is running and accessible. Details: {e}"
        ) from e
    except WeaviateCollectionError as e:
        raise RuntimeError(
            f"Failed to create or verify Weaviate collection {collection_name}: {e}"
        ) from e

    # Prepare objects with all fields (no full text, only summary)
    objects: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        objects.append(
            {
                "reference": row["reference"],
                "date": str(row.get("date", "")),
                "central_bank": row.get("central_bank", "Unknown"),
                "speaker": row.get("speaker", "Unknown"),
                "title": row.get("title", "Untitled"),
                "summary": (row.get("summary", "") or "")[:2000],
                "monetary_stance": int(row.get("monetary_stance", 3)),
                "trade_stance": int(row.get("trade_stance", 3)),
                "tariff_mention": bool(row.get("tariff_mention", 0)),
                "economic_outlook": int(row.get("economic_outlook", 3)),
                "is_governor": bool(row.get("is_governor", False)),
                "is_synthetic": True,
            }
        )

    # Insert objects with embeddings
    try:
        count = weaviate.insert_objects(
            collection_name=collection_name,
            objects=objects,
            vectors=embeddings,
        )
    except WeaviateConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to Weaviate to insert objects into {collection_name}. "
            f"Ensure Weaviate is running and accessible. Details: {e}"
        ) from e
    except WeaviateCollectionError as e:
        raise RuntimeError(
            f"Failed to insert {len(objects)} objects into Weaviate collection {collection_name}: {e}"
        ) from e

    context.log.info(
        f"Indexed {count} synthetic summaries in Weaviate collection: {collection_name}"
    )

    return {
        "collection": collection_name,
        "object_count": count,
        "vector_dimensions": len(embeddings[0]),
    }


# ============================================================================
# INTERMEDIATE SNAPSHOTS - Persist intermediate stages to LakeFS
# ============================================================================


@dg.asset(
    description="Snapshot of synthetic summaries in LakeFS",
    group_name="synthetic_speeches",
    metadata={
        "layer": "intermediate",
        "destination": "lakefs",
    },
)
def synthetic_summaries_snapshot(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    synthetic_summaries: tuple[pl.DataFrame, dict[str, Any]],
    lakefs: LakeFSResource,
    synthetic_validation_report: dict[str, Any],  # Dependency to prevent concurrent commits
) -> dict[str, Any]:
    """Persist synthetic summaries to LakeFS for debugging and recovery.

    Stores the Safe Synthesizer generated summaries with their metadata.

    Depends on synthetic_validation_report to ensure sequential LakeFS commits
    (prevents "predicate failed" errors from concurrent commits).

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (is_trial for path selection).
        synthetic_summaries: Tuple of (DataFrame, evaluation metrics).
        lakefs: LakeFS resource for data versioning.
        synthetic_validation_report: Previous snapshot (unused, forces sequential execution).

    Returns:
        Dictionary with storage metadata.
    """
    del synthetic_validation_report  # Unused, exists only to enforce dependency
    from lakefs_sdk.models import CommitCreation  # type: ignore[attr-defined]

    df, evaluation = synthetic_summaries

    # Serialize to Parquet
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    # Get LakeFS client with proper exception handling
    try:
        lakefs_client = lakefs.get_client()
    except LakeFSConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to LakeFS to store synthetic summaries snapshot. "
            f"Ensure LakeFS is running and accessible. Details: {e}"
        ) from e

    # Upload to LakeFS intermediate path
    if config.is_trial:
        path = "synthetic-speeches/trial/intermediate/summaries.parquet"
    else:
        path = "synthetic-speeches/intermediate/summaries.parquet"

    try:
        lakefs_client.objects_api.upload_object(
            repository="data",
            branch="main",
            path=path,
            content=parquet_bytes,
        )
    except LakeFSError as e:
        raise RuntimeError(f"Failed to upload synthetic summaries snapshot to LakeFS: {e}") from e

    # Create commit
    # Note: Direct SDK calls raise standard exceptions, not LakeFSError
    commit_id = None
    try:
        commit = lakefs_client.commits_api.commit(
            repository="data",
            branch="main",
            commit_creation=CommitCreation(
                message=f"Snapshot: synthetic summaries ({len(df)} records)",
                metadata={
                    "dagster_run_id": context.run_id or "",
                    "snapshot_type": "synthetic_summaries",
                    "num_records": str(len(df)),
                },
                date=None,
                allow_empty=False,
            ),
        )
        commit_id = commit.id
        context.log.info(f"Synthetic summaries snapshot committed: {commit_id}")
    except Exception as e:
        if "no changes" in str(e).lower():
            context.log.info("No changes to commit (snapshot already exists)")
        else:
            raise

    context.log.info(f"Synthetic summaries snapshot: {len(df)} records")

    return {
        "path": f"lakefs://data/main/{path}",
        "commit_id": commit_id,
        "num_records": len(df),
        "evaluation": evaluation,
    }


@dg.asset(
    description="Snapshot of synthetic embeddings in LakeFS",
    group_name="synthetic_speeches",
    metadata={
        "layer": "intermediate",
        "destination": "lakefs",
    },
)
def synthetic_embeddings_snapshot(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    synthetic_embeddings: tuple[pl.DataFrame, list[list[float]]],
    lakefs: LakeFSResource,
    synthetic_summaries_snapshot: dict[str, Any],  # Dependency to prevent concurrent commits
) -> dict[str, Any]:
    """Persist synthetic embeddings to LakeFS for debugging and recovery.

    Stores the embedding vectors alongside their references.

    Depends on synthetic_summaries_snapshot to ensure sequential LakeFS commits
    (prevents "predicate failed" errors from concurrent commits).

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (is_trial for path selection).
        synthetic_embeddings: Tuple of (DataFrame, embeddings list).
        lakefs: LakeFS resource for data versioning.
        synthetic_summaries_snapshot: Previous snapshot (unused, forces sequential execution).

    Returns:
        Dictionary with storage metadata.
    """
    del synthetic_summaries_snapshot  # Unused, exists only to enforce dependency
    from lakefs_sdk.models import CommitCreation  # type: ignore[attr-defined]

    df, embeddings = synthetic_embeddings

    # Create DataFrame with embeddings
    embedding_records: list[dict[str, object]] = []
    for i, row in enumerate(df.iter_rows(named=True)):
        embedding_records.append(
            {
                "reference": row["reference"],
                "embedding": embeddings[i],
            }
        )
    df_snapshot = pl.DataFrame(embedding_records)

    # Serialize to Parquet
    buffer = io.BytesIO()
    df_snapshot.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    # Get LakeFS client with proper exception handling
    try:
        lakefs_client = lakefs.get_client()
    except LakeFSConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to LakeFS to store synthetic embeddings snapshot. "
            f"Ensure LakeFS is running and accessible. Details: {e}"
        ) from e

    # Upload to LakeFS intermediate path
    if config.is_trial:
        path = "synthetic-speeches/trial/intermediate/embeddings.parquet"
    else:
        path = "synthetic-speeches/intermediate/embeddings.parquet"

    try:
        lakefs_client.objects_api.upload_object(
            repository="data",
            branch="main",
            path=path,
            content=parquet_bytes,
        )
    except LakeFSError as e:
        raise RuntimeError(f"Failed to upload synthetic embeddings snapshot to LakeFS: {e}") from e

    # Create commit
    # Note: Direct SDK calls raise standard exceptions, not LakeFSError
    commit_id = None
    try:
        commit = lakefs_client.commits_api.commit(
            repository="data",
            branch="main",
            commit_creation=CommitCreation(
                message=f"Snapshot: synthetic embeddings ({len(df_snapshot)} records, {len(embeddings[0])}d)",
                metadata={
                    "dagster_run_id": context.run_id or "",
                    "snapshot_type": "synthetic_embeddings",
                    "num_records": str(len(df_snapshot)),
                    "dimensions": str(len(embeddings[0])),
                },
                date=None,
                allow_empty=False,
            ),
        )
        commit_id = commit.id
        context.log.info(f"Synthetic embeddings snapshot committed: {commit_id}")
    except Exception as e:
        if "no changes" in str(e).lower():
            context.log.info("No changes to commit (snapshot already exists)")
        else:
            raise

    context.log.info(f"Synthetic embeddings snapshot: {len(df_snapshot)} vectors")
    context.log.info(f"Snapshot size: {len(parquet_bytes) / 1024 / 1024:.1f} MB")

    return {
        "path": f"lakefs://data/main/{path}",
        "commit_id": commit_id,
        "num_records": len(df_snapshot),
        "dimensions": len(embeddings[0]),
        "size_mb": len(parquet_bytes) / 1024 / 1024,
    }


# Export all synthetic speech assets (summary-based, no full text expansion)
synthetic_speeches_assets = [
    enriched_data_for_synthesis,
    synthetic_summaries,
    synthetic_validation_report,
    synthetic_embeddings,
    synthetic_data_product,
    synthetic_weaviate_index,
    # Intermediate snapshots for debugging/recovery
    synthetic_summaries_snapshot,
    synthetic_embeddings_snapshot,
]
