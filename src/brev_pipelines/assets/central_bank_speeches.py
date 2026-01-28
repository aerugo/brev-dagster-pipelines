"""Central Bank Speeches ETL Pipeline.

This pipeline demonstrates end-to-end AI data product development:
1. Ingest dataset from Kaggle
2. Version data in LakeFS
3. Generate embeddings via local NIM embedding model
4. Multi-dimensional classification via GPT-OSS (monetary, trade, outlook, tariffs)
5. Generate summaries via GPT-OSS
6. Store enriched data product in LakeFS
7. Index text and embeddings in Weaviate for vector search

All AI inference uses local NIM endpoints - no external API dependencies.

Data enrichment for synthetic data pipeline:
- Numeric classifications (1-5 scales): Capture overall sentiment/stance
- Bullet-point summaries: Key points from each speech
- Both fit within Safe Synthesizer's context window for faithful reproduction

Trial Run Mode:
    Run with sample_size config to test with limited records:
    - speeches_trial_run job: 10 records
    - Or use run config: {"ops": {"raw_speeches": {"config": {"sample_size": 10}}}}
"""

from datetime import UTC, datetime
from typing import Any, cast

import dagster as dg
import polars as pl

from brev_pipelines.config import PipelineConfig
from brev_pipelines.io_managers.checkpoint import LLMCheckpointManager, process_with_checkpoint
from brev_pipelines.resources.k8s_scaler import K8sScalerResource
from brev_pipelines.resources.llm_retry import (
    RetryConfig,
    retry_classification,
    retry_with_backoff,
    validate_summary_response,
)
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMResource
from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource
from brev_pipelines.resources.weaviate import (
    WeaviateCollectionError,
    WeaviateConnectionError,
    WeaviateResource,
)
from brev_pipelines.types import (
    LLMAssetMetadata,
    LLMFailureBreakdown,
    SpeechClassification,
    WeaviatePropertyDef,
)


def _ensure_nim_reasoning_ready(
    context: dg.AssetExecutionContext,
    k8s_scaler: K8sScalerResource,
) -> None:
    """Ensure nim-reasoning is scaled up and ready before LLM calls.

    The embedding step scales down nim-reasoning to free GPU memory.
    This helper ensures it's running before classification/summarization.

    Args:
        context: Dagster execution context for logging.
        k8s_scaler: Kubernetes scaler to manage deployments.
    """
    current_replicas = k8s_scaler.get_replicas("nim-reasoning", "nvidia-ai")

    if current_replicas == 0:
        context.log.info("nim-reasoning is scaled to 0, scaling up before LLM calls")
        k8s_scaler.scale(
            deployment="nim-reasoning",
            namespace="nvidia-ai",
            replicas=1,
            wait_ready=True,
        )
        context.log.info("nim-reasoning is now ready")
    else:
        context.log.debug(f"nim-reasoning already running with {current_replicas} replicas")


# Collection schema for Weaviate
SPEECHES_SCHEMA: list[WeaviatePropertyDef] = [
    WeaviatePropertyDef(name="reference", type="text", description="Unique identifier from source"),
    WeaviatePropertyDef(name="date", type="text", description="Speech date (ISO format)"),
    WeaviatePropertyDef(name="central_bank", type="text", description="Issuing institution"),
    WeaviatePropertyDef(name="speaker", type="text", description="Speaker name"),
    WeaviatePropertyDef(name="title", type="text", description="Speech title"),
    WeaviatePropertyDef(name="text", type="text", description="Full speech text"),
    WeaviatePropertyDef(
        name="monetary_stance", type="int", description="1=very_dovish to 5=very_hawkish"
    ),
    WeaviatePropertyDef(
        name="trade_stance", type="int", description="1=very_protectionist to 5=very_globalist"
    ),
    WeaviatePropertyDef(
        name="tariff_mention",
        type="boolean",
        description="Contains tariff/protectionist discussion",
    ),
    WeaviatePropertyDef(
        name="economic_outlook", type="int", description="1=very_negative to 5=very_positive"
    ),
    WeaviatePropertyDef(
        name="is_governor", type="boolean", description="Speaker is governor/president/chair"
    ),
]


@dg.asset(
    description="Raw central bank speeches from Kaggle dataset",
    group_name="central_bank_speeches",
    io_manager_key="lakefs_parquet",
    metadata={
        "layer": "raw",
        "source": "kaggle/davidgauthier/central-bank-speeches",
    },
)
def raw_speeches(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Ingest central bank speeches dataset from Kaggle.

    Downloads the dataset using KaggleHub. The IO manager handles
    persistence to LakeFS with automatic versioning.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (sample_size for trial runs).

    Returns:
        Raw speeches DataFrame from Kaggle.
    """
    import os

    import kagglehub

    context.log.info("Downloading central-bank-speeches dataset from Kaggle...")

    # Download dataset first to get the local path
    dataset_path = kagglehub.dataset_download("davidgauthier/central-bank-speeches")
    context.log.info(f"Dataset downloaded to: {dataset_path}")

    # List files in the dataset
    files = os.listdir(dataset_path)
    context.log.info(f"Files in dataset: {files}")

    # Find CSV file (most Kaggle datasets use CSV)
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        msg = f"No CSV files found in dataset. Available files: {files}"
        raise ValueError(msg)

    # Load the first CSV file
    csv_path = os.path.join(dataset_path, csv_files[0])
    context.log.info(f"Loading: {csv_path}")

    df = pl.read_csv(csv_path)
    context.log.info(f"Loaded {len(df)} speeches from Kaggle")

    # Apply sample_size limit for trial runs
    if config.sample_size > 0:
        original_count = len(df)
        df = df.head(config.sample_size)
        context.log.info(
            f"TRIAL RUN: Limited to {config.sample_size} records (from {original_count})"
        )

    # Log column info
    context.log.info(f"Columns: {df.columns}")
    context.log.info(f"Schema: {df.schema}")

    # IO manager handles persistence to LakeFS
    return df


@dg.asset(
    description="Cleaned speeches with null values filled",
    group_name="central_bank_speeches",
    io_manager_key="lakefs_parquet",
    metadata={"layer": "cleaned"},
)
def cleaned_speeches(
    context: dg.AssetExecutionContext,
    raw_speeches: pl.DataFrame,
) -> pl.DataFrame:
    """Clean the raw speeches data.

    Performs the following transformations:
    - Fill null values for all columns
    - Filter out empty speeches

    Args:
        context: Dagster execution context for logging.
        raw_speeches: Raw DataFrame from Kaggle.

    Returns:
        Cleaned DataFrame with nulls filled.
    """
    df = raw_speeches

    context.log.info(f"Input columns: {df.columns}")
    context.log.info(f"Input schema: {df.schema}")

    # Fill nulls for all columns based on their type
    # Note: Pyright warnings about polars types are expected (incomplete stubs)
    fill_expressions: list[pl.Expr] = []
    for col_name in df.columns:
        dtype = df.schema[col_name]
        dtype_str = str(dtype)
        if dtype_str in ("Utf8", "String"):
            fill_expressions.append(pl.col(col_name).fill_null(""))
        elif "Int" in dtype_str or "UInt" in dtype_str:
            fill_expressions.append(pl.col(col_name).fill_null(0))
        elif "Float" in dtype_str:
            fill_expressions.append(pl.col(col_name).fill_null(0.0))
        elif dtype_str == "Boolean":
            fill_expressions.append(pl.col(col_name).fill_null(False))
        # For other types (Date, Datetime, etc.), leave as-is

    if fill_expressions:
        df = df.with_columns(fill_expressions)

    # Filter out empty speeches (less than 100 chars)
    df = df.filter(pl.col("text").str.len_chars() > 100)

    context.log.info(f"Cleaned {len(df)} speeches")
    context.log.info(f"Output columns: {df.columns}")

    return df


@dg.asset(
    description="Speeches with generated embeddings from local NIM",
    group_name="central_bank_speeches",
    metadata={
        "layer": "enriched",
        "uses_nim_embedding": "true",
    },
    # Run after classification completes - embeddings scales down
    # nim-reasoning to free GPU memory, which would break concurrent LLM calls
    deps=[dg.AssetDep("speech_classification")],
)
def speech_embeddings(
    context: dg.AssetExecutionContext,
    cleaned_speeches: pl.DataFrame,
    speech_summaries: pl.DataFrame,
    nim_embedding: NIMEmbeddingResource,
    minio: MinIOResource,
    k8s_scaler: K8sScalerResource,
) -> tuple[pl.DataFrame, list[list[float]]]:
    """Generate embeddings for all speeches using local NIM.

    Uses the speech summary for embedding instead of full text to stay within
    the model's 512 token limit. Summaries are truncated to ~1500 chars to be safe.

    Uses nv-embedqa-e5-v5 model (1024 dimensions).
    Returns tuple of (DataFrame, embeddings) for downstream storage.

    In production: Uses checkpointing to save progress every 32 rows,
    allowing recovery from failures without reprocessing completed embeddings.

    In local dev: Skips checkpointing when NIM service is unavailable and
    mock embeddings are being used (they're instant and deterministic).

    Args:
        context: Dagster execution context for logging.
        cleaned_speeches: Cleaned DataFrame with speech text.
        speech_summaries: DataFrame with speech summaries from IO manager.
        nim_embedding: NIM embedding resource for vector generation.
        minio: MinIO resource for checkpoint storage.
        k8s_scaler: Kubernetes scaler to manage GPU resources.

    Returns:
        Tuple of (DataFrame, list of 1024-dim embedding vectors).
    """
    context.log.info(f"Received {len(speech_summaries)} summaries from IO manager")

    # Join summaries with cleaned speeches
    df = cleaned_speeches.join(
        speech_summaries.select(["reference", "summary"]),
        on="reference",
        how="left",
    )

    # Check if NIM service is available (determines mock mode)
    use_mock = nim_embedding.use_mock_fallback and not nim_embedding.health_check()

    if use_mock:
        # Mock mode: skip checkpointing, generate all embeddings directly
        context.log.info(f"Mock mode: generating {len(df)} embeddings without checkpointing")

        # Prepare all texts using summaries (truncated to fit 512 token limit)
        texts: list[str] = []
        for row in df.iter_rows(named=True):
            title = row.get("title", "") or ""
            summary = row.get("summary", "") or ""
            # Use summary if available, fallback to title
            combined = f"{title}\n\n{summary}" if summary else title or "Untitled speech"
            # Truncate to ~1500 chars to stay within 512 token limit
            texts.append(combined[:1500])

        # Generate all embeddings in one batch (mock is instant)
        embeddings = nim_embedding.embed_texts(texts, batch_size=32)
        context.log.info(
            f"Generated {len(embeddings)} mock embeddings, dimension: {len(embeddings[0])}"
        )

        return (df, embeddings)

    # Production mode: use checkpointing for recovery
    # Scale nim-reasoning to 0 (its default) to free GPU for embedding model.
    # Don't use temporarily_scale — nim-reasoning defaults to 0 replicas and
    # will be scaled up on demand by _ensure_nim_reasoning_ready in the next job.
    context.log.info("Production mode: scaling down nim-reasoning for embedding")
    k8s_scaler.scale(
        deployment="nim-reasoning",
        namespace="nvidia-ai",
        replicas=0,
        wait_ready=False,
    )
    # Restore safe-synth now that nim-reasoning is down
    context.log.info("Restoring safe-synth after nim-reasoning scale-down")
    k8s_scaler.scale(
        deployment="nvidia-safe-synth-safe-synthesizer",
        namespace="nvidia-ai",
        replicas=1,
        wait_ready=False,
    )
    context.log.info("nim-reasoning scaled down, starting embedding generation")

    checkpoint_mgr = LLMCheckpointManager(
        minio=minio,
        asset_name="speech_embeddings",
        run_id=context.run_id,
        checkpoint_interval=32,
    )

    existing_checkpoint = checkpoint_mgr.load()
    processed_refs: set[str] = set()
    if existing_checkpoint is not None:
        processed_refs = set(existing_checkpoint["reference"].to_list())
        context.log.info(f"Loaded checkpoint with {len(processed_refs)} embeddings")

    to_process = df.filter(~pl.col("reference").is_in(list(processed_refs)))
    context.log.info(
        f"Processing {len(to_process)} remaining rows (skipping {len(processed_refs)} already done)"
    )

    batch_size = 32
    rows = to_process.to_dicts()
    total_batches = (len(rows) + batch_size - 1) // batch_size
    log_interval = max(1, total_batches // 10)

    for batch_num, i in enumerate(range(0, len(rows), batch_size)):
        batch = rows[i : i + batch_size]

        texts = []
        for row in batch:
            title = row.get("title", "") or ""
            summary = row.get("summary", "") or ""
            combined = f"{title}\n\n{summary}" if summary else title or "Untitled speech"
            combined = combined.replace("\x00", "").strip()
            combined = "".join(c if c.isprintable() or c in "\n\t" else " " for c in combined)
            combined = combined[:1500]
            if not combined or len(combined) < 10:
                combined = f"Speech: {title or 'Untitled'}"
            texts.append(combined)

        try:
            batch_embeddings = nim_embedding.embed_texts(texts, batch_size=batch_size)
        except Exception as e:
            context.log.error(f"Embedding batch {batch_num} failed: {e}")
            context.log.error(f"First text in batch (truncated): {texts[0][:200]}")
            raise

        for j, row in enumerate(batch):
            checkpoint_mgr.save_batch(
                [{"reference": row["reference"], "embedding": batch_embeddings[j]}],
                force=(j == len(batch) - 1),
            )

        if batch_num % log_interval == 0 or batch_num == total_batches - 1:
            context.log.info(
                f"Embedding progress: {checkpoint_mgr.processed_count}/{len(rows)} complete"
            )

    final_checkpoint = checkpoint_mgr.finalize()
    checkpoint_mgr.cleanup()

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
            title = row.get("title", "") or ""
            summary = row.get("summary", "") or ""
            if summary:
                combined = f"{title}\n\n{summary}"[:1500]
            else:
                combined = title[:1500] or "Untitled speech"
            embeddings.append(nim_embedding.embed_text(combined))

    context.log.info(f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")

    return (df, embeddings)


# Classification scale mappings
MONETARY_STANCE_SCALE = {
    "very_dovish": 1,
    "somewhat_dovish": 2,
    "neutral": 3,
    "somewhat_hawkish": 4,
    "very_hawkish": 5,
}

TRADE_STANCE_SCALE = {
    "very_protectionist": 1,
    "somewhat_protectionist": 2,
    "neutral": 3,
    "somewhat_globalist": 4,
    "very_globalist": 5,
}

OUTLOOK_SCALE = {
    "very_negative": 1,
    "somewhat_negative": 2,
    "neutral": 3,
    "somewhat_positive": 4,
    "very_positive": 5,
}

# Few-shot examples based on central bank communication research
# References:
# - Lucca & Trebbi (2009) "Measuring Central Bank Communication"
# - Hansen & McMahon (2016) "Shocking Language"
# - Apel & Blix Grimaldi (2014) "How Informative Are Central Bank Minutes?"
CLASSIFICATION_FEW_SHOT_EXAMPLES = """
Example 1 - Very Hawkish, Neutral Trade, Negative Outlook:
"Inflation remains unacceptably high at 6.2%, well above our 2% target. The Committee judges that ongoing increases in the policy rate will be appropriate. We are strongly committed to returning inflation to our target, and we will keep at it until the job is done. The labor market remains extremely tight, contributing to upward pressure on wages and prices."
Classification: {"monetary_stance": "very_hawkish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "somewhat_negative"}

Example 2 - Very Dovish, Neutral Trade, Negative Outlook:
"Economic activity has weakened considerably. We have cut our policy rate by 50 basis points and stand ready to act further if needed. The Committee will use its full range of tools to support the economy. Credit conditions have tightened significantly, and we are closely monitoring financial stability risks."
Classification: {"monetary_stance": "very_dovish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "very_negative"}

Example 3 - Neutral Monetary, Protectionist, Mentions Tariffs:
"We are monitoring the impact of recently announced tariffs on imported goods. These trade measures may affect inflation dynamics and supply chains. The central bank remains vigilant about pass-through effects from customs duties on consumer prices. Our monetary policy stance remains appropriate given current conditions."
Classification: {"monetary_stance": "neutral", "trade_stance": "somewhat_protectionist", "tariff_mention": 1, "economic_outlook": "neutral"}

Example 4 - Somewhat Hawkish, Globalist, Positive Outlook:
"The economy continues to expand at a solid pace. International trade flows remain robust, supporting our export-oriented sectors. We see benefits from our open trade agreements and cross-border investment. Given the strength of economic activity, we judge that some further gradual increases in the policy rate will be appropriate."
Classification: {"monetary_stance": "somewhat_hawkish", "trade_stance": "somewhat_globalist", "tariff_mention": 0, "economic_outlook": "somewhat_positive"}

Example 5 - Somewhat Dovish, Very Protectionist, Mentions Tariffs:
"We have lowered interest rates to support domestic industry facing headwinds from global competition. Import levies and trade barriers are necessary to protect strategic sectors. The government's tariff policy on steel and aluminum imports aligns with our objective of supporting domestic employment."
Classification: {"monetary_stance": "somewhat_dovish", "trade_stance": "very_protectionist", "tariff_mention": 1, "economic_outlook": "neutral"}
"""


@dg.asset(
    description="Multi-dimensional speech classification using GPT-OSS 120B",
    group_name="central_bank_speeches",
    io_manager_key="lakefs_parquet",
    metadata={
        "layer": "enriched",
        "uses_gpu": "true",
        "model": "GPT-OSS 120B",
    },
)
def speech_classification(
    context: dg.AssetExecutionContext,
    cleaned_speeches: pl.DataFrame,
    nim_reasoning: NIMResource,
    minio: MinIOResource,
    k8s_scaler: K8sScalerResource,
) -> pl.DataFrame:
    """Classify speeches on multiple dimensions using GPT-OSS 120B.

    Performs four classifications per speech in a single LLM call:
    1. Monetary stance: very_dovish to very_hawkish (1-5 scale)
    2. Trade stance: very_protectionist to very_globalist (1-5 scale)
    3. Tariff mention: binary (0/1)
    4. Economic outlook: very_negative to very_positive (1-5 scale)

    Uses checkpointing to save progress every 10 rows, allowing recovery
    from failures without reprocessing completed classifications.

    Args:
        context: Dagster execution context for logging.
        cleaned_speeches: Cleaned DataFrame with speech text.
        nim_reasoning: NIM reasoning resource (GPT-OSS 120B).
        minio: MinIO resource for checkpoint storage.
        k8s_scaler: Kubernetes scaler to ensure NIM is ready.

    Returns:
        DataFrame with classification columns added.
    """
    # Ensure nim-reasoning is running before making LLM calls
    _ensure_nim_reasoning_ready(context, k8s_scaler)

    df = cleaned_speeches

    # Create checkpoint manager
    checkpoint_mgr = LLMCheckpointManager(
        minio=minio,
        asset_name="speech_classification",
        run_id=context.run_id,
        checkpoint_interval=10,
    )

    # Retry configuration for LLM calls
    retry_config = RetryConfig(
        max_retries=5,
        base_delay=1.0,
        exponential_base=2.0,
    )

    def classify_speech(row: dict[str, Any]) -> dict[str, Any]:
        """Classify a single speech with retry logic and dead letter tracking.

        Uses GPT-OSS generate_classification method which handles:
        - json_object response format (avoids vLLM bug #23120)
        - include_reasoning: false parameter
        - Harmony token cleanup
        - Pydantic validation

        This function is designed to NEVER raise exceptions - all errors are
        caught and recorded in the result dict with fallback values used.
        """
        reference = str(row.get("reference", "unknown"))
        text_excerpt = (row.get("text", "") or "")[:4000]

        def get_fallback() -> SpeechClassification:
            """Return neutral fallback classification."""
            return SpeechClassification(
                monetary_stance=3,
                trade_stance=3,
                tariff_mention=0,
                economic_outlook=3,
            )

        try:
            # Execute with retry wrapper using new generate_classification method
            # This handles Harmony tokens, json_object format, and validation
            result = retry_classification(
                nim_resource=nim_reasoning,
                speech_text=text_excerpt,
                record_id=reference,
                config=retry_config,
            )

            # Get values (either parsed or fallback)
            values = result.parsed_data if result.status == "success" else result.fallback_values
            if values is None:
                values = get_fallback()

            return {
                "reference": reference,
                "monetary_stance": values["monetary_stance"],
                "trade_stance": values["trade_stance"],
                "tariff_mention": values["tariff_mention"],
                "economic_outlook": values["economic_outlook"],
                "_llm_status": result.status,
                "_llm_error": result.error_message or "",  # Always string, never None
                "_llm_attempts": result.attempts,
                "_llm_fallback_used": result.fallback_used,
            }
        except Exception as e:
            # Catch-all: should never happen, but ensures job doesn't crash
            fallback = get_fallback()
            return {
                "reference": reference,
                "monetary_stance": fallback["monetary_stance"],
                "trade_stance": fallback["trade_stance"],
                "tariff_mention": fallback["tariff_mention"],
                "economic_outlook": fallback["economic_outlook"],
                "_llm_status": "failed",
                "_llm_error": f"Unexpected error: {e!s}",
                "_llm_attempts": 0,
                "_llm_fallback_used": True,
            }

    # Process with checkpointing (5 parallel workers for faster processing)
    context.log.info(f"Starting classification of {len(df)} speeches with checkpointing")
    results_df = process_with_checkpoint(
        df=df,
        id_column="reference",
        process_fn=classify_speech,
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,
        logger=context.log,
        parallel_workers=5,
    )

    # Clean up checkpoint on success
    checkpoint_mgr.cleanup()

    if results_df is None:
        msg = "Classification checkpoint returned no results"
        raise RuntimeError(msg)

    # Join results back to original DataFrame (including dead letter columns)
    df = df.join(
        results_df.select(
            [
                "reference",
                pl.col("monetary_stance").cast(pl.Int8),
                pl.col("trade_stance").cast(pl.Int8),
                pl.col("tariff_mention").cast(pl.Int8),
                pl.col("economic_outlook").cast(pl.Int8),
                pl.col("_llm_status").cast(pl.Utf8),
                pl.col("_llm_error").cast(pl.Utf8),
                pl.col("_llm_attempts").cast(pl.Int64),
                pl.col("_llm_fallback_used").cast(pl.Boolean),
            ]
        ),
        on="reference",
        how="left",
    )

    # Calculate classification statistics
    total = len(df)
    failed_df = df.filter(pl.col("_llm_status") == "failed")
    failed_count = failed_df.height
    success_count = total - failed_count
    success_rate = f"{100 * success_count / total:.1f}%" if total > 0 else "N/A"

    # Calculate failure breakdown by error type
    failure_breakdown: LLMFailureBreakdown = {
        "ValidationError": 0,
        "LLMTimeoutError": 0,
        "LLMRateLimitError": 0,
        "LLMServerError": 0,
        "unexpected_error": 0,
    }
    if failed_count > 0:
        error_types = failed_df["_llm_error"].to_list()
        for error in error_types:
            error_str = str(error) if error else ""
            if "ValidationError" in error_str:
                failure_breakdown["ValidationError"] += 1
            elif "LLMTimeoutError" in error_str or "timeout" in error_str.lower():
                failure_breakdown["LLMTimeoutError"] += 1
            elif "LLMRateLimitError" in error_str or "429" in error_str:
                failure_breakdown["LLMRateLimitError"] += 1
            elif "LLMServerError" in error_str or any(
                code in error_str for code in ("500", "502", "503", "504")
            ):
                failure_breakdown["LLMServerError"] += 1
            else:
                failure_breakdown["unexpected_error"] += 1

    # Calculate average attempts
    avg_attempts = df.select(pl.col("_llm_attempts").mean()).item() or 1.0

    # Get failed references (limit to 100)
    failed_refs = failed_df["reference"].to_list()[:100] if failed_count > 0 else []

    # Build metadata
    metadata: LLMAssetMetadata = {
        "total_processed": total,
        "successful": success_count,
        "failed": failed_count,
        "success_rate": success_rate,
        "failed_references": [str(ref) for ref in failed_refs],
        "failure_breakdown": failure_breakdown,
        "avg_attempts": float(avg_attempts),
        "total_duration_ms": 0,  # Not tracked at asset level
    }

    # Add metadata to Dagster context (single call - Dagster limitation)
    output_metadata: dict[str, object] = {
        "total_processed": total,
        "successful": success_count,
        "failed": failed_count,
        "success_rate": success_rate,
        "failure_breakdown": dict(failure_breakdown),
        "avg_attempts": round(metadata["avg_attempts"], 2),
    }
    if failed_count > 0:
        output_metadata["failed_references"] = metadata["failed_references"][:100]
    context.add_output_metadata(output_metadata)

    # Structured logging
    context.log.info("=" * 60)
    context.log.info("CLASSIFICATION SUMMARY")
    context.log.info("=" * 60)
    context.log.info(f"Total records:     {total}")
    context.log.info(f"Successful:        {success_count} ({success_rate})")
    context.log.info(f"Failed (fallback): {failed_count}")
    context.log.info(f"Average attempts:  {metadata['avg_attempts']:.2f}")

    if failed_count > 0:
        context.log.info("-" * 40)
        context.log.info("FAILURE BREAKDOWN:")
        breakdown_dict = cast("dict[str, int]", dict(failure_breakdown))
        for error_type, count in breakdown_dict.items():
            if count > 0:
                context.log.info(f"  {error_type}: {count}")
        context.log.info("-" * 40)
        context.log.warning(f"Failed references (first 10): {metadata['failed_references'][:10]}")

    context.log.info("=" * 60)

    # Classification distribution stats
    context.log.info(f"Monetary stance distribution: {df['monetary_stance'].value_counts()}")
    context.log.info(f"Trade stance distribution: {df['trade_stance'].value_counts()}")
    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    context.log.info(f"Speeches mentioning tariffs: {tariff_count}/{len(df)}")
    context.log.info(f"Economic outlook distribution: {df['economic_outlook'].value_counts()}")

    return df


@dg.asset(
    description="Compact speech summaries for Safe Synthesizer training",
    group_name="central_bank_speeches",
    io_manager_key="lakefs_parquet",
    metadata={
        "layer": "enriched",
        "uses_gpu": "true",
        "model": "GPT-OSS 120B",
    },
)
def speech_summaries(
    context: dg.AssetExecutionContext,
    cleaned_speeches: pl.DataFrame,
    nim_reasoning: NIMResource,
    minio: MinIOResource,
    k8s_scaler: K8sScalerResource,
) -> pl.DataFrame:
    """Generate bullet-point summaries of central bank speeches.

    Generates a simple summary of each speech capturing the key points
    in bullet-point format. The LLM is given freedom to determine what
    aspects of the speech are most important to highlight.

    Uses checkpointing to save progress every 10 rows, allowing recovery
    from failures without reprocessing completed summaries.

    Args:
        context: Dagster execution context for logging.
        cleaned_speeches: Cleaned DataFrame with speech text.
        nim_reasoning: NIM reasoning resource for summary generation (GPT-OSS 120B).
        minio: MinIO resource for checkpoint storage.
        k8s_scaler: Kubernetes scaler to ensure NIM is ready.

    Returns:
        DataFrame with reference and summary columns.
    """
    # Ensure nim-reasoning is running before making LLM calls
    _ensure_nim_reasoning_ready(context, k8s_scaler)

    df = cleaned_speeches

    # Create checkpoint manager
    checkpoint_mgr = LLMCheckpointManager(
        minio=minio,
        asset_name="speech_summaries",
        run_id=context.run_id,
        checkpoint_interval=10,
    )

    # Retry configuration for LLM calls
    retry_config = RetryConfig(
        max_retries=5,
        base_delay=1.0,
        exponential_base=2.0,
    )

    def summarize_speech(row: dict[str, Any]) -> dict[str, Any]:
        """Generate summary with retry logic and dead letter tracking.

        Uses structured JSON output with separate reasoning and summary fields
        to ensure clean summaries without any reasoning artifacts.

        This function is designed to NEVER raise exceptions - all errors are
        caught and recorded in the result dict with fallback values used.
        """
        reference = str(row.get("reference", "unknown"))
        title = row.get("title", "") or "Untitled"
        speaker = row.get("speaker", "") or "Unknown"
        central_bank = row.get("central_bank", "") or "Unknown"
        text = row.get("text", "") or ""

        def get_fallback() -> str:
            """Return fallback summary with basic info."""
            return f"• Topic: {title[:100]}\n• Speaker: {speaker}\n• Bank: {central_bank}"

        try:
            # Execute with retry wrapper using structured JSON output
            # The generate_summary method returns only the summary field from JSON
            result = retry_with_backoff(
                fn=lambda: nim_reasoning.generate_summary(
                    title=title,
                    speaker=speaker,
                    central_bank=central_bank,
                    text=text,
                    max_tokens=2000,
                    temperature=0.2,
                ),
                validate_fn=validate_summary_response,
                record_id=reference,
                fallback_fn=get_fallback,
                config=retry_config,
            )

            # Get summary (either parsed or fallback)
            summary = result.parsed_data if result.status == "success" else result.fallback_values
            if summary is None:
                summary = get_fallback()

            return {
                "reference": reference,
                "summary": summary,
                "_llm_status": result.status,
                "_llm_error": result.error_message or "",  # Always string, never None
                "_llm_attempts": result.attempts,
                "_llm_fallback_used": result.fallback_used,
            }
        except Exception as e:
            # Catch-all: should never happen, but ensures job doesn't crash
            return {
                "reference": reference,
                "summary": get_fallback(),
                "_llm_status": "failed",
                "_llm_error": f"Unexpected error: {e!s}",
                "_llm_attempts": 0,
                "_llm_fallback_used": True,
            }

    # Process with checkpointing (5 parallel workers for faster processing)
    context.log.info(f"Starting summarization of {len(df)} speeches with checkpointing")
    results_df = process_with_checkpoint(
        df=df,
        id_column="reference",
        process_fn=summarize_speech,
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,
        logger=context.log,
        parallel_workers=5,
    )

    # Clean up checkpoint on success
    checkpoint_mgr.cleanup()

    if results_df is None:
        msg = "Summarization checkpoint returned no results"
        raise RuntimeError(msg)

    # Calculate summary statistics
    total = len(results_df)
    failed_df = results_df.filter(pl.col("_llm_status") == "failed")
    failed_count = failed_df.height
    success_count = total - failed_count
    success_rate = f"{100 * success_count / total:.1f}%" if total > 0 else "N/A"

    # Calculate failure breakdown by error type
    failure_breakdown: LLMFailureBreakdown = {
        "ValidationError": 0,
        "LLMTimeoutError": 0,
        "LLMRateLimitError": 0,
        "LLMServerError": 0,
        "unexpected_error": 0,
    }
    if failed_count > 0:
        error_types = failed_df["_llm_error"].to_list()
        for error in error_types:
            error_str = str(error) if error else ""
            if "ValidationError" in error_str:
                failure_breakdown["ValidationError"] += 1
            elif "LLMTimeoutError" in error_str or "timeout" in error_str.lower():
                failure_breakdown["LLMTimeoutError"] += 1
            elif "LLMRateLimitError" in error_str or "429" in error_str:
                failure_breakdown["LLMRateLimitError"] += 1
            elif "LLMServerError" in error_str or any(
                code in error_str for code in ("500", "502", "503", "504")
            ):
                failure_breakdown["LLMServerError"] += 1
            else:
                failure_breakdown["unexpected_error"] += 1

    # Calculate average attempts
    avg_attempts = results_df.select(pl.col("_llm_attempts").mean()).item() or 1.0

    # Get failed references (limit to 100)
    failed_refs = failed_df["reference"].to_list()[:100] if failed_count > 0 else []

    # Add metadata to Dagster context (single call - Dagster limitation)
    output_metadata: dict[str, object] = {
        "total_processed": total,
        "successful": success_count,
        "failed": failed_count,
        "success_rate": success_rate,
        "failure_breakdown": dict(failure_breakdown),
        "avg_attempts": round(float(avg_attempts), 2),
    }
    if failed_count > 0:
        output_metadata["failed_references"] = [str(ref) for ref in failed_refs][:100]
    context.add_output_metadata(output_metadata)

    # Structured logging
    context.log.info("=" * 60)
    context.log.info("SUMMARIZATION SUMMARY")
    context.log.info("=" * 60)
    context.log.info(f"Total records:     {total}")
    context.log.info(f"Successful:        {success_count} ({success_rate})")
    context.log.info(f"Failed (fallback): {failed_count}")
    context.log.info(f"Average attempts:  {float(avg_attempts):.2f}")

    if failed_count > 0:
        context.log.info("-" * 40)
        context.log.info("FAILURE BREAKDOWN:")
        breakdown_dict = cast("dict[str, int]", dict(failure_breakdown))
        for error_type, count in breakdown_dict.items():
            if count > 0:
                context.log.info(f"  {error_type}: {count}")
        context.log.info("-" * 40)
        context.log.warning(
            f"Failed references (first 10): {[str(ref) for ref in failed_refs[:10]]}"
        )

    context.log.info("=" * 60)

    # Summary length statistics
    summaries = results_df["summary"].to_list()
    summary_lengths = [len(s) for s in summaries]
    avg_len = sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0
    max_len = max(summary_lengths) if summary_lengths else 0
    min_len = min(summary_lengths) if summary_lengths else 0
    context.log.info(f"Summary lengths: avg={avg_len:.0f} chars, min={min_len}, max={max_len}")

    return results_df


@dg.asset(
    description="Combined enriched speeches data product with summaries",
    group_name="central_bank_speeches",
    io_manager_key="lakefs_parquet",
    metadata={"layer": "product"},
)
def enriched_speeches(
    context: dg.AssetExecutionContext,
    speech_embeddings: tuple[pl.DataFrame, list[list[float]]],
    speech_summaries: pl.DataFrame,
    speech_classification: pl.DataFrame,
) -> pl.DataFrame:
    """Combine embeddings, summaries, and classification into final data product.

    This is the main data product that combines all enrichment:
    - Original speech text and metadata
    - GPT-OSS generated summaries (for synthetic training)
    - Multi-dimensional classification (monetary, trade, tariff, outlook)
    - is_governor from source data

    Args:
        context: Dagster execution context for logging.
        speech_embeddings: Tuple of (DataFrame, embeddings) from embedding step.
        speech_summaries: DataFrame with GPT-OSS generated summaries.
        speech_classification: DataFrame with multi-dimensional classifications.

    Returns:
        Combined DataFrame with all enrichment columns including summaries.
    """
    df_with_embeddings, _ = speech_embeddings
    df_with_classification = speech_classification

    # Join classification results with dead letter columns (renamed with _class suffix)
    classification_cols = [
        "reference",
        "monetary_stance",
        "trade_stance",
        "tariff_mention",
        "economic_outlook",
        "_llm_status",
        "_llm_error",
        "_llm_attempts",
        "_llm_fallback_used",
    ]
    df_classification_renamed = df_with_classification.select(
        [c for c in classification_cols if c in df_with_classification.columns]
    ).rename(
        {
            "_llm_status": "_llm_status_class",
            "_llm_error": "_llm_error_class",
            "_llm_attempts": "_llm_attempts_class",
            "_llm_fallback_used": "_llm_fallback_class",
        }
    )
    df = df_with_embeddings.join(
        df_classification_renamed,
        on="reference",
        how="left",
    )

    # Join summaries with dead letter columns (renamed with _summary suffix)
    summary_cols = [
        "reference",
        "summary",
        "_llm_status",
        "_llm_error",
        "_llm_attempts",
        "_llm_fallback_used",
    ]
    df_summaries_renamed = speech_summaries.select(
        [c for c in summary_cols if c in speech_summaries.columns]
    ).rename(
        {
            "_llm_status": "_llm_status_summary",
            "_llm_error": "_llm_error_summary",
            "_llm_attempts": "_llm_attempts_summary",
            "_llm_fallback_used": "_llm_fallback_summary",
        }
    )
    df = df.join(
        df_summaries_renamed,
        on="reference",
        how="left",
    )

    # Add processing timestamp
    df = df.with_columns(pl.lit(datetime.now(UTC).isoformat()).alias("processed_at"))

    context.log.info(f"Created enriched data product with {len(df)} speeches")
    context.log.info(f"Columns: {df.columns}")

    # Log summary coverage
    summary_count = df.filter(pl.col("summary").is_not_null()).height
    context.log.info(f"Speeches with summaries: {summary_count}/{len(df)}")

    # Log classification statistics
    context.log.info(
        f"Monetary stance distribution: {df['monetary_stance'].value_counts().sort('monetary_stance')}"
    )
    context.log.info(
        f"Trade stance distribution: {df['trade_stance'].value_counts().sort('trade_stance')}"
    )
    context.log.info(
        f"Economic outlook distribution: {df['economic_outlook'].value_counts().sort('economic_outlook')}"
    )
    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    context.log.info(f"Speeches mentioning tariffs: {tariff_count}/{len(df)}")
    governor_count = df.filter(pl.col("is_gov") == 1).height
    context.log.info(f"Governor speeches: {governor_count}/{len(df)}")

    return df


@dg.asset(
    description="Data product metadata and statistics",
    group_name="central_bank_speeches",
    metadata={
        "layer": "output",
    },
)
def speeches_data_product(
    context: dg.AssetExecutionContext,
    enriched_speeches: pl.DataFrame,
) -> dict[str, Any]:
    """Compute and return metadata about the final data product.

    The enriched_speeches asset is persisted to LakeFS via the IO manager.
    This asset computes statistics for observability.

    Args:
        context: Dagster execution context for logging.
        enriched_speeches: Final enriched DataFrame (persisted by IO manager).

    Returns:
        Dictionary with data product statistics.
    """
    df = enriched_speeches

    # Compute statistics
    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    governor_count = df.filter(pl.col("is_gov") == 1).height

    context.log.info(f"Data product: {len(df)} speeches")
    context.log.info(f"Tariff mentions: {tariff_count}")
    context.log.info(f"Governor speeches: {governor_count}")

    # Add output metadata for Dagster UI
    context.add_output_metadata(
        {
            "num_records": len(df),
            "tariff_mentions": tariff_count,
            "governor_speeches": governor_count,
        }
    )

    return {
        "num_records": len(df),
        "tariff_mentions": tariff_count,
        "governor_speeches": governor_count,
    }


@dg.asset(
    description="Vector search index in Weaviate",
    group_name="central_bank_speeches",
    metadata={
        "layer": "output",
        "destination": "weaviate",
    },
)
def weaviate_index(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    speech_embeddings: tuple[pl.DataFrame, list[list[float]]],
    speech_classification: pl.DataFrame,
    weaviate: WeaviateResource,
) -> dict[str, Any]:
    """Index speeches in Weaviate for vector search.

    Creates the CentralBankSpeeches collection and inserts all speeches
    with their embeddings for similarity search.
    Uses trial-specific collection when is_trial=True.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (is_trial for collection selection).
        speech_embeddings: Tuple of (DataFrame, embeddings) from embedding step.
        speech_classification: DataFrame with multi-dimensional classifications.
        weaviate: Weaviate resource for vector storage.

    Returns:
        Dictionary with indexing metadata (collection, count, dimensions).
    """
    df, embeddings = speech_embeddings

    # Join classifications
    df = df.join(
        speech_classification.select(
            ["reference", "monetary_stance", "trade_stance", "tariff_mention", "economic_outlook"]
        ),
        on="reference",
        how="left",
    )

    # Use trial collection if is_trial
    if config.is_trial:
        collection_name = "CentralBankSpeechesTrial"
        context.log.info("TRIAL RUN: Using trial-specific Weaviate collection")
    else:
        collection_name = "CentralBankSpeeches"

    # Ensure collection exists with proper exception handling
    try:
        weaviate.ensure_collection(
            name=collection_name,
            properties=SPEECHES_SCHEMA,
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

    # Prepare objects for insertion
    objects: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        objects.append(
            {
                "reference": row["reference"],
                "date": str(row.get("date", "")),
                "central_bank": row.get("central_bank", "Unknown"),
                "speaker": row.get("speaker", "Unknown"),
                "title": row.get("title", "Untitled"),
                "text": (row.get("text", "") or "")[:10000],  # Truncate for Weaviate
                "monetary_stance": int(row.get("monetary_stance", 3)),
                "trade_stance": int(row.get("trade_stance", 3)),
                "tariff_mention": bool(row.get("tariff_mention", 0)),
                "economic_outlook": int(row.get("economic_outlook", 3)),
                "is_governor": bool(row.get("is_governor", 0)),
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

    context.log.info(f"Indexed {count} speeches in Weaviate collection: {collection_name}")

    return {
        "collection": collection_name,
        "object_count": count,
        "vector_dimensions": len(embeddings[0]),
    }


# ============================================================================
# INTERMEDIATE SNAPSHOTS - Persist intermediate stages to LakeFS
# Export all central bank speech assets
central_bank_speeches_assets = [
    raw_speeches,
    cleaned_speeches,
    speech_embeddings,
    speech_classification,
    speech_summaries,
    enriched_speeches,
    speeches_data_product,
    weaviate_index,
]
