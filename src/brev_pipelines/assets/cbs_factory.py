"""Central Bank Speeches Asset Factory.

Generates separate asset sets for trial and production runs to prevent
trial data (10 rows) from overwriting production data (7000+ rows).

Each asset set has its own:
- Asset key prefix (trial/ vs no prefix)
- IO manager (lakefs_parquet_trial vs lakefs_parquet)
- LakeFS storage path (trial/ subdirectory)
- Weaviate collection (CentralBankSpeechesTrial vs CentralBankSpeeches)

Usage:
    # Production assets (no prefix)
    production_assets = build_cbs_assets(is_trial=False)

    # Trial assets (trial/ prefix)
    trial_assets = build_cbs_assets(is_trial=True)
"""

from datetime import UTC, datetime
from typing import Any, Sequence

import dagster as dg
import polars as pl

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
    SpeechClassification,
    WeaviatePropertyDef,
)


def _ensure_nim_reasoning_ready(
    context: dg.AssetExecutionContext,
    k8s_scaler: K8sScalerResource,
) -> None:
    """Ensure nim-reasoning is scaled up and ready before LLM calls."""
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


def build_cbs_assets(*, is_trial: bool) -> Sequence[dg.AssetsDefinition]:
    """Build central bank speeches assets for trial or production.

    Creates a complete asset graph with appropriate prefixes and IO managers
    to keep trial and production data completely separate.

    Args:
        is_trial: If True, creates trial assets with 'trial' prefix.

    Returns:
        List of asset definitions for the speeches pipeline.
    """
    # Configuration based on trial mode
    key_prefix: list[str] = ["trial"] if is_trial else []
    group_name = "cbs_trial" if is_trial else "central_bank_speeches"
    io_manager_key = "lakefs_parquet_trial" if is_trial else "lakefs_parquet"
    weaviate_collection = "CentralBankSpeechesTrial" if is_trial else "CentralBankSpeeches"
    sample_size = 10 if is_trial else 0  # 0 means no limit

    # Helper to build asset keys with correct prefix
    def _key(name: str) -> dg.AssetKey:
        """Build asset key with prefix."""
        if key_prefix:
            return dg.AssetKey([*key_prefix, name])
        return dg.AssetKey(name)

    @dg.asset(
        name="raw_speeches",
        key_prefix=key_prefix,
        description="Raw central bank speeches from Kaggle dataset",
        group_name=group_name,
        io_manager_key=io_manager_key,
        metadata={
            "layer": "raw",
            "source": "kaggle/davidgauthier/central-bank-speeches",
            "is_trial": str(is_trial),
        },
    )
    def raw_speeches(
        context: dg.AssetExecutionContext,
    ) -> pl.DataFrame:
        """Ingest central bank speeches dataset from Kaggle."""
        import os

        import kagglehub

        context.log.info("Downloading central-bank-speeches dataset from Kaggle...")

        dataset_path = kagglehub.dataset_download("davidgauthier/central-bank-speeches")
        context.log.info(f"Dataset downloaded to: {dataset_path}")

        files = os.listdir(dataset_path)
        csv_files = [f for f in files if f.endswith(".csv")]
        if not csv_files:
            msg = f"No CSV files found in dataset. Available files: {files}"
            raise ValueError(msg)

        csv_path = os.path.join(dataset_path, csv_files[0])
        df = pl.read_csv(csv_path)
        context.log.info(f"Loaded {len(df)} speeches from Kaggle")

        # Apply sample_size limit for trial runs
        if sample_size > 0:
            original_count = len(df)
            df = df.head(sample_size)
            context.log.info(
                f"TRIAL RUN: Limited to {sample_size} records (from {original_count})"
            )

        return df

    @dg.asset(
        name="cleaned_speeches",
        key_prefix=key_prefix,
        description="Cleaned speeches with null values filled",
        group_name=group_name,
        io_manager_key=io_manager_key,
        ins={"raw_speeches": dg.AssetIn(key=_key("raw_speeches"))},
        metadata={"layer": "cleaned", "is_trial": str(is_trial)},
    )
    def cleaned_speeches(
        context: dg.AssetExecutionContext,
        raw_speeches: pl.DataFrame,
    ) -> pl.DataFrame:
        """Clean the raw speeches data."""
        df = raw_speeches

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

        if fill_expressions:
            df = df.with_columns(fill_expressions)

        df = df.filter(pl.col("text").str.len_chars() > 100)
        context.log.info(f"Cleaned {len(df)} speeches")
        return df

    @dg.asset(
        name="speech_classification",
        key_prefix=key_prefix,
        description="Multi-dimensional speech classification using GPT-OSS 120B",
        group_name=group_name,
        io_manager_key=io_manager_key,
        ins={"cleaned_speeches": dg.AssetIn(key=_key("cleaned_speeches"))},
        metadata={
            "layer": "enriched",
            "uses_gpu": "true",
            "model": "GPT-OSS 120B",
            "is_trial": str(is_trial),
        },
    )
    def speech_classification(
        context: dg.AssetExecutionContext,
        cleaned_speeches: pl.DataFrame,
        nim_reasoning: NIMResource,
        minio: MinIOResource,
        k8s_scaler: K8sScalerResource,
    ) -> pl.DataFrame:
        """Classify speeches on multiple dimensions using GPT-OSS 120B."""
        _ensure_nim_reasoning_ready(context, k8s_scaler)

        df = cleaned_speeches

        checkpoint_mgr = LLMCheckpointManager(
            minio=minio,
            asset_name=f"{'trial_' if is_trial else ''}speech_classification",
            run_id=context.run_id,
            checkpoint_interval=10,
        )

        retry_config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            exponential_base=2.0,
        )

        def classify_speech(row: dict[str, Any]) -> dict[str, Any]:
            reference = str(row.get("reference", "unknown"))
            text_excerpt = (row.get("text", "") or "")[:4000]

            def get_fallback() -> SpeechClassification:
                return SpeechClassification(
                    monetary_stance=3,
                    trade_stance=3,
                    tariff_mention=0,
                    economic_outlook=3,
                )

            try:
                result = retry_classification(
                    nim_resource=nim_reasoning,
                    speech_text=text_excerpt,
                    record_id=reference,
                    config=retry_config,
                )

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
                    "_llm_error": result.error_message or "",
                    "_llm_attempts": result.attempts,
                    "_llm_fallback_used": result.fallback_used,
                }
            except Exception as e:
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

        checkpoint_mgr.cleanup()

        if results_df is None:
            msg = "Classification checkpoint returned no results"
            raise RuntimeError(msg)

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

        # Log statistics
        total = len(df)
        failed_count = df.filter(pl.col("_llm_status") == "failed").height
        success_count = total - failed_count
        success_rate = f"{100 * success_count / total:.1f}%" if total > 0 else "N/A"

        context.log.info(f"Classification complete: {success_count}/{total} ({success_rate})")

        return df

    @dg.asset(
        name="speech_summaries",
        key_prefix=key_prefix,
        description="Compact speech summaries for Safe Synthesizer training",
        group_name=group_name,
        io_manager_key=io_manager_key,
        ins={"cleaned_speeches": dg.AssetIn(key=_key("cleaned_speeches"))},
        metadata={
            "layer": "enriched",
            "uses_gpu": "true",
            "model": "GPT-OSS 120B",
            "is_trial": str(is_trial),
        },
    )
    def speech_summaries(
        context: dg.AssetExecutionContext,
        cleaned_speeches: pl.DataFrame,
        nim_reasoning: NIMResource,
        minio: MinIOResource,
        k8s_scaler: K8sScalerResource,
    ) -> pl.DataFrame:
        """Generate bullet-point summaries of central bank speeches."""
        _ensure_nim_reasoning_ready(context, k8s_scaler)

        df = cleaned_speeches

        checkpoint_mgr = LLMCheckpointManager(
            minio=minio,
            asset_name=f"{'trial_' if is_trial else ''}speech_summaries",
            run_id=context.run_id,
            checkpoint_interval=10,
        )

        retry_config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            exponential_base=2.0,
        )

        def summarize_speech(row: dict[str, Any]) -> dict[str, Any]:
            reference = str(row.get("reference", "unknown"))
            title = row.get("title", "") or "Untitled"
            speaker = row.get("speaker", "") or "Unknown"
            central_bank = row.get("central_bank", "") or "Unknown"
            text = row.get("text", "") or ""

            def get_fallback() -> str:
                return f"* Topic: {title[:100]}\n* Speaker: {speaker}\n* Bank: {central_bank}"

            try:
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

                summary = result.parsed_data if result.status == "success" else result.fallback_values
                if summary is None:
                    summary = get_fallback()

                return {
                    "reference": reference,
                    "summary": summary,
                    "_llm_status": result.status,
                    "_llm_error": result.error_message or "",
                    "_llm_attempts": result.attempts,
                    "_llm_fallback_used": result.fallback_used,
                }
            except Exception as e:
                return {
                    "reference": reference,
                    "summary": get_fallback(),
                    "_llm_status": "failed",
                    "_llm_error": f"Unexpected error: {e!s}",
                    "_llm_attempts": 0,
                    "_llm_fallback_used": True,
                }

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

        checkpoint_mgr.cleanup()

        if results_df is None:
            msg = "Summarization checkpoint returned no results"
            raise RuntimeError(msg)

        # Log statistics
        total = len(results_df)
        failed_count = results_df.filter(pl.col("_llm_status") == "failed").height
        success_count = total - failed_count
        success_rate = f"{100 * success_count / total:.1f}%" if total > 0 else "N/A"

        context.log.info(f"Summarization complete: {success_count}/{total} ({success_rate})")

        return results_df

    @dg.asset(
        name="speech_embeddings",
        key_prefix=key_prefix,
        description="Speeches with generated embeddings from local NIM",
        group_name=group_name,
        ins={
            "cleaned_speeches": dg.AssetIn(key=_key("cleaned_speeches")),
            "speech_summaries": dg.AssetIn(key=_key("speech_summaries")),
        },
        deps=[dg.AssetDep(_key("speech_classification"))],
        metadata={
            "layer": "enriched",
            "uses_nim_embedding": "true",
            "is_trial": str(is_trial),
        },
    )
    def speech_embeddings(
        context: dg.AssetExecutionContext,
        cleaned_speeches: pl.DataFrame,
        speech_summaries: pl.DataFrame,
        nim_embedding: NIMEmbeddingResource,
        minio: MinIOResource,
        k8s_scaler: K8sScalerResource,
    ) -> tuple[pl.DataFrame, list[list[float]]]:
        """Generate embeddings for all speeches using local NIM."""
        context.log.info(f"Received {len(speech_summaries)} summaries from IO manager")

        df = cleaned_speeches.join(
            speech_summaries.select(["reference", "summary"]),
            on="reference",
            how="left",
        )

        use_mock = nim_embedding.use_mock_fallback and not nim_embedding.health_check()

        if use_mock:
            context.log.info(f"Mock mode: generating {len(df)} embeddings without checkpointing")

            texts: list[str] = []
            for row in df.iter_rows(named=True):
                title = row.get("title", "") or ""
                summary = row.get("summary", "") or ""
                combined = f"{title}\n\n{summary}" if summary else title or "Untitled speech"
                texts.append(combined[:1500])

            embeddings = nim_embedding.embed_texts(texts, batch_size=32)
            context.log.info(
                f"Generated {len(embeddings)} mock embeddings, dimension: {len(embeddings[0])}"
            )

            return (df, embeddings)

        # Production mode
        context.log.info("Production mode: scaling down nim-reasoning for embedding")

        with k8s_scaler.temporarily_scale(
            deployment="nim-reasoning",
            namespace="nvidia-ai",
            replicas=0,
            restore_wait_ready=True,
        ):
            context.log.info("nim-reasoning scaled down, starting embedding generation")

            checkpoint_mgr = LLMCheckpointManager(
                minio=minio,
                asset_name=f"{'trial_' if is_trial else ''}speech_embeddings",
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

    @dg.asset(
        name="enriched_speeches",
        key_prefix=key_prefix,
        description="Combined enriched speeches data product with summaries",
        group_name=group_name,
        io_manager_key=io_manager_key,
        ins={
            "speech_embeddings": dg.AssetIn(key=_key("speech_embeddings")),
            "speech_summaries": dg.AssetIn(key=_key("speech_summaries")),
            "speech_classification": dg.AssetIn(key=_key("speech_classification")),
        },
        metadata={"layer": "product", "is_trial": str(is_trial)},
    )
    def enriched_speeches(
        context: dg.AssetExecutionContext,
        speech_embeddings: tuple[pl.DataFrame, list[list[float]]],
        speech_summaries: pl.DataFrame,
        speech_classification: pl.DataFrame,
    ) -> pl.DataFrame:
        """Combine embeddings, summaries, and classification into final data product."""
        df_with_embeddings, _ = speech_embeddings
        df_with_classification = speech_classification

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

        df = df.with_columns(pl.lit(datetime.now(UTC).isoformat()).alias("processed_at"))

        context.log.info(f"Created enriched data product with {len(df)} speeches")
        return df

    @dg.asset(
        name="speeches_data_product",
        key_prefix=key_prefix,
        description="Data product metadata and statistics",
        group_name=group_name,
        ins={"enriched_speeches": dg.AssetIn(key=_key("enriched_speeches"))},
        metadata={"layer": "output", "is_trial": str(is_trial)},
    )
    def speeches_data_product(
        context: dg.AssetExecutionContext,
        enriched_speeches: pl.DataFrame,
    ) -> dict[str, Any]:
        """Compute and return metadata about the final data product."""
        df = enriched_speeches

        tariff_count = df.filter(pl.col("tariff_mention") == 1).height
        governor_count = df.filter(pl.col("is_gov") == 1).height

        context.log.info(f"Data product: {len(df)} speeches")
        context.add_output_metadata(
            {
                "num_records": len(df),
                "tariff_mentions": tariff_count,
                "governor_speeches": governor_count,
                "is_trial": is_trial,
            }
        )

        return {
            "num_records": len(df),
            "tariff_mentions": tariff_count,
            "governor_speeches": governor_count,
        }

    @dg.asset(
        name="weaviate_index",
        key_prefix=key_prefix,
        description="Vector search index in Weaviate",
        group_name=group_name,
        ins={
            "speech_embeddings": dg.AssetIn(key=_key("speech_embeddings")),
            "speech_classification": dg.AssetIn(key=_key("speech_classification")),
        },
        metadata={
            "layer": "output",
            "destination": "weaviate",
            "is_trial": str(is_trial),
        },
    )
    def weaviate_index(
        context: dg.AssetExecutionContext,
        speech_embeddings: tuple[pl.DataFrame, list[list[float]]],
        speech_classification: pl.DataFrame,
        weaviate: WeaviateResource,
    ) -> dict[str, Any]:
        """Index speeches in Weaviate for vector search."""
        df, embeddings = speech_embeddings

        df = df.join(
            speech_classification.select(
                ["reference", "monetary_stance", "trade_stance", "tariff_mention", "economic_outlook"]
            ),
            on="reference",
            how="left",
        )

        context.log.info(f"Using Weaviate collection: {weaviate_collection}")

        try:
            weaviate.ensure_collection(
                name=weaviate_collection,
                properties=SPEECHES_SCHEMA,
                vector_dimensions=len(embeddings[0]),
            )
        except WeaviateConnectionError as e:
            raise RuntimeError(f"Cannot connect to Weaviate: {e}") from e
        except WeaviateCollectionError as e:
            raise RuntimeError(f"Failed to create Weaviate collection: {e}") from e

        objects: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            objects.append(
                {
                    "reference": row["reference"],
                    "date": str(row.get("date", "")),
                    "central_bank": row.get("central_bank", "Unknown"),
                    "speaker": row.get("speaker", "Unknown"),
                    "title": row.get("title", "Untitled"),
                    "text": (row.get("text", "") or "")[:10000],
                    "monetary_stance": int(row.get("monetary_stance", 3)),
                    "trade_stance": int(row.get("trade_stance", 3)),
                    "tariff_mention": bool(row.get("tariff_mention", 0)),
                    "economic_outlook": int(row.get("economic_outlook", 3)),
                    "is_governor": bool(row.get("is_governor", 0)),
                }
            )

        try:
            count = weaviate.insert_objects(
                collection_name=weaviate_collection,
                objects=objects,
                vectors=embeddings,
            )
        except WeaviateConnectionError as e:
            raise RuntimeError(f"Cannot connect to Weaviate: {e}") from e
        except WeaviateCollectionError as e:
            raise RuntimeError(f"Failed to insert into Weaviate: {e}") from e

        context.log.info(f"Indexed {count} speeches in Weaviate collection: {weaviate_collection}")

        return {
            "collection": weaviate_collection,
            "object_count": count,
            "vector_dimensions": len(embeddings[0]),
        }

    return [
        raw_speeches,
        cleaned_speeches,
        speech_classification,
        speech_summaries,
        speech_embeddings,
        enriched_speeches,
        speeches_data_product,
        weaviate_index,
    ]


# Pre-built asset sets for convenience
production_cbs_assets = build_cbs_assets(is_trial=False)
trial_cbs_assets = build_cbs_assets(is_trial=True)
