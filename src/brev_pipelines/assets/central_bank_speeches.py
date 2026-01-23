"""Central Bank Speeches ETL Pipeline.

This pipeline demonstrates end-to-end AI data product development:
1. Ingest dataset from Kaggle
2. Version data in LakeFS
3. Generate embeddings via local NIM embedding model
4. Classify tariff mentions via NIM LLM
5. Store enriched data product in LakeFS
6. Index text and embeddings in Weaviate for vector search

All AI inference uses local NIM endpoints - no external API dependencies.

Trial Run Mode:
    Run with sample_size config to test with limited records:
    - speeches_trial_run job: 10 records
    - Or use run config: {"ops": {"raw_speeches": {"config": {"sample_size": 10}}}}
"""

import io
import json
import re
from datetime import datetime, timezone
from typing import Any

import dagster as dg
import polars as pl

from brev_pipelines.config import PipelineConfig
from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMResource
from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource
from brev_pipelines.resources.weaviate import WeaviateResource

# Collection schema for Weaviate
SPEECHES_SCHEMA: list[dict[str, str]] = [
    {"name": "speech_id", "type": "text", "description": "Unique identifier"},
    {"name": "date", "type": "text", "description": "Speech date (ISO format)"},
    {"name": "central_bank", "type": "text", "description": "Issuing institution"},
    {"name": "speaker", "type": "text", "description": "Speaker name"},
    {"name": "title", "type": "text", "description": "Speech title"},
    {"name": "text", "type": "text", "description": "Full speech text"},
    {"name": "tariff_mention", "type": "boolean", "description": "Contains tariff discussion"},
]


@dg.asset(
    description="Raw central bank speeches from Kaggle dataset",
    group_name="central_bank_speeches",
    metadata={
        "layer": "raw",
        "source": "kaggle/davidgauthier/central-bank-speeches",
    },
)
def raw_speeches(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    minio: MinIOResource,
) -> pl.DataFrame:
    """Ingest central bank speeches dataset from Kaggle.

    Downloads the dataset using KaggleHub and stores raw data in MinIO.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (sample_size for trial runs).
        minio: MinIO resource for raw data storage.

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

    # Store raw data in MinIO
    minio.ensure_bucket("raw-data")
    client = minio.get_client()

    # Save as Parquet to MinIO
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    client.put_object(
        "raw-data",
        "central-bank-speeches/raw_speeches.parquet",
        io.BytesIO(parquet_bytes),
        len(parquet_bytes),
        content_type="application/octet-stream",
    )

    context.log.info(
        "Stored raw data to MinIO: raw-data/central-bank-speeches/raw_speeches.parquet"
    )

    # Log column info
    context.log.info(f"Columns: {df.columns}")
    context.log.info(f"Schema: {df.schema}")

    return df


@dg.asset(
    description="Cleaned and normalized speeches with unique IDs",
    group_name="central_bank_speeches",
    metadata={"layer": "cleaned"},
)
def cleaned_speeches(
    context: dg.AssetExecutionContext,
    raw_speeches: pl.DataFrame,
) -> pl.DataFrame:
    """Clean and normalize the raw speeches data.

    Performs the following transformations:
    - Add unique speech_id
    - Normalize column names
    - Parse dates
    - Handle missing values
    - Filter out empty speeches

    Args:
        context: Dagster execution context for logging.
        raw_speeches: Raw DataFrame from Kaggle.

    Returns:
        Cleaned and normalized DataFrame.

    Raises:
        ValueError: If dataset lacks required text column.
    """
    df = raw_speeches

    # Map common column variations to standard names
    column_mapping = {
        "date": "date",
        "Date": "date",
        "speech_date": "date",
        "central_bank": "central_bank",
        "institution": "central_bank",
        "bank": "central_bank",
        "speaker": "speaker",
        "speaker_name": "speaker",
        "title": "title",
        "speech_title": "title",
        "text": "text",
        "content": "text",
        "speech": "text",
        "speech_text": "text",
    }

    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and old_name != new_name:
            df = df.rename({old_name: new_name})

    # Generate unique IDs
    df = df.with_row_index("_row_idx")
    df = df.with_columns(
        (pl.lit("SPEECH-") + pl.col("_row_idx").cast(pl.Utf8).str.zfill(6)).alias("speech_id")
    )
    df = df.drop("_row_idx")

    # Ensure required columns exist (with defaults)
    if "date" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("date"))
    if "central_bank" not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias("central_bank"))
    if "speaker" not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias("speaker"))
    if "title" not in df.columns:
        df = df.with_columns(pl.lit("Untitled").alias("title"))
    if "text" not in df.columns:
        msg = "Dataset must have a 'text' or 'content' column"
        raise ValueError(msg)

    # Fill nulls
    df = df.with_columns(
        [
            pl.col("central_bank").fill_null("Unknown"),
            pl.col("speaker").fill_null("Unknown"),
            pl.col("title").fill_null("Untitled"),
            pl.col("text").fill_null(""),
        ]
    )

    # Select and order columns
    df = df.select(
        [
            "speech_id",
            "date",
            "central_bank",
            "speaker",
            "title",
            "text",
        ]
    )

    # Filter out empty speeches (less than 100 chars)
    df = df.filter(pl.col("text").str.len_chars() > 100)

    context.log.info(f"Cleaned {len(df)} speeches")
    unique_banks = df["central_bank"].unique().to_list()[:10]
    context.log.info(f"Central banks (sample): {unique_banks}")

    return df


@dg.asset(
    description="Speeches with generated embeddings from local NIM",
    group_name="central_bank_speeches",
    metadata={
        "layer": "enriched",
        "uses_nim_embedding": "true",
    },
)
def speech_embeddings(
    context: dg.AssetExecutionContext,
    cleaned_speeches: pl.DataFrame,
    nim_embedding: NIMEmbeddingResource,
) -> tuple[pl.DataFrame, list[list[float]]]:
    """Generate embeddings for all speeches using local NIM.

    Uses llama-3_2-nemoretriever-300m-embed-v2 model (1024 dimensions).
    Returns tuple of (DataFrame, embeddings) for downstream storage.

    Args:
        context: Dagster execution context for logging.
        cleaned_speeches: Cleaned DataFrame with speech text.
        nim_embedding: NIM embedding resource for vector generation.

    Returns:
        Tuple of (DataFrame, list of 1024-dim embedding vectors).
    """
    df = cleaned_speeches

    # Prepare texts for embedding (use title + text excerpt)
    texts: list[str] = []
    for row in df.iter_rows(named=True):
        # Combine title and first 2000 chars of text for embedding
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        combined = f"{title}\n\n{text[:2000]}"
        texts.append(combined)

    context.log.info(f"Generating embeddings for {len(texts)} speeches...")

    # Generate embeddings in batches
    embeddings = nim_embedding.embed_texts(texts, batch_size=32)

    context.log.info(
        f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}"
    )

    return (df, embeddings)


@dg.asset(
    description="Speeches classified for tariff mentions using NIM LLM",
    group_name="central_bank_speeches",
    metadata={
        "layer": "enriched",
        "uses_gpu": "true",
    },
)
def tariff_classification(
    context: dg.AssetExecutionContext,
    cleaned_speeches: pl.DataFrame,
    nim: NIMResource,
) -> pl.DataFrame:
    """Classify speeches for tariff mentions using NIM LLM.

    Uses NIM to analyze each speech and determine if it discusses tariffs,
    trade barriers, customs duties, or related trade policy topics.

    Args:
        context: Dagster execution context for logging.
        cleaned_speeches: Cleaned DataFrame with speech text.
        nim: NIM LLM resource for text classification.

    Returns:
        DataFrame with tariff_mention and tariff_confidence columns.
    """
    df = cleaned_speeches

    tariff_mentions: list[int] = []
    tariff_confidences: list[float] = []

    # Process in batches to manage GPU memory
    batch_size = 10
    total = len(df)

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch = df.slice(i, batch_end - i)

        for row in batch.iter_rows(named=True):
            # Take first 3000 chars for classification
            text_excerpt = (row.get("text", "") or "")[:3000]
            title = row.get("title", "") or "Untitled"

            prompt = f"""Analyze this central bank speech excerpt and determine if it discusses tariffs, trade barriers, customs duties, import/export restrictions, or trade policy.

Title: {title}

Excerpt:
{text_excerpt}

Respond with ONLY a JSON object in this exact format:
{{"tariff_mention": 0 or 1, "confidence": 0.0 to 1.0}}

Where tariff_mention is 1 if the speech discusses tariffs/trade barriers, 0 otherwise."""

            response = nim.generate(prompt, max_tokens=50, temperature=0.1)

            # Parse response
            try:
                # Try to extract JSON from response
                json_match = re.search(r"\{[^}]+\}", response)
                if json_match:
                    result = json.loads(json_match.group())
                    tariff_mentions.append(int(result.get("tariff_mention", 0)))
                    tariff_confidences.append(float(result.get("confidence", 0.5)))
                else:
                    # Default to 0 if parsing fails
                    tariff_mentions.append(0)
                    tariff_confidences.append(0.0)
            except Exception as e:
                context.log.warning(f"Failed to parse LLM response: {e}")
                tariff_mentions.append(0)
                tariff_confidences.append(0.0)

        context.log.info(f"Processed {batch_end}/{total} speeches")

    # Add classification columns
    df = df.with_columns(
        [
            pl.Series("tariff_mention", tariff_mentions).cast(pl.Int8),
            pl.Series("tariff_confidence", tariff_confidences).cast(pl.Float64),
        ]
    )

    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    context.log.info(f"Found {tariff_count}/{len(df)} speeches mentioning tariffs")

    return df


@dg.asset(
    description="Combined enriched speeches data product",
    group_name="central_bank_speeches",
    metadata={"layer": "product"},
)
def enriched_speeches(
    context: dg.AssetExecutionContext,
    speech_embeddings: tuple[pl.DataFrame, list[list[float]]],
    tariff_classification: pl.DataFrame,
) -> pl.DataFrame:
    """Combine embeddings and classification into final data product.

    This is the main data product that combines all enrichment.

    Args:
        context: Dagster execution context for logging.
        speech_embeddings: Tuple of (DataFrame, embeddings) from embedding step.
        tariff_classification: DataFrame with tariff classification results.

    Returns:
        Combined DataFrame with all enrichment columns.
    """
    df_with_embeddings, _ = speech_embeddings
    df_with_classification = tariff_classification

    # Join classification results
    df = df_with_embeddings.join(
        df_with_classification.select(
            ["speech_id", "tariff_mention", "tariff_confidence"]
        ),
        on="speech_id",
        how="left",
    )

    # Add processing timestamp
    df = df.with_columns(
        pl.lit(datetime.now(timezone.utc).isoformat()).alias("processed_at")
    )

    context.log.info(f"Created enriched data product with {len(df)} speeches")
    context.log.info(f"Columns: {df.columns}")

    return df


@dg.asset(
    description="Versioned data product stored in LakeFS",
    group_name="central_bank_speeches",
    metadata={
        "layer": "output",
        "destination": "lakefs",
    },
)
def speeches_data_product(
    context: dg.AssetExecutionContext,
    enriched_speeches: pl.DataFrame,
    lakefs: LakeFSResource,
) -> dict[str, Any]:
    """Store final data product in LakeFS with versioning.

    Creates a versioned Parquet file in LakeFS for downstream consumption.

    Args:
        context: Dagster execution context for logging.
        enriched_speeches: Final enriched DataFrame.
        lakefs: LakeFS resource for data versioning.

    Returns:
        Dictionary with storage metadata (path, commit_id, counts).
    """
    from lakefs_sdk.models import CommitCreation

    df = enriched_speeches

    # Serialize to Parquet
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    # Get LakeFS client
    lakefs_client = lakefs.get_client()

    # Upload to LakeFS
    path = "central-bank-speeches/speeches.parquet"
    lakefs_client.objects_api.upload_object(
        repository="data",
        branch="main",
        path=path,
        content=parquet_bytes,
    )

    # Create commit
    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    commit = lakefs_client.commits_api.commit(
        repository="data",
        branch="main",
        commit_creation=CommitCreation(
            message=f"Update central bank speeches data product ({len(df)} records)",
            metadata={
                "dagster_run_id": context.run_id or "",
                "num_records": str(len(df)),
                "tariff_mentions": str(tariff_count),
            },
        ),
    )

    context.log.info(f"Committed to LakeFS: {commit.id}")

    return {
        "path": f"lakefs://data/main/{path}",
        "commit_id": commit.id,
        "num_records": len(df),
        "tariff_mentions": tariff_count,
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
    speech_embeddings: tuple[pl.DataFrame, list[list[float]]],
    tariff_classification: pl.DataFrame,
    weaviate: WeaviateResource,
) -> dict[str, Any]:
    """Index speeches in Weaviate for vector search.

    Creates the CentralBankSpeeches collection and inserts all speeches
    with their embeddings for similarity search.

    Args:
        context: Dagster execution context for logging.
        speech_embeddings: Tuple of (DataFrame, embeddings) from embedding step.
        tariff_classification: DataFrame with tariff classification results.
        weaviate: Weaviate resource for vector storage.

    Returns:
        Dictionary with indexing metadata (collection, count, dimensions).
    """
    df, embeddings = speech_embeddings

    # Join classification to include tariff_mention
    df = df.join(
        tariff_classification.select(["speech_id", "tariff_mention"]),
        on="speech_id",
        how="left",
    )

    # Ensure collection exists
    weaviate.ensure_collection(
        name="CentralBankSpeeches",
        properties=SPEECHES_SCHEMA,
        vector_dimensions=len(embeddings[0]),
    )

    # Prepare objects for insertion
    objects: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        objects.append(
            {
                "speech_id": row["speech_id"],
                "date": str(row.get("date", "")),
                "central_bank": row.get("central_bank", "Unknown"),
                "speaker": row.get("speaker", "Unknown"),
                "title": row.get("title", "Untitled"),
                "text": (row.get("text", "") or "")[:10000],  # Truncate for Weaviate
                "tariff_mention": bool(row.get("tariff_mention", 0)),
            }
        )

    # Insert objects with embeddings
    count = weaviate.insert_objects(
        collection_name="CentralBankSpeeches",
        objects=objects,
        vectors=embeddings,
    )

    context.log.info(f"Indexed {count} speeches in Weaviate")

    return {
        "collection": "CentralBankSpeeches",
        "object_count": count,
        "vector_dimensions": len(embeddings[0]),
    }


# Export all central bank speech assets
central_bank_speeches_assets = [
    raw_speeches,
    cleaned_speeches,
    speech_embeddings,
    tariff_classification,
    enriched_speeches,
    speeches_data_product,
    weaviate_index,
]
