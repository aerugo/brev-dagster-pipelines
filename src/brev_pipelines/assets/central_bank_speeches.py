"""Central Bank Speeches ETL Pipeline.

This pipeline demonstrates end-to-end AI data product development:
1. Ingest dataset from Kaggle
2. Version data in LakeFS
3. Generate embeddings via local NIM embedding model
4. Multi-dimensional classification via GPT-OSS (monetary, trade, outlook, tariffs)
5. Generate compact summaries via GPT-OSS (for synthetic data training)
6. Store enriched data product in LakeFS
7. Index text and embeddings in Weaviate for vector search

All AI inference uses local NIM endpoints - no external API dependencies.

Data enrichment for synthetic data pipeline:
- Numeric classifications (1-5 scales): Capture overall sentiment/stance
- Compact summaries (~1000 chars): Capture specific metrics, regions, sectors,
  timelines, risks, and policy tools not in numeric classifications
- Both fit within Safe Synthesizer's context window for faithful reproduction

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
from brev_pipelines.io_managers.checkpoint import LLMCheckpointManager, process_with_checkpoint
from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMResource
from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource
from brev_pipelines.resources.weaviate import WeaviateResource

# Collection schema for Weaviate
SPEECHES_SCHEMA: list[dict[str, str]] = [
    {"name": "reference", "type": "text", "description": "Unique identifier from source"},
    {"name": "date", "type": "text", "description": "Speech date (ISO format)"},
    {"name": "central_bank", "type": "text", "description": "Issuing institution"},
    {"name": "speaker", "type": "text", "description": "Speaker name"},
    {"name": "title", "type": "text", "description": "Speech title"},
    {"name": "text", "type": "text", "description": "Full speech text"},
    {"name": "monetary_stance", "type": "int", "description": "1=very_dovish to 5=very_hawkish"},
    {"name": "trade_stance", "type": "int", "description": "1=very_protectionist to 5=very_globalist"},
    {"name": "tariff_mention", "type": "boolean", "description": "Contains tariff/protectionist discussion"},
    {"name": "economic_outlook", "type": "int", "description": "1=very_negative to 5=very_positive"},
    {"name": "is_governor", "type": "boolean", "description": "Speaker is governor/president/chair"},
]


@dg.asset(
    description="Raw central bank speeches from Kaggle dataset",
    group_name="central_bank_speeches",
    metadata={
        "layer": "raw",
        "source": "kaggle/davidgauthier/central-bank-speeches",
        "destination": "lakefs",
    },
)
def raw_speeches(
    context: dg.AssetExecutionContext,
    config: PipelineConfig,
    lakefs: LakeFSResource,
) -> pl.DataFrame:
    """Ingest central bank speeches dataset from Kaggle.

    Downloads the dataset using KaggleHub and stores raw data in LakeFS
    for version control of source material.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (sample_size for trial runs).
        lakefs: LakeFS resource for versioned data storage.

    Returns:
        Raw speeches DataFrame from Kaggle.
    """
    import os

    import kagglehub
    from lakefs_sdk.models import CommitCreation

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

    # Serialize to Parquet
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    # Get LakeFS client
    lakefs_client = lakefs.get_client()

    # Upload raw data to LakeFS
    path = "central-bank-speeches/raw_speeches.parquet"
    lakefs_client.objects_api.upload_object(
        repository="data",
        branch="main",
        path=path,
        content=parquet_bytes,
    )

    # Create commit for versioned raw data (skip if no changes)
    try:
        commit = lakefs_client.commits_api.commit(
            repository="data",
            branch="main",
            commit_creation=CommitCreation(
                message=f"Ingest raw central bank speeches ({len(df)} records)",
                metadata={
                    "dagster_run_id": context.run_id or "",
                    "source": "kaggle/davidgauthier/central-bank-speeches",
                    "num_records": str(len(df)),
                    "sample_size": str(config.sample_size) if config.sample_size > 0 else "full",
                },
            ),
        )
        context.log.info(f"LakeFS commit: {commit.id}")
    except Exception as e:
        if "no changes" in str(e).lower():
            context.log.info("No changes to commit (data already exists in LakeFS)")
        else:
            raise

    context.log.info(f"Stored raw data to LakeFS: lakefs://data/main/{path}")

    # Log column info
    context.log.info(f"Columns: {df.columns}")
    context.log.info(f"Schema: {df.schema}")

    return df


@dg.asset(
    description="Cleaned speeches with null values filled",
    group_name="central_bank_speeches",
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
    fill_expressions = []
    for col_name in df.columns:
        dtype = df.schema[col_name]
        if dtype == pl.Utf8 or dtype == pl.String:
            fill_expressions.append(pl.col(col_name).fill_null(""))
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            fill_expressions.append(pl.col(col_name).fill_null(0))
        elif dtype in (pl.Float32, pl.Float64):
            fill_expressions.append(pl.col(col_name).fill_null(0.0))
        elif dtype == pl.Boolean:
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

    Returns:
        DataFrame with classification columns added.
    """
    df = cleaned_speeches

    # Create checkpoint manager
    checkpoint_mgr = LLMCheckpointManager(
        minio=minio,
        asset_name="speech_classification",
        run_id=context.run_id,
        checkpoint_interval=10,
    )

    def classify_speech(row: dict) -> dict:
        """Classify a single speech and return result dict."""
        reference = row["reference"]
        text_excerpt = (row.get("text", "") or "")[:4000]
        title = row.get("title", "") or "Untitled"
        speaker = row.get("speaker", "") or "Unknown"
        central_bank = row.get("central_bank", "") or "Unknown"

        prompt = f"""You are an expert analyst of central bank communications. Classify the following speech on multiple dimensions.

{CLASSIFICATION_FEW_SHOT_EXAMPLES}

Now classify this speech:

Title: {title}
Speaker: {speaker}
Central Bank: {central_bank}

Speech excerpt:
{text_excerpt}

Respond with ONLY a JSON object in this exact format:
{{"monetary_stance": "very_dovish|somewhat_dovish|neutral|somewhat_hawkish|very_hawkish", "trade_stance": "very_protectionist|somewhat_protectionist|neutral|somewhat_globalist|very_globalist", "tariff_mention": 0 or 1, "economic_outlook": "very_negative|somewhat_negative|neutral|somewhat_positive|very_positive"}}

Classification:"""

        response = nim_reasoning.generate(prompt, max_tokens=150, temperature=0.1)

        # Parse response with fallback to neutral values
        monetary, trade, tariff, outlook = 3, 3, 0, 3
        try:
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                result = json.loads(json_match.group())
                monetary = MONETARY_STANCE_SCALE.get(result.get("monetary_stance", "neutral"), 3)
                trade = TRADE_STANCE_SCALE.get(result.get("trade_stance", "neutral"), 3)
                tariff = int(result.get("tariff_mention", 0))
                outlook = OUTLOOK_SCALE.get(result.get("economic_outlook", "neutral"), 3)
        except Exception:
            pass  # Keep defaults

        return {
            "reference": reference,
            "monetary_stance": monetary,
            "trade_stance": trade,
            "tariff_mention": tariff,
            "economic_outlook": outlook,
        }

    # Process with checkpointing
    context.log.info(f"Starting classification of {len(df)} speeches with checkpointing")
    results_df = process_with_checkpoint(
        df=df,
        id_column="reference",
        process_fn=classify_speech,
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,
        logger=context.log,
    )

    # Clean up checkpoint on success
    checkpoint_mgr.cleanup()

    # Join results back to original DataFrame
    df = df.join(
        results_df.select([
            "reference",
            pl.col("monetary_stance").cast(pl.Int8),
            pl.col("trade_stance").cast(pl.Int8),
            pl.col("tariff_mention").cast(pl.Int8),
            pl.col("economic_outlook").cast(pl.Int8),
        ]),
        on="reference",
        how="left",
    )

    # Log classification statistics
    context.log.info(f"Monetary stance distribution: {df['monetary_stance'].value_counts()}")
    context.log.info(f"Trade stance distribution: {df['trade_stance'].value_counts()}")
    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    context.log.info(f"Speeches mentioning tariffs: {tariff_count}/{len(df)}")
    context.log.info(f"Economic outlook distribution: {df['economic_outlook'].value_counts()}")

    return df


@dg.asset(
    description="Compact speech summaries for Safe Synthesizer training",
    group_name="central_bank_speeches",
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
) -> pl.DataFrame:
    """Generate compact, structured summaries for Safe Synthesizer training.

    These summaries are designed to:
    1. Fit within Safe Synthesizer's context window (~2000 chars target)
    2. Capture semantic nuance NOT in numeric classifications
    3. Use bullet-point format for easier TinyLlama reproduction

    Uses checkpointing to save progress every 10 rows, allowing recovery
    from failures without reprocessing completed summaries.

    What numeric classifications capture:
    - monetary_stance (1-5): Overall hawkish/dovish direction
    - trade_stance (1-5): Protectionist vs globalist leaning
    - economic_outlook (1-5): Positive/negative sentiment
    - tariff_mention (0/1): Whether tariffs are discussed

    What summaries capture (the nuance):
    - SPECIFIC METRICS: Exact numbers cited (inflation %, GDP growth, unemployment)
    - GEOGRAPHIC FOCUS: Which regions/countries mentioned
    - SECTOR COMMENTARY: Which industries discussed (housing, energy, finance)
    - FORWARD GUIDANCE: Timeline language (next quarter, 2024, medium-term)
    - RISK FACTORS: Specific risks mentioned (supply chain, geopolitical, banking)
    - POLICY TOOLS: Which instruments mentioned (rates, QE, reserves, guidance)

    Target length: 800-1200 chars (fits in Safe Synthesizer with other metadata)

    Args:
        context: Dagster execution context for logging.
        cleaned_speeches: Cleaned DataFrame with speech text.
        nim_reasoning: NIM reasoning resource for summary generation (GPT-OSS 120B).
        minio: MinIO resource for checkpoint storage.

    Returns:
        DataFrame with reference and summary columns.
    """
    df = cleaned_speeches

    # Create checkpoint manager
    checkpoint_mgr = LLMCheckpointManager(
        minio=minio,
        asset_name="speech_summaries",
        run_id=context.run_id,
        checkpoint_interval=10,
    )

    def summarize_speech(row: dict) -> dict:
        """Generate summary for a single speech and return result dict."""
        reference = row["reference"]
        title = row.get("title", "") or "Untitled"
        speaker = row.get("speaker", "") or "Unknown"
        central_bank = row.get("central_bank", "") or "Unknown"
        text = row.get("text", "") or ""
        text_excerpt = text[:10000]

        prompt = f"""Extract key details from this central bank speech into a COMPACT bullet-point summary.

IMPORTANT: Keep total output under 1000 characters. Use terse, information-dense bullet points.

Focus on SPECIFIC DETAILS not captured by general sentiment scores:

• METRICS: Exact numbers (inflation %, GDP growth, unemployment rate, rate changes)
• REGIONS: Countries/regions specifically discussed
• SECTORS: Industries mentioned (housing, energy, labor, banking, trade)
• TIMELINE: Forward guidance timeframes (next meeting, Q2 2024, medium-term)
• RISKS: Specific concerns (supply chain, geopolitical, financial stability)
• TOOLS: Policy instruments discussed (rates, QE, reserves, forward guidance)

Speech: {title}
Speaker: {speaker} ({central_bank})

Text excerpt:
{text_excerpt}

Generate a COMPACT bullet-point summary (under 1000 characters total):"""

        summary = nim_reasoning.generate(prompt, max_tokens=400, temperature=0.2)

        # Handle LLM errors with fallback
        if summary.startswith("LLM error:"):
            summary = f"• Topic: {title[:100]}\n• Speaker: {speaker}\n• Bank: {central_bank}"

        # Truncate if too long
        if len(summary) > 1500:
            summary = summary[:1500] + "..."

        return {"reference": reference, "summary": summary}

    # Process with checkpointing
    context.log.info(f"Starting summarization of {len(df)} speeches with checkpointing")
    results_df = process_with_checkpoint(
        df=df,
        id_column="reference",
        process_fn=summarize_speech,
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,
        logger=context.log,
    )

    # Clean up checkpoint on success
    checkpoint_mgr.cleanup()

    # Log summary statistics
    summaries = results_df["summary"].to_list()
    summary_lengths = [len(s) for s in summaries]
    avg_len = sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0
    max_len = max(summary_lengths) if summary_lengths else 0
    min_len = min(summary_lengths) if summary_lengths else 0
    context.log.info(
        f"Generated {len(summaries)} compact summaries: "
        f"avg={avg_len:.0f} chars, min={min_len}, max={max_len}"
    )

    return results_df


@dg.asset(
    description="Combined enriched speeches data product with summaries",
    group_name="central_bank_speeches",
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

    # Join classification results (monetary_stance, trade_stance, tariff_mention, economic_outlook)
    df = df_with_embeddings.join(
        df_with_classification.select(
            ["reference", "monetary_stance", "trade_stance", "tariff_mention", "economic_outlook"]
        ),
        on="reference",
        how="left",
    )

    # Join summaries
    df = df.join(
        speech_summaries.select(["reference", "summary"]),
        on="reference",
        how="left",
    )

    # Add processing timestamp
    df = df.with_columns(
        pl.lit(datetime.now(timezone.utc).isoformat()).alias("processed_at")
    )

    context.log.info(f"Created enriched data product with {len(df)} speeches")
    context.log.info(f"Columns: {df.columns}")

    # Log summary coverage
    summary_count = df.filter(pl.col("summary").is_not_null()).height
    context.log.info(f"Speeches with summaries: {summary_count}/{len(df)}")

    # Log classification statistics
    context.log.info(f"Monetary stance distribution: {df['monetary_stance'].value_counts().sort('monetary_stance')}")
    context.log.info(f"Trade stance distribution: {df['trade_stance'].value_counts().sort('trade_stance')}")
    context.log.info(f"Economic outlook distribution: {df['economic_outlook'].value_counts().sort('economic_outlook')}")
    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    context.log.info(f"Speeches mentioning tariffs: {tariff_count}/{len(df)}")
    governor_count = df.filter(pl.col("is_gov") == 1).height
    context.log.info(f"Governor speeches: {governor_count}/{len(df)}")

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
    config: PipelineConfig,
    enriched_speeches: pl.DataFrame,
    lakefs: LakeFSResource,
) -> dict[str, Any]:
    """Store final data product in LakeFS with versioning.

    Creates a versioned Parquet file in LakeFS for downstream consumption.
    Uses trial-specific path when is_trial=True to keep trial data separate.

    Args:
        context: Dagster execution context for logging.
        config: Pipeline configuration (is_trial for path selection).
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

    # Upload to LakeFS - use trial path if is_trial
    if config.is_trial:
        path = "central-bank-speeches/trial/speeches.parquet"
        context.log.info("TRIAL RUN: Using trial-specific LakeFS path")
    else:
        path = "central-bank-speeches/speeches.parquet"
    lakefs_client.objects_api.upload_object(
        repository="data",
        branch="main",
        path=path,
        content=parquet_bytes,
    )

    # Create commit (skip if no changes)
    tariff_count = df.filter(pl.col("tariff_mention") == 1).height
    commit_id = None
    try:
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
        commit_id = commit.id
        context.log.info(f"Committed to LakeFS: {commit_id}")
    except Exception as e:
        if "no changes" in str(e).lower():
            context.log.info("No changes to commit (data already exists in LakeFS)")
        else:
            raise

    return {
        "path": f"lakefs://data/main/{path}",
        "commit_id": commit_id,
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

    # Ensure collection exists
    weaviate.ensure_collection(
        name=collection_name,
        properties=SPEECHES_SCHEMA,
        vector_dimensions=len(embeddings[0]),
    )

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
    count = weaviate.insert_objects(
        collection_name=collection_name,
        objects=objects,
        vectors=embeddings,
    )

    context.log.info(f"Indexed {count} speeches in Weaviate collection: {collection_name}")

    return {
        "collection": collection_name,
        "object_count": count,
        "vector_dimensions": len(embeddings[0]),
    }


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
