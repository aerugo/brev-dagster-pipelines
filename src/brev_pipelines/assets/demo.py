"""Demo pipeline assets for Brev Data Platform."""

import io
import json
import random
from typing import Any

import dagster as dg
import pandas as pd

from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMError, NIMResource


@dg.asset(
    description="Raw sample data for demo pipeline",
    group_name="demo",
    metadata={"layer": "raw"},
)
def raw_sample_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Generate sample customer data."""
    random.seed(42)

    regions = ["North", "South", "East", "West", "Central"]
    categories = ["Premium", "Standard", "Basic"]

    data: list[dict[str, Any]] = []
    for i in range(100):
        data.append(
            {
                "id": f"CUST-{i:04d}",
                "name": f"Customer {i}",
                "age": random.randint(18, 75),
                "region": random.choice(regions),
                "category": random.choice(categories),
                "spend": round(random.uniform(100, 10000), 2),
                "active": random.random() > 0.2,
            }
        )

    df = pd.DataFrame(data)
    context.log.info(f"Generated {len(df)} sample records")
    return df


@dg.asset(
    description="Cleaned and validated data",
    group_name="demo",
    metadata={"layer": "cleaned"},
)
def cleaned_data(
    context: dg.AssetExecutionContext,
    raw_sample_data: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and validate sample data."""
    df = raw_sample_data.copy()

    # Normalize region names
    df["region"] = df["region"].str.title()

    # Cap extreme spend values
    df["spend"] = df["spend"].clip(upper=9000)

    # Add tier classification
    df["tier"] = pd.cut(
        df["spend"],
        bins=[0, 1000, 5000, float("inf")],
        labels=["Low Value", "Medium Value", "High Value"],
    )

    context.log.info(f"Cleaned {len(df)} records")
    return df


@dg.asset(
    description="Data enriched with NIM LLM descriptions",
    group_name="demo",
    metadata={"layer": "enriched", "uses_gpu": "true"},
)
def nim_enriched_data(
    context: dg.AssetExecutionContext,
    cleaned_data: pd.DataFrame,
    nim: NIMResource,
) -> pd.DataFrame:
    """Enrich data with AI-generated profiles using NIM LLM.

    Handles NIM errors gracefully by recording error message in profile field.
    """
    df = cleaned_data.copy()

    # Only enrich a sample to save time/cost
    sample_size = min(10, len(df))
    sample_indices = df.sample(sample_size, random_state=42).index

    profiles: list[str] = []
    error_count = 0
    for idx, row in df.iterrows():
        if idx in sample_indices:
            prompt = f"""Generate a brief customer profile (1 sentence) for:
- Age: {row["age"]}, Region: {row["region"]}, Category: {row["category"]}, Tier: {row["tier"]}
Be concise."""
            context.log.info(f"Calling NIM for {row['id']}...")
            try:
                profile = nim.generate(prompt, max_tokens=50)
                profiles.append(profile)
            except NIMError as e:
                error_count += 1
                context.log.warning(f"NIM error for {row['id']}: {e}")
                profiles.append(f"Error: {e}")
        else:
            profiles.append("Not enriched (sample limit)")

    df["ai_profile"] = profiles
    enriched_count = sum(
        1 for p in profiles if not p.startswith("Error:") and "Not enriched" not in p
    )
    context.log.info(f"Enriched {enriched_count}/{len(df)} records ({error_count} errors)")
    return df


@dg.asset(
    description="Summary statistics stored in MinIO",
    group_name="demo",
    metadata={"layer": "output", "destination": "minio"},
)
def data_summary(
    context: dg.AssetExecutionContext,
    nim_enriched_data: pd.DataFrame,
    minio: MinIOResource,
) -> dict[str, Any]:
    """Generate summary and store in MinIO.

    Handles MinIO errors gracefully by including error info in output.
    """
    df = nim_enriched_data

    summary: dict[str, Any] = {
        "total_records": len(df),
        "region_distribution": df["region"].value_counts().to_dict(),
        "tier_distribution": df["tier"].value_counts().to_dict(),
        "average_spend": round(df["spend"].mean(), 2),
        "enriched_count": sum(
            1 for p in df["ai_profile"] if not p.startswith("Error:") and "Not enriched" not in p
        ),
        "storage_error": None,
    }

    # Store in MinIO with error handling
    bucket = "data-products"
    try:
        minio.ensure_bucket(bucket)
        client = minio.get_client()
        data = json.dumps(summary, indent=2).encode()

        client.put_object(
            bucket,
            "demo/summary.json",
            io.BytesIO(data),
            len(data),
            content_type="application/json",
        )
        context.log.info(f"Stored summary to {bucket}/demo/summary.json")
    except Exception as e:
        summary["storage_error"] = str(e)
        context.log.warning(f"Failed to store summary in MinIO: {e}")

    return summary


# Export all demo assets
demo_assets = [
    raw_sample_data,
    cleaned_data,
    nim_enriched_data,
    data_summary,
]
