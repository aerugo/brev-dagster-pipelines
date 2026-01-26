# Brev Dagster Pipelines

Dagster pipeline code for Brev Data Platform.

## Overview

This repository contains the Dagster assets, resources, jobs, and configurations for the Brev Data Platform. The main pipeline is the **Central Bank Speeches** AI data product that demonstrates end-to-end ML data processing with vector search and synthetic data generation.

## Table of Contents

- [Structure](#structure)
- [Assets](#assets)
- [Jobs](#jobs)
- [Local Development](#local-development)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Running Tests](#running-tests)
  - [Running Dagster UI Locally](#running-dagster-ui-locally)
  - [Configuration Options](#configuration-options)
  - [Trial Runs vs Full Runs](#trial-runs-vs-full-runs)
- [Architecture](#architecture)
  - [Pipeline Flow](#pipeline-flow)
  - [Error Handling & Retry Patterns](#error-handling--retry-patterns)
  - [Checkpointing](#checkpointing)
- [Deployment](#deployment-to-kubernetes)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

## Structure

```
src/brev_pipelines/
├── definitions.py          # Main Dagster Definitions
├── config.py               # Pipeline configuration (sample_size for trial runs)
├── jobs.py                 # Pre-configured jobs (trial runs, full runs)
├── types.py                # TypedDict definitions for type safety
├── assets/
│   ├── central_bank_speeches.py   # Main AI data pipeline (ETL + enrichment)
│   ├── synthetic_speeches.py      # Synthetic data pipeline
│   ├── demo.py                    # Demo pipeline assets
│   ├── health.py                  # Platform health checks
│   └── validation.py              # Validation utilities
├── resources/
│   ├── minio.py            # MinIO S3 storage
│   ├── lakefs.py           # LakeFS data versioning
│   ├── nim.py              # NVIDIA NIM LLM
│   ├── nim_embedding.py    # NVIDIA NIM Embedding model
│   ├── weaviate.py         # Weaviate vector database
│   ├── safe_synth.py       # NVIDIA Safe Synthesizer
│   ├── llm_retry.py        # LLM retry wrapper with backoff
│   └── safe_synth_retry.py # Safe Synthesizer retry wrapper
└── io_managers/            # Custom I/O managers
    ├── checkpoint.py       # LLM checkpointing for resumable processing
    ├── minio_polars.py     # MinIO Polars DataFrame I/O
    ├── lakefs_polars.py    # LakeFS Polars DataFrame I/O
    └── weaviate_io.py      # Weaviate vector I/O
```

## Assets

### Central Bank Speeches Pipeline

| Asset | Description | Layer |
|-------|-------------|-------|
| `raw_speeches` | Ingest from Kaggle, store in MinIO | raw |
| `cleaned_speeches` | Normalize, add IDs, filter empty | cleaned |
| `speech_classification` | Classify monetary/trade stance via NIM LLM | enriched |
| `speech_summaries` | Generate summaries via NIM LLM | enriched |
| `speech_embeddings` | Generate embeddings via NIM Embedding | enriched |
| `enriched_speeches` | Combined data product | product |
| `speeches_data_product` | Version in LakeFS | output |
| `weaviate_index` | Index in Weaviate for vector search | output |

### Synthetic Data Pipeline

| Asset | Description | Layer |
|-------|-------------|-------|
| `enriched_data_for_synthesis` | Load enriched data from LakeFS | input |
| `safe_synth_model` | Train & generate via NVIDIA Safe Synthesizer | synthetic |
| `synthetic_validation_report` | Privacy validation (MIA/AIA scores) | validation |
| `synthetic_embeddings` | Generate embeddings for synthetic data | enriched |
| `synthetic_data_product` | Version synthetic data in LakeFS | output |
| `synthetic_weaviate_index` | Index in Weaviate (separate collection) | output |

### Utility Assets

| Asset | Description |
|-------|-------------|
| `validate_platform` | Check MinIO, LakeFS, NIM, Weaviate connectivity |
| `nim_health` | NIM LLM health check with error handling |
| `raw_sample_data` | Demo: Generate 100 sample records |
| `cleaned_data` | Demo: Clean and normalize |
| `nim_enriched_data` | Demo: Enrich with AI profiles |
| `data_summary` | Demo: Summary statistics stored in MinIO |

## Jobs

### Trial Runs (10 records)

For testing the pipeline end-to-end with minimal data:

| Job | Description |
|-----|-------------|
| `speeches_trial_run` | Real speeches pipeline with 10 records |
| `synthetic_trial_run` | Synthetic generation with 10 records |
| `full_pipeline_trial_run` | Complete pipeline (real + synthetic) with 10 records |

### Full Runs (all records)

| Job | Description |
|-----|-------------|
| `speeches_full_run` | Process all speeches |
| `synthetic_full_run` | Generate synthetic for all speeches |
| `full_pipeline_full_run` | Complete pipeline (real + synthetic) |

---

## Local Development

### Prerequisites

- **Python 3.11+** (required)
- **uv** (recommended) or pip for dependency management
- **Docker** (optional, for running services locally)
- **kubectl** (optional, for port-forwarding to cluster services)

### Quick Start

```bash
cd dagster

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Verify installation
uv run python -c "from brev_pipelines.definitions import defs; print('✓ Dagster definitions loaded')"
```

### Running Tests

The test suite uses comprehensive mocks for all external services (MinIO, LakeFS, NIM, Weaviate, Kubernetes). No external services are required to run tests.

```bash
# Run all tests (520 tests)
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=brev_pipelines --cov-report=term-missing

# Run specific test categories
uv run pytest tests/unit/ -v                    # Unit tests only
uv run pytest tests/unit/assets/ -v             # Asset tests
uv run pytest tests/unit/resources/ -v          # Resource tests
uv run pytest tests/unit/io_managers/ -v        # I/O manager tests

# Run tests for specific functionality
uv run pytest tests/ -v -k "retry"              # Retry logic tests
uv run pytest tests/ -v -k "checkpoint"         # Checkpointing tests
uv run pytest tests/ -v -k "classification"     # Classification tests

# Run linting and type checking
uv run ruff check src/ tests/                   # Linting
uv run ruff format --check src/ tests/          # Format check
uv run mypy src/brev_pipelines/ --strict        # Type checking
```

### Running Dagster UI Locally

There are three ways to run Dagster locally, depending on your needs:

#### Option 1: Mock Mode (No External Services)

For UI exploration and basic testing without any external services:

```bash
cd dagster

# Start Dagster with mock environment
# (Assets will fail if materialized, but UI works)
uv run dagster dev -m brev_pipelines.definitions
```

Access the UI at http://localhost:3000 to explore:
- Asset graph visualization
- Job definitions
- Resource configurations
- Schedule/sensor definitions

#### Option 2: Port-Forward to Cluster Services

If you have access to the Kubernetes cluster:

```bash
# Terminal 1: Port forward all services
kubectl port-forward svc/minio -n minio 9000:9000 &
kubectl port-forward svc/lakefs -n lakefs 8000:8000 &
kubectl port-forward svc/nim-llm -n nvidia-ai 8001:8000 &
kubectl port-forward svc/nvidia-nim-embedding -n nvidia-nim 8002:8000 &
kubectl port-forward svc/weaviate -n weaviate 8080:80 &
kubectl port-forward svc/weaviate-grpc -n weaviate 50051:50051 &

# Terminal 2: Set environment and start Dagster
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=admin
export MINIO_SECRET_KEY=password
export LAKEFS_ENDPOINT=localhost:8000
export LAKEFS_ACCESS_KEY_ID=admin
export LAKEFS_SECRET_ACCESS_KEY=password
export NIM_ENDPOINT=http://localhost:8001
export NIM_EMBEDDING_ENDPOINT=http://localhost:8002
export WEAVIATE_HOST=localhost
export WEAVIATE_PORT=8080
export WEAVIATE_GRPC_HOST=localhost
export WEAVIATE_GRPC_PORT=50051

cd dagster
uv run dagster dev -m brev_pipelines.definitions
```

#### Option 3: Docker Compose (Local Services)

For a fully local development environment:

```bash
# Create docker-compose.yml for local services
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password
    command: server /data --console-address ":9001"

  lakefs:
    image: treeverse/lakefs:latest
    ports:
      - "8000:8000"
    environment:
      LAKEFS_DATABASE_TYPE: local
      LAKEFS_AUTH_ENCRYPT_SECRET_KEY: some-secret-key
      LAKEFS_BLOCKSTORE_TYPE: local
      LAKEFS_BLOCKSTORE_LOCAL_PATH: /data
    depends_on:
      - minio

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 100
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
      ENABLE_MODULES: ""
EOF

# Start services
docker-compose up -d

# Set environment for Dagster
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=admin
export MINIO_SECRET_KEY=password
export LAKEFS_ENDPOINT=localhost:8000
export LAKEFS_ACCESS_KEY_ID=admin
export LAKEFS_SECRET_ACCESS_KEY=password
export WEAVIATE_HOST=localhost
export WEAVIATE_PORT=8080
export WEAVIATE_GRPC_HOST=localhost
export WEAVIATE_GRPC_PORT=50051

# Start Dagster (NIM calls will fail without GPU)
cd dagster
uv run dagster dev -m brev_pipelines.definitions
```

**Note:** NIM LLM and Safe Synthesizer require NVIDIA GPUs and are not easily run locally. Use port-forwarding to cluster services for full pipeline testing.

### Configuration Options

#### PipelineConfig

All configurable assets accept a `PipelineConfig` with these options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sample_size` | int | 0 | Max records to process (0 = no limit) |
| `is_trial` | bool | False | Use trial-specific collections/paths |

#### Preset Configurations

```python
from brev_pipelines.config import (
    TRIAL_RUN_CONFIG,   # {"sample_size": 10, "is_trial": True}
    SMALL_RUN_CONFIG,   # {"sample_size": 100}
    MEDIUM_RUN_CONFIG,  # {"sample_size": 1000}
)
```

#### Custom Configuration via CLI

```bash
# Run with custom sample size
dagster job launch -j speeches_full_run \
  --config-json '{"ops": {"raw_speeches": {"config": {"sample_size": 50}}}}'

# Run trial on specific asset
dagster asset materialize -m brev_pipelines.definitions \
  --select raw_speeches \
  --config-json '{"ops": {"raw_speeches": {"config": {"sample_size": 20, "is_trial": true}}}}'
```

#### Custom Configuration via UI

1. Go to **Jobs** → Select a job
2. Click **Launchpad**
3. In the YAML config editor, add:

```yaml
ops:
  raw_speeches:
    config:
      sample_size: 25
      is_trial: true
```

### Trial Runs vs Full Runs

| Aspect | Trial Run | Full Run |
|--------|-----------|----------|
| Records | 10 | All (~5000+) |
| LakeFS Path | `trial/speeches/` | `speeches/` |
| Weaviate Collection | `TrialSpeeches` | `Speeches` |
| GPU Time | ~2-5 minutes | ~30-60 minutes |
| Use Case | Testing, development | Production |

**Important:** Trial runs write to separate paths/collections, so they never affect production data.

---

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ETL Pipeline (speeches_*)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────────────────────┐│
│  │  Kaggle  │───▶│ raw_speeches│───▶│         cleaned_speeches             ││
│  │   API    │    │   (MinIO)   │    │    (normalize, filter, add IDs)      ││
│  └──────────┘    └─────────────┘    └──────────────────────────────────────┘│
│                                                  │                           │
│                    ┌─────────────────────────────┼─────────────────────────┐ │
│                    │                             │                         │ │
│                    ▼                             ▼                         ▼ │
│         ┌──────────────────┐        ┌──────────────────┐      ┌───────────┐ │
│         │ speech_summaries │        │speech_classific. │      │speech_emb.│ │
│         │   (NIM LLM)      │        │   (NIM LLM)      │      │(NIM Embed)│ │
│         └────────┬─────────┘        └────────┬─────────┘      └─────┬─────┘ │
│                  │                           │                      │       │
│                  └───────────────┬───────────┴──────────────────────┘       │
│                                  ▼                                          │
│                       ┌───────────────────┐                                 │
│                       │ enriched_speeches │                                 │
│                       │   (join all)      │                                 │
│                       └─────────┬─────────┘                                 │
│                                 │                                           │
│              ┌──────────────────┼──────────────────┐                        │
│              ▼                  ▼                  ▼                        │
│     ┌────────────────┐ ┌───────────────┐ ┌────────────────┐                │
│     │speeches_data_  │ │ weaviate_idx  │ │  snapshots     │                │
│     │product (LakeFS)│ │ (vector DB)   │ │  (MinIO)       │                │
│     └────────────────┘ └───────────────┘ └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Synthesis Pipeline (synthetic_*)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐    ┌─────────────────┐    ┌───────────────────────┐ │
│  │enriched_data_for_  │───▶│ safe_synth_model│───▶│synthetic_validation_  │ │
│  │synthesis (LakeFS)  │    │(Safe Synthesizer│    │report (privacy check) │ │
│  └────────────────────┘    └────────┬────────┘    └───────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│                          ┌─────────────────────┐                            │
│                          │ synthetic_embeddings│                            │
│                          │    (NIM Embed)      │                            │
│                          └──────────┬──────────┘                            │
│                                     │                                        │
│              ┌──────────────────────┼──────────────────┐                    │
│              ▼                      ▼                  ▼                    │
│     ┌────────────────┐    ┌────────────────┐ ┌────────────────┐            │
│     │synthetic_data_ │    │synthetic_weav_ │ │synthetic_snap_ │            │
│     │product (LakeFS)│    │idx (vector DB) │ │shots (MinIO)   │            │
│     └────────────────┘    └────────────────┘ └────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Error Handling & Retry Patterns

All LLM and external service calls use retry wrappers with exponential backoff:

#### LLM Retry (NIM)

```python
from brev_pipelines.resources.llm_retry import (
    retry_llm_call,
    LLMRetryConfig,
)

# Automatic retry with exponential backoff
result = retry_llm_call(
    lambda: nim.generate(prompt, max_tokens=100),
    config=LLMRetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        jitter=True,
    ),
    logger=context.log,
)
```

**Retry behavior:**
- Retries on: `NIMTimeoutError`, `NIMServerError`, `NIMRateLimitError`
- Does NOT retry on: `NIMError` (client errors), `ValueError`
- Backoff: 1s → 2s → 4s (with jitter)

#### Safe Synthesizer Retry

```python
from brev_pipelines.resources.safe_synth_retry import (
    retry_safe_synth_call,
    SafeSynthRetryConfig,
)

# Longer delays for GPU-intensive jobs
result = retry_safe_synth_call(
    lambda: safe_synth.synthesize(data, run_id, config),
    run_id=run_id,
    config=SafeSynthRetryConfig(
        max_retries=3,
        initial_delay=30.0,   # Safe Synth jobs are slow
        max_delay=300.0,      # Max 5 minutes between retries
    ),
    logger=context.log,
)
```

#### Dead Letter Columns

All LLM output includes dead letter columns for observability:

| Column | Type | Description |
|--------|------|-------------|
| `_llm_status` | str | `"success"` or `"failed"` |
| `_llm_error` | str\|None | Error message if failed |
| `_llm_attempts` | int | Number of attempts made |
| `_llm_fallback_used` | bool | Whether fallback value was used |

### Checkpointing

Long-running LLM operations use checkpointing for resumability:

```python
from brev_pipelines.io_managers.checkpoint import (
    LLMCheckpointManager,
    process_with_checkpoint,
)

# Checkpoint saves every 10 records to MinIO
checkpoint = LLMCheckpointManager(
    minio=minio,
    asset_name="speech_classification",
    run_id=context.run_id,
    checkpoint_interval=10,
)

# Process with automatic checkpoint/resume
result = process_with_checkpoint(
    df=input_df,
    id_column="reference",
    process_fn=classify_row,
    checkpoint_manager=checkpoint,
    batch_size=10,
    logger=context.log,
)
```

If a run fails, re-running will:
1. Load existing checkpoint from MinIO
2. Skip already-processed records
3. Continue from where it left off

---

## Deployment to Kubernetes

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dagster Namespace                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Webserver  │  │   Daemon    │  │  User Deployments       │  │
│  │   (UI)      │  │ (schedules) │  │  (brev-pipelines)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                           │                                      │
│                    ┌──────┴──────┐                              │
│                    │  PostgreSQL │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### Build and Push Docker Image

```bash
cd dagster

# Build image
docker build -t ghcr.io/aerugo/brev-data-platform/dagster:latest .

# Push to registry
docker push ghcr.io/aerugo/brev-data-platform/dagster:latest
```

### Configure Secrets

```bash
cd k8s/apps/dagster/secrets

# Create secrets file
cat > secrets.yaml << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: dagster-env-secrets
  namespace: dagster
type: Opaque
stringData:
  MINIO_ACCESS_KEY: "your-minio-key"
  MINIO_SECRET_KEY: "your-minio-secret"
  LAKEFS_ACCESS_KEY_ID: "your-lakefs-key"
  LAKEFS_SECRET_ACCESS_KEY: "your-lakefs-secret"
  KAGGLE_USERNAME: "your-kaggle-user"
  KAGGLE_KEY: "your-kaggle-key"
EOF

# Encrypt with SOPS
sops --encrypt secrets.yaml > secrets.enc.yaml
rm secrets.yaml
```

### Deploy with ArgoCD

```bash
# Sync the application
argocd app sync dagster

# Check status
argocd app get dagster
```

### Verify Deployment

```bash
# Check pods
kubectl get pods -n dagster

# View logs
kubectl logs -n dagster -l app=dagster-webserver

# Port forward to access UI
kubectl port-forward svc/dagster-webserver -n dagster 3000:80
```

### Run a Pipeline

From the Dagster UI:
1. Go to **Jobs** in the left sidebar
2. Select `speeches_trial_run` for testing
3. Click **Launch Run**
4. Monitor progress in the **Runs** tab

Or from CLI:

```bash
kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
  dagster job launch -j speeches_trial_run
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIO_ENDPOINT` | MinIO host:port | `minio.minio.svc.cluster.local:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | Required |
| `MINIO_SECRET_KEY` | MinIO secret key | Required |
| `LAKEFS_ENDPOINT` | LakeFS host:port | `lakefs.lakefs.svc.cluster.local:8000` |
| `LAKEFS_ACCESS_KEY_ID` | LakeFS access key | Required |
| `LAKEFS_SECRET_ACCESS_KEY` | LakeFS secret key | Required |
| `NIM_ENDPOINT` | NIM LLM endpoint | `http://nim-llm.nvidia-ai.svc.cluster.local:8000` |
| `NIM_REASONING_ENDPOINT` | NIM reasoning model | `http://nim-reasoning.nvidia-ai.svc.cluster.local:8000` |
| `NIM_EMBEDDING_ENDPOINT` | NIM Embedding endpoint | `http://nvidia-nim-embedding.nvidia-nim.svc.cluster.local:8000` |
| `WEAVIATE_HOST` | Weaviate hostname | `weaviate.weaviate.svc.cluster.local` |
| `WEAVIATE_PORT` | Weaviate HTTP port | `80` |
| `WEAVIATE_GRPC_HOST` | Weaviate gRPC hostname | `weaviate-grpc.weaviate.svc.cluster.local` |
| `WEAVIATE_GRPC_PORT` | Weaviate gRPC port | `50051` |
| `SAFE_SYNTH_NAMESPACE` | Safe Synth K8s namespace | `nvidia-ai` |
| `SAFE_SYNTH_ENDPOINT` | Safe Synth service URL | `http://nemo-safe-synthesizer.nvidia-ai.svc.cluster.local:8000` |
| `SAFE_SYNTH_PRIORITY` | K8s priority class | `batch-high` |
| `KAGGLE_USERNAME` | Kaggle API username | Required for speeches pipeline |
| `KAGGLE_KEY` | Kaggle API key | Required for speeches pipeline |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `NIMError: Connection refused` | NIM service not available | Check NIM pod status, port-forward if local |
| `NIMTimeoutError` | GPU overloaded or slow | Increase timeout, check GPU memory |
| `NIMRateLimitError` | Too many requests | Retry automatically handles this |
| `LakeFSConnectionError` | LakeFS service down | Check LakeFS pod, verify endpoint |
| `LakeFSNotFoundError` | File doesn't exist | Ensure upstream asset ran successfully |
| `WeaviateConnectionError` | Weaviate service down | Check Weaviate pod, verify host/port |
| `FileNotFoundError` (MinIO) | Object doesn't exist | Check bucket and path, run upstream asset |
| `SafeSynthJobFailedError` | Synthesis job failed | Check job logs, may need more data |

### Debugging Commands

```bash
# Check Dagster pod logs
kubectl logs -n dagster -l app=dagster-user-deployments --tail=100

# Check service connectivity from pod
kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
  curl -s http://minio.minio.svc.cluster.local:9000/minio/health/live

# Check NIM health
kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
  curl -s http://nim-llm.nvidia-ai.svc.cluster.local:8000/v1/health/ready

# Check Weaviate
kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
  curl -s http://weaviate.weaviate.svc.cluster.local/v1/.well-known/ready

# View Dagster run logs
kubectl logs -n dagster -l dagster/run_id=<run-id>

# Check events for errors
kubectl get events -n dagster --sort-by='.lastTimestamp' | tail -20
```

### Viewing Asset Metadata

After a run completes, check the **Asset Details** page in Dagster UI for:
- `success_rate`: Percentage of successful LLM calls
- `total_processed`: Number of records processed
- `failed_references`: List of failed record IDs (up to 100)
- `average_attempts`: Average retry attempts per record
- `fallback_count`: Number of records using fallback values

### Resuming Failed Runs

If a run fails mid-way:

1. **Check checkpoint exists:**
   ```bash
   # List checkpoint files
   kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
     mc ls myminio/dagster-checkpoints/checkpoints/
   ```

2. **Re-run the same job** - it will automatically resume from checkpoint

3. **Force fresh start** (delete checkpoint):
   ```bash
   kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
     mc rm myminio/dagster-checkpoints/checkpoints/<asset_name>/<run_id>.parquet
   ```

---

## License

Proprietary - Internal Use Only
