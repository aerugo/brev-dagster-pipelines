# Brev Dagster Pipelines

Dagster pipeline code for Brev Data Platform.

## Overview

This repository contains the Dagster assets, resources, and I/O managers for the Brev Data Platform.

## Structure

```
src/brev_pipelines/
├── definitions.py      # Main Dagster Definitions
├── assets/             # Dagster assets
│   ├── demo.py         # Demo pipeline assets
│   └── health.py       # Platform health checks
├── resources/          # External service resources
│   ├── minio.py        # MinIO S3 storage
│   ├── lakefs.py       # LakeFS versioning
│   └── nim.py          # NVIDIA NIM LLM
└── io_managers/        # Custom I/O managers
```

## Assets

### Demo Pipeline

| Asset | Description |
|-------|-------------|
| `raw_sample_data` | Generate 100 sample customer records |
| `cleaned_data` | Clean, normalize, add tier classification |
| `nim_enriched_data` | Enrich with AI profiles using NIM LLM |
| `data_summary` | Statistics stored to MinIO |

### Health Checks

| Asset | Description |
|-------|-------------|
| `platform_health` | Check MinIO, LakeFS, NIM connectivity |

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/ tests/

# Run tests
pytest tests/ -v

# Run Dagster locally
dagster dev -m brev_pipelines.definitions
```

## Docker

```bash
# Build image
docker build -t brev-dagster-pipelines .

# Run locally
docker run -p 4000:4000 \
  -e MINIO_ACCESS_KEY=admin \
  -e MINIO_SECRET_KEY=password \
  -e LAKEFS_ACCESS_KEY_ID=admin \
  -e LAKEFS_SECRET_ACCESS_KEY=password \
  brev-dagster-pipelines
```

## CI/CD

On push to main:
1. Runs linting and tests
2. Builds Docker image
3. Pushes to `ghcr.io/aerugo/brev-dagster-pipelines`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIO_ENDPOINT` | MinIO host:port | `minio.minio.svc.cluster.local:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | Required |
| `MINIO_SECRET_KEY` | MinIO secret key | Required |
| `LAKEFS_ENDPOINT` | LakeFS host:port | `lakefs.lakefs.svc.cluster.local:8000` |
| `LAKEFS_ACCESS_KEY_ID` | LakeFS access key | Required |
| `LAKEFS_SECRET_ACCESS_KEY` | LakeFS secret key | Required |
| `NIM_ENDPOINT` | NIM LLM endpoint URL | `http://nvidia-nim-llm.nvidia-nim.svc.cluster.local:8000` |

## License

Proprietary - Internal Use Only
