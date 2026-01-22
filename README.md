# Brev Dagster Pipelines

Dagster pipeline code for Brev Data Platform.

## Overview

This repository contains the Dagster assets, resources, jobs, and configurations for the Brev Data Platform. The main pipeline is the **Central Bank Speeches** AI data product that demonstrates end-to-end ML data processing with vector search and synthetic data generation.

## Structure

```
src/brev_pipelines/
├── definitions.py          # Main Dagster Definitions
├── config.py               # Pipeline configuration (sample_size for trial runs)
├── jobs.py                 # Pre-configured jobs (trial runs, full runs)
├── assets/
│   ├── central_bank_speeches.py   # Main AI data pipeline (Phases 1-3)
│   ├── synthetic_speeches.py      # Synthetic data pipeline (Phase 4)
│   ├── demo.py                    # Demo pipeline assets
│   ├── health.py                  # Platform health checks
│   └── validation.py              # Validation utilities
├── resources/
│   ├── minio.py            # MinIO S3 storage
│   ├── lakefs.py           # LakeFS data versioning
│   ├── nim.py              # NVIDIA NIM LLM
│   ├── nim_embedding.py    # NVIDIA NIM Embedding model
│   ├── weaviate.py         # Weaviate vector database
│   └── safe_synth.py       # NVIDIA Safe Synthesizer
└── io_managers/            # Custom I/O managers
```

## Assets

### Central Bank Speeches Pipeline

| Asset | Description | Layer |
|-------|-------------|-------|
| `raw_speeches` | Ingest from Kaggle, store in MinIO | raw |
| `cleaned_speeches` | Normalize, add IDs, filter empty | cleaned |
| `speech_embeddings` | Generate embeddings via NIM | enriched |
| `tariff_classification` | Classify tariff mentions via NIM LLM | enriched |
| `enriched_speeches` | Combined data product | product |
| `speeches_data_product` | Version in LakeFS | output |
| `weaviate_index` | Index in Weaviate for vector search | output |

### Synthetic Data Pipeline

| Asset | Description | Layer |
|-------|-------------|-------|
| `synthetic_speeches` | Generate via NVIDIA Safe Synthesizer | synthetic |
| `synthetic_validation_report` | Privacy validation (MIA/AIA scores) | validation |
| `synthetic_embeddings` | Generate embeddings for synthetic data | enriched |
| `synthetic_data_product` | Version synthetic data in LakeFS | output |
| `synthetic_weaviate_index` | Index in Weaviate (separate collection) | output |

### Utility Assets

| Asset | Description |
|-------|-------------|
| `platform_health` | Check MinIO, LakeFS, NIM connectivity |
| `raw_sample_data` | Demo: Generate 100 sample records |
| `cleaned_data` | Demo: Clean and normalize |
| `nim_enriched_data` | Demo: Enrich with AI profiles |

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

## Local Development

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Setup

```bash
cd dagster

# Install dependencies with uv
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Run Dagster UI Locally

```bash
# Set environment variables for local services
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
export WEAVIATE_GRPC_PORT=50051

# Port forward cluster services (if needed)
kubectl port-forward svc/minio -n minio 9000:9000 &
kubectl port-forward svc/lakefs -n lakefs 8000:8000 &
kubectl port-forward svc/nvidia-nim-llm -n nvidia-nim 8001:8000 &
kubectl port-forward svc/nvidia-nim-embedding -n nvidia-nim 8002:8000 &
kubectl port-forward svc/weaviate -n weaviate 8080:8080 &

# Start Dagster development server
dagster dev -m brev_pipelines.definitions
```

Access the UI at http://localhost:3000

### Running Jobs

```bash
# Trial run (10 records) - recommended for testing
dagster job launch -j speeches_trial_run

# Full pipeline trial (real + synthetic, 10 records)
dagster job launch -j full_pipeline_trial_run

# Custom sample size
dagster job launch -j speeches_full_run \
  --config-json '{"ops": {"raw_speeches": {"config": {"sample_size": 50}}}}'

# Full run (all records)
dagster job launch -j speeches_full_run
```

### Testing

```bash
# Run linting
ruff check src/ tests/

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=brev_pipelines
```

## Deployment to Kubernetes

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dagster Namespace                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Webserver  │  │   Daemon    │  │  User Deployments   │  │
│  │   (UI)      │  │ (schedules) │  │  (brev-pipelines)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                           │                                  │
│                    ┌──────┴──────┐                          │
│                    │  PostgreSQL │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: Build and Push Docker Image

The image is built automatically on push to `main` when files in `dagster/` change.

**Manual build:**

```bash
cd dagster

# Build image
docker build -t ghcr.io/aerugo/brev-data-platform/dagster:latest .

# Push to registry
docker push ghcr.io/aerugo/brev-data-platform/dagster:latest
```

### Step 2: Configure Secrets

Create the secrets file with SOPS encryption:

```bash
cd k8s/apps/dagster/secrets

# Create unencrypted secrets
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

### Step 3: Deploy with ArgoCD

The Dagster application is managed by ArgoCD. Ensure the ArgoCD Application exists:

```yaml
# k8s/apps/argocd-apps/templates/dagster.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: dagster
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/aerugo/brev-data-platform.git
    targetRevision: HEAD
    path: k8s/apps/dagster
  destination:
    server: https://kubernetes.default.svc
    namespace: dagster
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

**Manual sync:**

```bash
# Sync the application
argocd app sync dagster

# Check status
argocd app get dagster
```

### Step 4: Verify Deployment

```bash
# Check pods
kubectl get pods -n dagster

# Check services
kubectl get svc -n dagster

# View logs
kubectl logs -n dagster -l app=dagster-webserver

# Port forward to access UI
kubectl port-forward svc/dagster-webserver -n dagster 3000:80
```

Access the UI at http://localhost:3000

### Step 5: Run a Pipeline

From the Dagster UI:
1. Go to **Jobs** in the left sidebar
2. Select `speeches_trial_run` for testing
3. Click **Launch Run**
4. Monitor progress in the **Runs** tab

Or from CLI:

```bash
# Exec into the user deployment pod
kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
  dagster job launch -j speeches_trial_run
```

## CI/CD

### GitHub Actions Workflow

On push to `main` with changes in `dagster/`:

1. **Build** - Docker image built with buildx
2. **Push** - Image pushed to `ghcr.io/aerugo/brev-data-platform/dagster`
3. **Tags** - `latest`, SHA, and run number

### Image Tags

| Tag | Description |
|-----|-------------|
| `latest` | Most recent build from main |
| `<sha>` | Git commit SHA |
| `<run>` | GitHub Actions run number |

### Triggering Deployment

After image push, ArgoCD automatically syncs if configured with `selfHeal: true`. Otherwise:

```bash
# Manual sync
argocd app sync dagster

# Or restart the deployment to pull new image
kubectl rollout restart deployment/dagster-user-deployments-brev-pipelines -n dagster
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIO_ENDPOINT` | MinIO host:port | `minio.minio.svc.cluster.local:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | Required |
| `MINIO_SECRET_KEY` | MinIO secret key | Required |
| `LAKEFS_ENDPOINT` | LakeFS host:port | `lakefs.lakefs.svc.cluster.local:8000` |
| `LAKEFS_ACCESS_KEY_ID` | LakeFS access key | Required |
| `LAKEFS_SECRET_ACCESS_KEY` | LakeFS secret key | Required |
| `NIM_ENDPOINT` | NIM LLM endpoint | `http://nvidia-nim-llm.nvidia-nim.svc.cluster.local:8000` |
| `NIM_EMBEDDING_ENDPOINT` | NIM Embedding endpoint | `http://nvidia-nim-embedding.nvidia-nim.svc.cluster.local:8000` |
| `WEAVIATE_HOST` | Weaviate hostname | `weaviate.weaviate.svc.cluster.local` |
| `WEAVIATE_PORT` | Weaviate HTTP port | `8080` |
| `WEAVIATE_GRPC_PORT` | Weaviate gRPC port | `50051` |
| `SAFE_SYNTH_NAMESPACE` | Safe Synth K8s namespace | `nvidia-ai` |
| `SAFE_SYNTH_ENDPOINT` | Safe Synth service URL | `http://nvidia-safe-synth.nvidia-ai.svc.cluster.local:8080` |
| `KAGGLE_USERNAME` | Kaggle API username | Required for speeches pipeline |
| `KAGGLE_KEY` | Kaggle API key | Required for speeches pipeline |

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Pod stuck in `ImagePullBackOff` | Check GHCR credentials and image name |
| `ModuleNotFoundError` | Rebuild and push Docker image |
| `Connection refused` to services | Check service DNS and network policies |
| NIM timeout | Check NIM pod status and GPU availability |
| Kaggle download fails | Verify KAGGLE_USERNAME and KAGGLE_KEY secrets |

### Debugging

```bash
# Check pod logs
kubectl logs -n dagster -l app=dagster-user-deployments --tail=100

# Check events
kubectl get events -n dagster --sort-by='.lastTimestamp'

# Describe pod for detailed status
kubectl describe pod -n dagster -l app=dagster-user-deployments

# Test connectivity from pod
kubectl exec -it -n dagster deploy/dagster-user-deployments-brev-pipelines -- \
  curl -s http://minio.minio.svc.cluster.local:9000/minio/health/live
```

## License

Proprietary - Internal Use Only
