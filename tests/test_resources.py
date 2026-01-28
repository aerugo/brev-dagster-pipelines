"""Tests for resources."""


def test_minio_resource_init():
    """Test MinIO resource initialization."""
    from brev_pipelines.resources.minio import MinIOResource

    resource = MinIOResource(
        endpoint="localhost:9000",
        access_key="test",
        secret_key="test123",
        secure=False,
    )
    assert resource.endpoint == "localhost:9000"
    assert resource.secure is False


def test_lakefs_resource_init():
    """Test LakeFS resource initialization."""
    from brev_pipelines.resources.lakefs import LakeFSResource

    resource = LakeFSResource(
        endpoint="localhost:8000",
        access_key="test",
        secret_key="test123",
    )
    assert resource.endpoint == "localhost:8000"


def test_nim_resource_init():
    """Test NIM resource initialization with custom model."""
    from brev_pipelines.resources.nim import NIMResource

    resource = NIMResource(
        endpoint="http://localhost:8000",
        model="meta/llama-3.1-8b-instruct",
        timeout=30,
    )
    assert resource.endpoint == "http://localhost:8000"
    assert resource.model == "meta/llama-3.1-8b-instruct"


def test_nim_resource_default_model():
    """Test NIM resource has default model."""
    from brev_pipelines.resources.nim import NIMResource

    resource = NIMResource(endpoint="http://localhost:8000")
    assert resource.model == "meta/llama-3.1-8b-instruct"


def test_nim_embedding_resource_init():
    """Test NIM embedding resource initialization."""
    from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource

    resource = NIMEmbeddingResource(
        endpoint="http://localhost:8000",
    )
    assert resource.endpoint == "http://localhost:8000"
    assert resource.model == "nvidia/nv-embedqa-e5-v5"
    assert resource.dimensions == 1024


def test_nim_embedding_resource_defaults():
    """Test NIM embedding resource default values."""
    from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource

    resource = NIMEmbeddingResource()
    assert "nvidia-nim-embedding" in resource.endpoint
    assert resource.timeout == 120
    assert resource.max_retries == 3


def test_nim_embedding_resource_custom_config():
    """Test NIM embedding resource with custom configuration."""
    from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource

    resource = NIMEmbeddingResource(
        endpoint="http://custom-nim:8000",
        timeout=60,
        max_retries=5,
    )
    assert resource.endpoint == "http://custom-nim:8000"
    assert resource.timeout == 60
    assert resource.max_retries == 5


def test_weaviate_resource_init():
    """Test Weaviate resource initialization."""
    from brev_pipelines.resources.weaviate import WeaviateResource

    resource = WeaviateResource(
        host="weaviate.weaviate.svc.cluster.local",
        port=80,
    )
    assert resource.host == "weaviate.weaviate.svc.cluster.local"
    assert resource.port == 80


def test_weaviate_resource_defaults():
    """Test Weaviate resource default port configuration."""
    from brev_pipelines.resources.weaviate import WeaviateResource

    resource = WeaviateResource()
    assert resource.port == 80
    assert resource.grpc_host == "weaviate-grpc.weaviate.svc.cluster.local"
    assert resource.grpc_port == 50051


def test_weaviate_resource_custom_ports():
    """Test Weaviate resource with custom ports."""
    from brev_pipelines.resources.weaviate import WeaviateResource

    resource = WeaviateResource(
        host="localhost",
        port=9080,
        grpc_host="localhost",
        grpc_port=50052,
    )
    assert resource.port == 9080
    assert resource.grpc_host == "localhost"
    assert resource.grpc_port == 50052


class TestSafeSynthesizerResource:
    """Tests for Safe Synthesizer resource with KAI integration."""

    def test_initialization(self):
        """Test Safe Synthesizer resource initialization."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource(
            namespace="nvidia-ai",
            priority_class="build-preemptible",
        )
        assert resource.namespace == "nvidia-ai"
        assert resource.priority_class == "build-preemptible"

    def test_default_config(self):
        """Test default configuration values."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource()
        assert resource.poll_interval == 30
        assert resource.max_wait_time == 7200
        assert resource.gpu_memory == "40Gi"

    def test_priority_class_setting(self):
        """Test that priority class is configurable."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource(priority_class="train")
        assert resource.priority_class == "train"

    def test_custom_namespace(self):
        """Test custom namespace configuration."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource(namespace="custom-namespace")
        assert resource.namespace == "custom-namespace"

    def test_service_endpoint_default(self):
        """Test default service endpoint uses Safe Synthesizer API."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource()
        assert "nemo-safe-synthesizer" in resource.service_endpoint
        assert "8000" in resource.service_endpoint

    def test_custom_service_endpoint(self):
        """Test custom service endpoint."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource(service_endpoint="http://custom-synth:9000")
        assert resource.service_endpoint == "http://custom-synth:9000"

    def test_image_default(self):
        """Test default image configuration."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource()
        assert "safe-synthesizer" in resource.image
        assert "nvcr.io" in resource.image

    def test_kai_orchestration_settings(self):
        """Test KAI scheduler orchestration settings."""
        from brev_pipelines.resources.safe_synth import SafeSynthesizerResource

        resource = SafeSynthesizerResource(
            priority_class="build-preemptible",
            gpu_memory="40Gi",
        )
        # Safe Synth coexists with nim-llm via KAI fractional GPU allocation
        assert resource.priority_class == "build-preemptible"
        assert resource.gpu_memory == "40Gi"
