"""Tests for resources."""

import pytest


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
    """Test NIM resource initialization."""
    from brev_pipelines.resources.nim import NIMResource

    resource = NIMResource(
        endpoint="http://localhost:8000",
        model="meta/llama3-8b-instruct",
        timeout=30,
    )
    assert resource.endpoint == "http://localhost:8000"
    assert resource.model == "meta/llama3-8b-instruct"


def test_nim_resource_default_model():
    """Test NIM resource has default model."""
    from brev_pipelines.resources.nim import NIMResource

    resource = NIMResource(endpoint="http://localhost:8000")
    assert resource.model == "meta/llama3-8b-instruct"
