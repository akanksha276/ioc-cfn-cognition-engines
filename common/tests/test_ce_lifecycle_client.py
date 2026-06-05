# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CE lifecycle client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from common.ce_lifecycle.client import CELifecycleClient
from common.ce_lifecycle.models import (
    CERegistrationRequest,
)


@pytest.fixture
def sample_request():
    """Sample registration request for testing."""
    return CERegistrationRequest(
        name="Test CE",
        url="http://test-ce:9004",
        version="1.0.0",
        kind="knowledge",
        subkind="query",
        capabilities=["test_capability"],
        metrics=["test.metric"],
        config={"test_key": "test_value"},
        mas_config=None,
        mas_auto_associate=False,
    )


@pytest.fixture
def mock_success_response():
    """Mock successful registration response."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "ce_id": "test-ce-123",
        "cfn_id": "test-cfn-456",
        "name": "Test CE",
        "version": "1.0.0",
        "kind": "knowledge",
        "subkind": "query",
        "enabled": True,
        "status": "offline",
        "created": True,
    }
    return mock_response


@pytest.fixture
def mock_update_response():
    """Mock CE update response (already exists)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ce_id": "existing-ce-123",
        "cfn_id": "test-cfn-456",
        "name": "Test CE",
        "version": "1.0.0",
        "kind": "knowledge",
        "subkind": "query",
        "enabled": True,
        "status": "online",
        "created": False,  # indicates update, not create
    }
    return mock_response


@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initialization."""
    client = CELifecycleClient(
        cfn_base_url="http://localhost:9002",
        heartbeat_interval_sec=30.0,
        timeout=10.0,
    )

    assert client.cfn_base_url == "http://localhost:9002"
    assert client.heartbeat_interval_sec == 30.0
    assert client.timeout == 10.0
    assert client.ce_id is None
    assert client.cfn_id is None
    assert client._client is None
    assert client._heartbeat_task is None

    await client.close()


@pytest.mark.asyncio
async def test_client_strips_trailing_slash():
    """Test that client strips trailing slash from base URL."""
    client = CELifecycleClient("http://localhost:9002/")
    assert client.cfn_base_url == "http://localhost:9002"
    await client.close()


@pytest.mark.asyncio
async def test_register_success_created(sample_request, mock_success_response):
    """Test successful registration (new CE created)."""
    client = CELifecycleClient("http://localhost:9002")

    # Mock httpx client
    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_success_response
    client._client = mock_http_client

    # Register
    response = await client.register(sample_request)

    # Assert response
    assert response is not None
    assert response.ce_id == "test-ce-123"
    assert response.cfn_id == "test-cfn-456"
    assert response.name == "Test CE"
    assert response.version == "1.0.0"
    assert response.kind == "knowledge"
    assert response.subkind == "query"
    assert response.enabled is True
    assert response.status == "offline"
    assert response.created is True

    # Assert client state
    assert client.ce_id == "test-ce-123"
    assert client.cfn_id == "test-cfn-456"
    assert client.name == "Test CE"
    assert client.version == "1.0.0"

    # Verify HTTP call
    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    assert call_args[0][0] == "http://localhost:9002/api/cognition-engines"
    assert call_args[1]["json"]["name"] == "Test CE"
    assert call_args[1]["json"]["version"] == "1.0.0"

    await client.close()


@pytest.mark.asyncio
async def test_register_success_updated(sample_request, mock_update_response):
    """Test successful registration (existing CE updated)."""
    client = CELifecycleClient("http://localhost:9002")

    # Mock httpx client
    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_update_response
    client._client = mock_http_client

    # Register
    response = await client.register(sample_request)

    # Assert response
    assert response is not None
    assert response.ce_id == "existing-ce-123"
    assert response.created is False  # indicates update

    # Assert client state
    assert client.ce_id == "existing-ce-123"

    await client.close()


@pytest.mark.asyncio
async def test_register_failure_500(sample_request):
    """Test registration failure with 500 error."""
    client = CELifecycleClient("http://localhost:9002")

    # Mock httpx client with 500 error
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_http_client.post.return_value = mock_response
    client._client = mock_http_client

    # Register
    response = await client.register(sample_request)

    # Assert failure
    assert response is None
    assert client.ce_id is None

    await client.close()


@pytest.mark.asyncio
async def test_register_connection_error(sample_request):
    """Test registration failure with connection error (exhausts all retries)."""
    client = CELifecycleClient("http://localhost:9002", max_retries=3)

    # Mock httpx client with connection error
    mock_http_client = AsyncMock()
    mock_http_client.post.side_effect = httpx.ConnectError("Connection refused")
    client._client = mock_http_client

    # Register - should retry 3 times then fail
    response = await client.register(sample_request)

    # Assert failure after all retries
    assert response is None
    assert client.ce_id is None
    # Verify it tried 3 times
    assert mock_http_client.post.call_count == 3

    await client.close()


@pytest.mark.asyncio
async def test_register_retry_then_success(sample_request):
    """Test registration succeeds after initial connection failures."""
    client = CELifecycleClient("http://localhost:9002", max_retries=3)

    # Mock httpx client: fail twice, then succeed
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "ce_id": "test-ce-123",
        "cfn_id": "cfn-456",
        "name": "Test CE",
        "version": "1.0.0",
        "kind": "knowledge",
        "subkind": "query",
        "enabled": True,
        "status": "online",
        "created": True,
    }

    mock_http_client.post.side_effect = [
        httpx.ConnectError("Connection refused"),  # 1st attempt fails
        httpx.ConnectError("Connection refused"),  # 2nd attempt fails
        mock_response,  # 3rd attempt succeeds
    ]
    client._client = mock_http_client

    # Register - should succeed on 3rd try
    response = await client.register(sample_request)

    # Assert success
    assert response is not None
    assert response.ce_id == "test-ce-123"
    assert client.ce_id == "test-ce-123"
    # Verify it tried 3 times
    assert mock_http_client.post.call_count == 3

    await client.close()


@pytest.mark.asyncio
async def test_send_heartbeat_success():
    """Test successful heartbeat."""
    client = CELifecycleClient("http://localhost:9002")
    client.ce_id = "test-ce-123"
    client.name = "Test CE"

    # Mock httpx client
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "online",
        "last_seen": "2026-05-21T10:30:00Z",
    }
    mock_http_client.put.return_value = mock_response
    client._client = mock_http_client

    # Send heartbeat
    success = await client.send_heartbeat()

    # Assert success
    assert success is True

    # Verify HTTP call
    mock_http_client.put.assert_called_once()
    call_args = mock_http_client.put.call_args
    assert call_args[0][0] == "http://localhost:9002/api/cognition-engines/test-ce-123/heartbeat"

    await client.close()


@pytest.mark.asyncio
async def test_send_heartbeat_no_ce_id():
    """Test heartbeat when ce_id is not set."""
    client = CELifecycleClient("http://localhost:9002")

    # No ce_id set
    assert client.ce_id is None

    # Send heartbeat
    success = await client.send_heartbeat()

    # Assert failure
    assert success is False

    await client.close()


@pytest.mark.asyncio
async def test_send_heartbeat_failure():
    """Test heartbeat failure."""
    client = CELifecycleClient("http://localhost:9002")
    client.ce_id = "test-ce-123"
    client.name = "Test CE"

    # Mock httpx client with error
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "CE not found"
    mock_http_client.put.return_value = mock_response
    client._client = mock_http_client

    # Send heartbeat
    success = await client.send_heartbeat()

    # Assert failure
    assert success is False

    await client.close()


@pytest.mark.asyncio
async def test_start_heartbeat():
    """Test starting heartbeat task."""
    client = CELifecycleClient("http://localhost:9002", heartbeat_interval_sec=0.1)
    client.ce_id = "test-ce-123"
    client.name = "Test CE"

    # Start heartbeat
    client.start_heartbeat()

    # Assert task created
    assert client._heartbeat_task is not None
    assert not client._heartbeat_task.done()

    # Wait briefly
    await asyncio.sleep(0.05)

    # Stop
    await client.close()

    # Assert task cancelled
    assert client._heartbeat_task is None


@pytest.mark.asyncio
async def test_start_heartbeat_no_ce_id():
    """Test starting heartbeat when ce_id is not set."""
    client = CELifecycleClient("http://localhost:9002")

    # No ce_id
    assert client.ce_id is None

    # Start heartbeat
    client.start_heartbeat()

    # Assert task not created
    assert client._heartbeat_task is None

    await client.close()


@pytest.mark.asyncio
async def test_start_heartbeat_already_running():
    """Test starting heartbeat when already running."""
    client = CELifecycleClient("http://localhost:9002", heartbeat_interval_sec=0.1)
    client.ce_id = "test-ce-123"
    client.name = "Test CE"

    # Start heartbeat
    client.start_heartbeat()
    task1 = client._heartbeat_task

    # Try to start again
    client.start_heartbeat()
    task2 = client._heartbeat_task

    # Assert same task
    assert task1 is task2

    await client.close()


@pytest.mark.asyncio
async def test_heartbeat_loop_sends_periodically():
    """Test that heartbeat loop sends heartbeats periodically."""
    client = CELifecycleClient("http://localhost:9002", heartbeat_interval_sec=0.1)
    client.ce_id = "test-ce-123"
    client.name = "Test CE"

    # Mock httpx client
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "online", "last_seen": "2026-05-21T10:30:00Z"}
    mock_http_client.put.return_value = mock_response
    client._client = mock_http_client

    # Start heartbeat
    client.start_heartbeat()

    # Wait for multiple heartbeats
    await asyncio.sleep(0.35)

    # Stop
    await client.close()

    # Assert heartbeat was called multiple times (at least 2-3 times)
    assert mock_http_client.put.call_count >= 2


@pytest.mark.asyncio
async def test_close_cancels_heartbeat_task():
    """Test that close() cancels heartbeat task."""
    client = CELifecycleClient("http://localhost:9002", heartbeat_interval_sec=0.1)
    client.ce_id = "test-ce-123"
    client.name = "Test CE"

    # Start heartbeat
    client.start_heartbeat()
    assert client._heartbeat_task is not None

    # Close
    await client.close()

    # Assert task cancelled
    assert client._heartbeat_task is None


@pytest.mark.asyncio
async def test_close_closes_http_client():
    """Test that close() closes HTTP client."""
    client = CELifecycleClient("http://localhost:9002")

    # Create HTTP client
    _ = await client._get_client()
    assert client._client is not None

    # Close
    await client.close()

    # Assert HTTP client closed
    assert client._client is None


@pytest.mark.asyncio
async def test_heartbeat_loop_continues_after_error():
    """Test that heartbeat loop continues after transient errors."""
    client = CELifecycleClient("http://localhost:9002", heartbeat_interval_sec=0.1)
    client.ce_id = "test-ce-123"
    client.name = "Test CE"

    # Mock httpx client to fail first 2 times, then succeed
    mock_http_client = AsyncMock()

    # Track calls using a list to avoid closure issues
    calls = []

    async def mock_put_with_failures(url):
        calls.append(url)
        if len(calls) <= 2:
            # First 2 calls fail
            raise Exception("Transient network error")
        # Subsequent calls succeed
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "online",
            "last_seen": "2026-05-21T10:30:00Z",
        }
        return mock_response

    mock_http_client.put.side_effect = mock_put_with_failures
    client._client = mock_http_client

    # Start heartbeat
    client.start_heartbeat()

    # Wait for multiple heartbeat attempts (including failed ones)
    await asyncio.sleep(0.45)

    # Stop
    await client.close()

    # Assert heartbeat was called multiple times despite initial failures
    assert len(calls) >= 3, f"Expected at least 3 heartbeat attempts, got {len(calls)}"
