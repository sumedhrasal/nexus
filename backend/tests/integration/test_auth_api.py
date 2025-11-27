"""Integration tests for authentication API endpoints."""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import select, text
from sqlalchemy.pool import NullPool

from app.main import app
from app.models.database import Organization, APIKey
from app.storage.postgres import get_db
from app.config import settings


@pytest_asyncio.fixture(scope="function", autouse=True)
async def clean_db():
    """Clean database before each test.

    Uses TRUNCATE to remove all data before each test runs.
    This ensures complete test isolation.
    """
    # Create a new engine for each test to avoid event loop issues
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        poolclass=NullPool  # Disable connection pooling to avoid event loop issues
    )

    try:
        async with engine.begin() as conn:
            # Truncate all tables to ensure clean state
            await conn.execute(text("""
                TRUNCATE TABLE
                    search_analytics,
                    entities,
                    collections,
                    api_keys,
                    organizations
                RESTART IDENTITY CASCADE
            """))
    finally:
        await engine.dispose()

    yield


@pytest_asyncio.fixture
async def client():
    """Create async HTTP client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session():
    """Get a database session for test assertions."""
    async for session in get_db():
        yield session
        break


class TestBootstrapEndpoint:
    """Test /auth/bootstrap endpoint."""

    @pytest.mark.asyncio
    async def test_bootstrap_creates_organization(self, client, db_session):
        """Test bootstrap creates organization and API key."""
        # Bootstrap
        response = await client.post("/auth/bootstrap")

        assert response.status_code == 201
        data = response.json()

        # Verify response structure
        assert "id" in data
        assert "organization_id" in data
        assert "key" in data
        assert "name" in data
        assert "created_at" in data

        # Verify key format
        assert data["key"].startswith("nx_")
        assert len(data["key"]) > 40

        # Verify organization created in database
        result = await db_session.execute(select(Organization))
        orgs = result.scalars().all()
        assert len(orgs) == 1
        assert str(orgs[0].id) == data["organization_id"]

        # Verify API key created
        result = await db_session.execute(select(APIKey))
        keys = result.scalars().all()
        assert len(keys) == 1
        assert keys[0].is_active is True

    @pytest.mark.asyncio
    async def test_bootstrap_fails_if_organization_exists(self, client):
        """Test bootstrap fails if organization already exists."""
        # First bootstrap
        await client.post("/auth/bootstrap")

        # Try to bootstrap again
        response = await client.post("/auth/bootstrap")

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()


class TestAPIKeyManagement:
    """Test API key management endpoints."""

    @pytest_asyncio.fixture
    async def bootstrap_key(self, client):
        """Bootstrap and return API key."""
        # Bootstrap
        response = await client.post("/auth/bootstrap")
        return response.json()["key"]

    @pytest.mark.asyncio
    async def test_create_api_key_requires_auth(self, client):
        """Test creating API key requires authentication."""
        response = await client.post(
            "/auth/keys",
            json={"name": "Test Key"}
        )

        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_api_key_with_auth(self, client, bootstrap_key):
        """Test creating API key with valid authentication."""
        response = await client.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"},
            json={"name": "Secondary Key"}
        )

        assert response.status_code == 201
        data = response.json()

        assert data["name"] == "Secondary Key"
        assert data["key"].startswith("nx_")
        assert data["key"] != bootstrap_key  # Different key

    @pytest.mark.asyncio
    async def test_create_api_key_without_name(self, client, bootstrap_key):
        """Test creating API key without name."""
        response = await client.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"},
            json={}
        )

        assert response.status_code == 201
        data = response.json()

        assert data["name"] is None
        assert data["key"].startswith("nx_")

    @pytest.mark.asyncio
    async def test_list_api_keys_requires_auth(self, client):
        """Test listing API keys requires authentication."""
        response = await client.get("/auth/keys")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_api_keys_with_auth(self, client, bootstrap_key):
        """Test listing API keys with valid authentication."""
        # Create additional key
        await client.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"},
            json={"name": "Second Key"}
        )

        # List keys
        response = await client.get(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 2  # Bootstrap + new key
        assert all("id" in key for key in data)
        assert all("is_active" in key for key in data)
        assert all("created_at" in key for key in data)
        # Actual key should NOT be in list response
        assert all("key" not in key for key in data)

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, client, bootstrap_key, db_session):
        """Test revoking API key."""
        # Create key to revoke
        create_response = await client.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"},
            json={"name": "Key to Revoke"}
        )
        key_id = create_response.json()["id"]

        # Revoke it
        response = await client.delete(
            f"/auth/keys/{key_id}",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        assert response.status_code == 204

        # Verify it's deactivated
        result = await db_session.execute(
            select(APIKey).where(APIKey.id == key_id)
        )
        api_key = result.scalar_one()
        assert api_key.is_active is False

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key(self, client, bootstrap_key):
        """Test revoking non-existent key fails."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = await client.delete(
            f"/auth/keys/{fake_id}",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_activate_revoked_key(self, client, bootstrap_key, db_session):
        """Test reactivating a revoked key."""
        # Create and revoke key
        create_response = await client.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"},
            json={"name": "Key to Reactivate"}
        )
        key_id = create_response.json()["id"]

        await client.delete(
            f"/auth/keys/{key_id}",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        # Reactivate
        response = await client.post(
            f"/auth/keys/{key_id}/activate",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "activated"

        # Verify it's active
        result = await db_session.execute(
            select(APIKey).where(APIKey.id == key_id)
        )
        api_key = result.scalar_one()
        assert api_key.is_active is True


class TestAuthenticationHeaders:
    """Test different authentication header formats."""

    @pytest_asyncio.fixture
    async def bootstrap_key(self, client):
        """Bootstrap and return API key."""
        response = await client.post("/auth/bootstrap")
        return response.json()["key"]

    @pytest.mark.asyncio
    async def test_bearer_token_format(self, client, bootstrap_key):
        """Test Bearer token format."""
        response = await client.get(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_direct_token_format(self, client, bootstrap_key):
        """Test direct token format (no Bearer)."""
        response = await client.get(
            "/auth/keys",
            headers={"Authorization": bootstrap_key}
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_token_prefix(self, client):
        """Test invalid token prefix fails."""
        response = await client.get(
            "/auth/keys",
            headers={"Authorization": "Bearer invalid_prefix_12345"}
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_header(self, client):
        """Test missing auth header fails."""
        response = await client.get("/auth/keys")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_revoked_key_fails(self, client, bootstrap_key):
        """Test revoked key cannot authenticate."""
        # Create and revoke a key
        create_response = await client.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"},
            json={"name": "Key to Revoke"}
        )
        revoked_key = create_response.json()["key"]
        key_id = create_response.json()["id"]

        await client.delete(
            f"/auth/keys/{key_id}",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        # Try to use revoked key
        response = await client.get(
            "/auth/keys",
            headers={"Authorization": f"Bearer {revoked_key}"}
        )

        assert response.status_code == 401


class TestLastUsedTracking:
    """Test last_used_at timestamp tracking."""

    @pytest_asyncio.fixture
    async def bootstrap_key(self, client):
        """Bootstrap and return API key."""
        response = await client.post("/auth/bootstrap")
        return response.json()["key"]

    @pytest.mark.asyncio
    async def test_last_used_updates_on_use(self, client, bootstrap_key, db_session):
        """Test last_used_at updates when key is used."""
        # Get initial state
        result = await db_session.execute(select(APIKey))
        initial_key = result.scalars().first()
        initial_last_used = initial_key.last_used_at
        key_id = initial_key.id

        # Use the key
        await client.get(
            "/auth/keys",
            headers={"Authorization": f"Bearer {bootstrap_key}"}
        )

        # Check updated timestamp - need to close and reopen session to see committed changes
        await db_session.close()

        # Get a fresh session to see the updates
        async for fresh_session in get_db():
            result = await fresh_session.execute(
                select(APIKey).where(APIKey.id == key_id)
            )
            updated_key = result.scalar_one()
            assert updated_key.last_used_at is not None
            if initial_last_used:
                assert updated_key.last_used_at >= initial_last_used
            break
