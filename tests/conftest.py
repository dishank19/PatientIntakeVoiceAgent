import pytest
import os
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    test_dir = Path("tests/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

@pytest.fixture(scope="session")
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("DAILY_API_KEY", "test_daily_key")
    monkeypatch.setenv("CARTESIA_API_KEY", "test_cartesia_key")
    monkeypatch.setenv("DAILY_ROOM_URL", "https://test.daily.co/test-room")
    monkeypatch.setenv("DAILY_ROOM_NAME", "test-room")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG") 