import pytest
from src.voice_agent.config.settings import Settings, get_settings

def test_settings_loading(mock_env_vars):
    """Test that settings are loaded correctly from environment variables."""
    settings = get_settings()
    
    assert settings.OPENAI_API_KEY == "test_openai_key"
    assert settings.DAILY_API_KEY == "test_daily_key"
    assert settings.CARTESIA_API_KEY == "test_cartesia_key"
    assert settings.DAILY_ROOM_URL == "https://test.daily.co/test-room"
    assert settings.DAILY_ROOM_NAME == "test-room"
    assert settings.LOG_LEVEL == "DEBUG"
    
    # Test default values
    assert settings.OPENAI_MODEL == "gpt-4-turbo-preview"
    assert settings.OPENAI_TEMPERATURE == 0.7
    assert settings.SAMPLE_RATE == 16000
    assert settings.CHANNELS == 1
    assert settings.DATA_DIR == "data"

def test_settings_singleton():
    """Test that get_settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2 