import pytest
from loguru import logger
from src.voice_agent.utils.logging import setup_logging

def test_logging_setup(mock_env_vars, test_data_dir):
    """Test that logging is set up correctly."""
    # Setup logging
    logger = setup_logging()
    
    # Test that we can log messages
    test_message = "Test log message"
    logger.info(test_message)
    
    # Check that log file was created
    log_file = test_data_dir / "logs/voice_agent.log"
    assert log_file.exists()
    
    # Read the log file and check the message
    with open(log_file, "r") as f:
        log_content = f.read()
        assert test_message in log_content 