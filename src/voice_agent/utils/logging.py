import sys
from loguru import logger
from ..config.settings import get_settings

def setup_logging():
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler
    logger.add(
        f"{settings.DATA_DIR}/logs/voice_agent.log",
        rotation="500 MB",
        retention="10 days",
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        backtrace=True,
        diagnose=True
    )
    
    return logger 