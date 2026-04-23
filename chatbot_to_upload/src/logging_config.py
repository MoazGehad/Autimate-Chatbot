import logging
import time
from pythonjsonlogger import jsonlogger
from contextlib import contextmanager

def get_logger(name: str) -> logging.Logger:
    """Configures and returns a structured JSON logger."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    
    # Prevent log messages from being propagated to the root logger
    logger.propagate = False
    
    return logger

@contextmanager
def log_latency(logger: logging.Logger, operation: str):
    """Context manager to log the latency of an operation."""
    start_time = time.time()
    try:
        yield
    finally:
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Operation completed: {operation}", extra={
            "operation": operation,
            "latency_ms": round(latency_ms, 2)
        })
