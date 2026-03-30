from typing import Dict, Any

class ValidationException(Exception):
    """Custom exception for validation errors with detailed information."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.details = details or {}
