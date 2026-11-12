"""
AlphaLens - Custom Exceptions
Structured domain exceptions mapped to appropriate HTTP status codes and error payloads.
"""


class AlphaLensException(Exception):
    """Base exception for all domain errors."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentParsingError(AlphaLensException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, details=details)


class IngestionError(AlphaLensException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class VectorStoreError(AlphaLensException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=503, details=details)


class RateLimitExceededError(AlphaLensException):
    def __init__(self, message: str = "Rate limit exceeded. Please back off.", retry_after: int = 1):
        super().__init__(message, status_code=429, details={"retry_after_seconds": retry_after})
        self.retry_after = retry_after


class CircuitBreakerOpenError(AlphaLensException):
    def __init__(self, service_name: str, recovery_time_remaining: float):
        message = f"Service '{service_name}' is currently unavailable (circuit breaker open)."
        super().__init__(
            message,
            status_code=503,
            details={
                "service": service_name,
                "recovery_time_remaining_seconds": round(recovery_time_remaining, 2)
            }
        )


class UnanswerableQueryError(AlphaLensException):
    def __init__(self, message: str = "Information not disclosed in official company filings."):
        super().__init__(message, status_code=200, details={"is_unanswerable": True})


class AuthenticationError(AlphaLensException):
    def __init__(self, message: str = "Invalid or missing API key."):
        super().__init__(message, status_code=401)
