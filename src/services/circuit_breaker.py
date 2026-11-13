"""
AlphaLens - Circuit Breaker Pattern
Protects system from cascading failures during downstream LLM API outages or network degradation.
"""
from enum import Enum
import time
from typing import Callable, Any
from src.config import settings
from src.core.exceptions import CircuitBreakerOpenError
from src.observability.logging import logger


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal execution
    OPEN = "OPEN"            # Tripped; reject requests immediately
    HALF_OPEN = "HALF_OPEN"  # Testing recovery with trial request


class CircuitBreaker:
    """State machine governing failure thresholds and recovery timeouts."""

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = None,
        recovery_timeout: float = None
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold or settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout or settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS

        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0

    def can_execute(self) -> bool:
        """Determines whether a call can proceed based on current circuit state."""
        now = time.monotonic()

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = now - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    f"Circuit breaker for '{self.service_name}' transitioning from OPEN to HALF_OPEN (probing)."
                )
                self.state = CircuitState.HALF_OPEN
                return True
            else:
                return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self):
        """Records a successful execution, resetting circuit breaker to CLOSED."""
        if self.state != CircuitState.CLOSED:
            logger.info(
                f"Circuit breaker for '{self.service_name}' recovered! Transitioning to CLOSED."
            )
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0

    def record_failure(self):
        """Records a failure and trips the breaker to OPEN if threshold is reached."""
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker for '{self.service_name}' TRIPPED to OPEN! "
                f"Consecutive failures: {self.consecutive_failures}. "
                f"Cooling down for {self.recovery_timeout}s."
            )

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Executes a function wrapped in circuit breaker protection."""
        if not self.can_execute():
            remaining = self.recovery_timeout - (time.monotonic() - self.last_failure_time)
            raise CircuitBreakerOpenError(
                service_name=self.service_name,
                recovery_time_remaining=max(0.0, remaining)
            )

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


# Global singleton circuit breaker for LLM generation
llm_circuit_breaker = CircuitBreaker(service_name="LLM_Provider")

