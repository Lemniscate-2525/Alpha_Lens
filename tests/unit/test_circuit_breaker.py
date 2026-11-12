"""
Unit tests for Circuit Breaker State Machine & Failure Thresholds.
"""
import pytest
import time
from src.services.circuit_breaker import CircuitBreaker, CircuitState
from src.core.exceptions import CircuitBreakerOpenError


def test_circuit_breaker_transitions():
    cb = CircuitBreaker(service_name="TestService", failure_threshold=3, recovery_timeout=0.1)

    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # Record 2 failures (below threshold)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # Record 3rd failure (trips to OPEN)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

    # Attempting to execute while OPEN must raise CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError):
        cb.execute(lambda: "result")

    # Wait for recovery timeout
    time.sleep(0.12)
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Record success while HALF_OPEN -> transitions back to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0
