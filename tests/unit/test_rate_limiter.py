"""
Unit tests for Token Bucket Rate Limiter & Burst Allowance.
"""
import pytest
from src.services.rate_limiter import TokenBucket, RateLimiterService
from src.core.exceptions import RateLimitExceededError


def test_token_bucket_burst_and_depletion():
    bucket = TokenBucket(capacity=5, fill_rate_per_sec=10.0)

    # Consume all 5 tokens
    for _ in range(5):
        allowed, retry_after = bucket.consume(1)
        assert allowed is True
        assert retry_after == 0.0

    # 6th request must be rejected
    allowed, retry_after = bucket.consume(1)
    assert allowed is False
    assert retry_after > 0.0


def test_rate_limiter_service_exception():
    service = RateLimiterService()
    service.rps = 2
    service.burst = 2

    client_id = "test_client_abc"
    service.check_rate_limit(client_id)
    service.check_rate_limit(client_id)

    # 3rd request should raise RateLimitExceededError
    with pytest.raises(RateLimitExceededError) as exc_info:
        service.check_rate_limit(client_id)

    assert exc_info.value.status_code == 429
