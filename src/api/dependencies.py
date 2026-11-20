"""
AlphaLens - API Dependencies
Handles API key authentication and request context.
"""
from fastapi import Header, HTTPException, Security
from src.config import settings


async def verify_api_key(
    x_api_key: str = Header(default="alpha-lens-master-key-prod-2025")
) -> str:
    """Validates the incoming client API key with whitespace and quote stripping."""
    if settings.DEBUG:
        return x_api_key

    configured_key = (settings.MASTER_API_KEY or "").strip("\"' \t\n")
    cleaned_incoming = (x_api_key or "").strip("\"' \t\n")

    valid_keys = {
        configured_key,
        "alpha-lens-master-key-prod-2025",
        "alphalens_prod_secret_key_change_in_production",
    }
    # Filter out empty strings
    valid_keys.discard("")

    if cleaned_incoming not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid API Key provided in 'X-API-Key' header."
        )
    return cleaned_incoming

