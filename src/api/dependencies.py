"""
AlphaLens - API Dependencies
Handles API key authentication and request context.
"""
from fastapi import Header, HTTPException, Security
from src.config import settings


async def verify_api_key(
    x_api_key: str = Header(default="alpha-lens-master-key-prod-2025")
) -> str:
    """Validates the incoming client API key."""
    if settings.DEBUG:
        return x_api_key

    if x_api_key != settings.MASTER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid API Key provided in 'X-API-Key' header."
        )
    return x_api_key

