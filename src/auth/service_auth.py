import hmac
import os

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


async def verify_service_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    tokens = request.app.state.settings.inbound_service_tokens
    if not tokens:
        raise HTTPException(
            status_code=500, detail="MEDIA_SERVICE_TOKEN not configured"
        )
    if not any(hmac.compare_digest(credentials.credentials, token) for token in tokens):
        raise HTTPException(status_code=401, detail="Invalid service token")
    return credentials.credentials


async def verify_migration_coordinator_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = os.getenv("DATABASE_MIGRATION_COORDINATOR_SERVICE_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="Database migration capability is not configured",
        )
    if not hmac.compare_digest(credentials.credentials, token):
        raise HTTPException(
            status_code=401,
            detail="Invalid migration coordinator token",
        )
    return credentials.credentials
