import hmac

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
