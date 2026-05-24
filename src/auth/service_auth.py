from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


async def verify_service_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = request.app.state.settings.service_auth_token
    if not token:
        raise HTTPException(status_code=500, detail="SERVICE_AUTH_TOKEN not configured")
    if credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid service token")
    return credentials.credentials
