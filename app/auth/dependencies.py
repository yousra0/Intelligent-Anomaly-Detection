"""
app/auth/dependencies.py
FastAPI auth dependencies — JWT verification (same secret as Next.js).

JWT is issued by the Next.js /api/auth/login route (HS256, signed with JWT_SECRET).
FastAPI validates the same token from:
  1. Authorization: Bearer <token>  header
  2. pwc_token cookie (httpOnly, set by Next.js)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

_ALGORITHM = "HS256"
_FALLBACK_SECRET = "change-me-in-production-at-least-32-chars!!"

_bearer = HTTPBearer(auto_error=False)


def _jwt_secret() -> str:
    # Read at call time so load_dotenv() order never matters
    return os.getenv("JWT_SECRET", _FALLBACK_SECRET)


@dataclass
class CurrentUser:
    id: str
    email: str
    name: str
    role: str


def _decode(token: str) -> CurrentUser:
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token invalide : {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token sans sujet.")
    return CurrentUser(
        id=sub,
        email=payload.get("email", ""),
        name=payload.get("name", ""),
        role=payload.get("role", "auditor"),
    )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> CurrentUser:
    """Extract and verify the JWT from Bearer header or pwc_token cookie."""
    # 1. Bearer header
    if credentials and credentials.credentials:
        return _decode(credentials.credentials)

    # 2. httpOnly cookie set by Next.js
    token = request.cookies.get("pwc_token")
    if token:
        return _decode(token)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentification requise.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_roles(*roles: str):
    """Dependency factory — raises 403 if caller's role is not in `roles`."""
    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès refusé. Rôle requis : {', '.join(roles)}.",
            )
        return user
    return _check
