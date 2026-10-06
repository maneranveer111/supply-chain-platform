from enum import Enum

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.redis import get_redis
from app.db.session import get_db

oauth2_scheme = HTTPBearer()

class Role(str, Enum):
    PROCUREMENT_MANAGER = "procurement_manager"
    STORE_ANALYST = "store_analyst"
    ADMIN = "admin"


class CurrentUser:
    def __init__(self, user_id: int, email: str, role: str):
        self.user_id = user_id
        self.email = email
        self.role = role


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> CurrentUser:
    token = credentials.credentials

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return CurrentUser(
        user_id=payload.get("sub"),
        email=payload.get("email"),
        role=payload.get("role"),
    )


def require_role(*allowed_roles: Role):
    def checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in [r.value for r in allowed_roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this action",
            )
        return user

    return checker


def rate_limiter(max_requests: int, window_seconds: int):
    """
    Sliding-window rate limiter backed by Redis sorted sets.
    Mirrors the limiter pattern already used in the URL-shortener project.
    """

    async def limiter(request: Request, user: CurrentUser = Depends(get_current_user)):
        redis = await get_redis()
        key = f"ratelimit:{user.user_id}:{request.url.path}"
        now = __import__("time").time()
        window_start = now - window_seconds

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds)
        _, _, request_count, _ = await pipe.execute()

        if request_count > max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
            )

    return limiter
