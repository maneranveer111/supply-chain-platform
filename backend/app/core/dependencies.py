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


import logging

logger = logging.getLogger(__name__)


def rate_limiter(max_requests: int, window_seconds: int):
    """
    Sliding-window rate limiter backed by Redis sorted sets.
    Fails open gracefully if Redis is unavailable or times out.
    """

    async def limiter(request: Request, user: CurrentUser = Depends(get_current_user)):
        try:
            redis = await get_redis()
            identifier = user.user_id if user and hasattr(user, "user_id") else (
                request.client.host if request.client else "anonymous"
            )
            key = f"ratelimit:{identifier}:{request.url.path}"
            now = __import__("time").time()
            window_start = now - window_seconds

            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, window_seconds)
            results = await pipe.execute()
            request_count = results[2]

            if request_count > max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Try again later.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            # Production fail-open guarantee: Redis failure must not break customer API traffic
            logger.warning("Rate limiter failed open due to Redis error: %s", exc)
            return

    return limiter

