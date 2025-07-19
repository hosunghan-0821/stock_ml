from __future__ import annotations

from typing import Optional

from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException, status
import secrets

security = HTTPBasic(auto_error=False)

def basic_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(security)
):
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )

    user_ok = secrets.compare_digest(credentials.username, "admin")
    pass_ok = secrets.compare_digest(credentials.password, "secret")
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    return credentials.username