"""
Authentication Utilities

Authentication and authorization utilities for the API endpoints.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
import os

from app.utils.logging_utils import get_logger
from app.utils.exceptions import AuthenticationError, AuthorizationError

logger = get_logger("auth")

# Security scheme
security = HTTPBearer()


class User(BaseModel):
    """User model for authentication"""
    id: str
    email: str
    name: str
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class TokenData(BaseModel):
    """Token data structure"""
    user_id: Optional[str] = None
    email: Optional[str] = None
    exp: Optional[datetime] = None


# Mock user database for MVP (replace with real database)
MOCK_USERS = {
    "user_123": User(
        id="user_123",
        email="test@example.com",
        name="Test User",
        is_active=True,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    # In production, use a secure secret key from environment variables
    SECRET_KEY = "your-secret-key-here"  # TODO: Move to environment config
    ALGORITHM = "HS256"
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify JWT token and extract data"""
    try:
        SECRET_KEY = "your-secret-key-here"  # TODO: Move to environment config
        ALGORITHM = "HS256"
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None:
            raise AuthenticationError("Invalid token: missing user ID")
        
        token_data = TokenData(
            user_id=user_id,
            email=email,
            exp=datetime.fromtimestamp(payload.get("exp", 0))
        )
        return token_data
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.JWTError:
        raise AuthenticationError("Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    
    try:
        # Extract token from authorization header
        token = credentials.credentials
        
        # Verify token
        token_data = verify_token(token)
        
        # Get user from database (mock implementation)
        user = MOCK_USERS.get(token_data.user_id)
        if user is None:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("User account is inactive")
        
        logger.debug(f"Authenticated user: {user.id}")
        return user
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_permission(permission: str):
    """Decorator to require specific permissions"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        # In a real implementation, check user permissions here
        # For MVP, all authenticated users have all permissions
        return current_user
    
    return permission_checker


async def validate_book_ownership(book_id: str, user_id: str, storage_service) -> bool:
    """Validate that a user owns a specific book"""
    try:
        book_metadata = await storage_service.get_book_metadata(book_id)
        return book_metadata and book_metadata.get("owner_user_id") == user_id
    except Exception as e:
        logger.error(f"Error validating book ownership: {str(e)}")
        return False


async def validate_workflow_ownership(workflow_id: str, user_id: str, publishing_service) -> bool:
    """Validate that a user owns a specific publishing workflow"""
    try:
        workflow = await publishing_service.get_workflow(workflow_id)
        return workflow and workflow.user_id == user_id
    except Exception as e:
        logger.error(f"Error validating workflow ownership: {str(e)}")
        return False