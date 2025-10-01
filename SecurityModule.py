# icrms/modules/security/auth.py
"""
Security and Authentication Module
JWT-based authentication and data encryption
"""

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthManager:
    """Handle authentication and authorization"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire_minutes = 30
        
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error(f"Token verification failed: {e}")
            return None


class DataEncryption:
    """Handle data encryption and decryption"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key if key else Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
    def encrypt(self, data: str) -> bytes:
        """Encrypt data"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt data"""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def save_key(self, filepath: str):
        """Save encryption key to file"""
        with open(filepath, 'wb') as f:
            f.write(self.key)
    
    @classmethod
    def load_key(cls, filepath: str):
        """Load encryption key from file"""
        with open(filepath, 'rb') as f:
            key = f.read()
        return cls(key)


# icrms/modules/security/__init__.py
from .auth import AuthManager, DataEncryption

__all__ = ['AuthManager', 'DataEncryption']
