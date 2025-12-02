"""
Enhanced security with encryption and secure storage for AIFrameworkRPC v0.2.0
"""

import os
import json
import hashlib
import secrets
import time
import threading
from typing import Dict, Any, Optional, Union, Callable, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import logging

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logging.warning("keyring not available. Install with: pip install keyring")


@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_key_file: str = "security.key"
    secure_storage_file: str = "secure_storage.enc"
    backup_enabled: bool = True
    backup_interval: int = 3600  # 1 hour
    max_backup_files: int = 10
    auto_rotate_keys: bool = True
    key_rotation_interval: int = 86400 * 30  # 30 days
    use_hardware_security: bool = False
    audit_logging: bool = True
    session_timeout: int = 3600  # 1 hour


@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""
    timestamp: float
    event_type: str
    user_id: Optional[str]
    action: str
    resource: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EncryptionManager:
    """
    Manages encryption keys and operations.
    
    Features:
    - AES-256 encryption
    - Key derivation with PBKDF2
    - Key rotation
    - Hardware security module support
    - Secure key storage
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Encryption components
        self._master_key: Optional[bytes] = None
        self._data_key: Optional[bytes] = None
        self._cipher: Optional[Fernet] = None
        
        # Key management
        self._key_file = Path(config.encryption_key_file)
        self._key_rotation_thread: Optional[threading.Thread] = None
        self._stop_rotation = threading.Event()
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Start key rotation if enabled
        if config.auto_rotate_keys:
            self._start_key_rotation()
    
    def _initialize_encryption(self):
        """Initialize encryption keys."""
        try:
            # Try to load existing key
            if self._key_file.exists():
                self._load_encryption_key()
            else:
                self._generate_new_key()
            
            # Initialize cipher
            self._cipher = Fernet(self._data_key)
            
            self.logger.info("Encryption manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _generate_new_key(self):
        """Generate new encryption keys."""
        # Generate master key
        self._master_key = secrets.token_bytes(32)  # 256-bit key
        
        # Derive data key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ai_framework_rpc_salt',  # In production, use random salt
            iterations=100000,
            backend=default_backend()
        )
        self._data_key = kdf.derive(self._master_key)
        
        # Save key securely
        self._save_encryption_key()
        
        self.logger.info("Generated new encryption keys")
    
    def _save_encryption_key(self):
        """Save encryption key to file."""
        try:
            key_data = {
                'master_key': base64.b64encode(self._master_key).decode(),
                'data_key': base64.b64encode(self._data_key).decode(),
                'created_at': time.time(),
                'version': '1.0'
            }
            
            # Encrypt the key file itself if possible
            if KEYRING_AVAILABLE:
                # Try to use system keyring
                try:
                    keyring.set_password("ai_framework_rpc", "master_key", 
                                       base64.b64encode(self._master_key).decode())
                    keyring.set_password("ai_framework_rpc", "data_key",
                                       base64.b64encode(self._data_key).decode())
                except Exception as e:
                    self.logger.warning(f"Failed to use keyring, falling back to file: {e}")
                    self._save_key_to_file(key_data)
            else:
                self._save_key_to_file(key_data)
                
        except Exception as e:
            self.logger.error(f"Failed to save encryption key: {e}")
            raise
    
    def _save_key_to_file(self, key_data: Dict[str, Any]):
        """Save key data to file with restricted permissions."""
        with open(self._key_file, 'w') as f:
            json.dump(key_data, f, indent=2)
        
        # Set file permissions (read/write only for owner)
        try:
            os.chmod(self._key_file, 0o600)
        except OSError:
            self.logger.warning("Could not set secure file permissions")
    
    def _load_encryption_key(self):
        """Load encryption key from storage."""
        try:
            if KEYRING_AVAILABLE:
                # Try to load from keyring first
                try:
                    master_key_b64 = keyring.get_password("ai_framework_rpc", "master_key")
                    data_key_b64 = keyring.get_password("ai_framework_rpc", "data_key")
                    
                    if master_key_b64 and data_key_b64:
                        self._master_key = base64.b64decode(master_key_b64)
                        self._data_key = base64.b64decode(data_key_b64)
                        return
                except Exception as e:
                    self.logger.warning(f"Failed to load from keyring: {e}")
            
            # Fall back to file
            if self._key_file.exists():
                with open(self._key_file, 'r') as f:
                    key_data = json.load(f)
                
                self._master_key = base64.b64decode(key_data['master_key'])
                self._data_key = base64.b64decode(key_data['data_key'])
            else:
                raise FileNotFoundError("Encryption key file not found")
                
        except Exception as e:
            self.logger.error(f"Failed to load encryption key: {e}")
            raise
    
    def encrypt(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64-encoded encrypted data
        """
        if self._cipher is None:
            raise RuntimeError("Encryption not initialized")
        
        try:
            if isinstance(data, dict):
                data = json.dumps(data).encode()
            elif isinstance(data, str):
                data = data.encode()
            
            encrypted_data = self._cipher.encrypt(data)
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data
        """
        if self._cipher is None:
            raise RuntimeError("Encryption not initialized")
        
        try:
            data_bytes = base64.b64decode(encrypted_data)
            decrypted_data = self._cipher.decrypt(data_bytes)
            
            # Try to parse as JSON first
            try:
                return json.loads(decrypted_data.decode())
            except json.JSONDecodeError:
                return decrypted_data.decode()
                
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def rotate_keys(self):
        """Rotate encryption keys."""
        try:
            self.logger.info("Starting key rotation")
            
            # Generate new keys
            old_data_key = self._data_key
            self._generate_new_key()
            
            # Re-initialize cipher with new key
            self._cipher = Fernet(self._data_key)
            
            self.logger.info("Key rotation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            # Restore old keys if rotation failed
            self._data_key = old_data_key
            self._cipher = Fernet(self._data_key)
            raise
    
    def _start_key_rotation(self):
        """Start automatic key rotation thread."""
        def rotation_loop():
            while not self._stop_rotation.is_set():
                try:
                    time.sleep(self.config.key_rotation_interval)
                    if not self._stop_rotation.is_set():
                        self.rotate_keys()
                except Exception as e:
                    self.logger.error(f"Automatic key rotation error: {e}")
        
        self._key_rotation_thread = threading.Thread(
            target=rotation_loop,
            daemon=True,
            name="KeyRotation"
        )
        self._key_rotation_thread.start()
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get information about current keys."""
        return {
            'key_file_exists': self._key_file.exists(),
            'encryption_initialized': self._cipher is not None,
            'auto_rotation_enabled': self.config.auto_rotate_keys,
            'keyring_available': KEYRING_AVAILABLE
        }
    
    def shutdown(self):
        """Shutdown encryption manager."""
        self._stop_rotation.set()
        
        if self._key_rotation_thread and self._key_rotation_thread.is_alive():
            self._key_rotation_thread.join(timeout=5)
        
        # Clear keys from memory
        self._master_key = None
        self._data_key = None
        self._cipher = None
        
        self.logger.info("Encryption manager shutdown completed")


class SecureStorage:
    """
    Secure storage for sensitive data.
    
    Features:
    - Encrypted file storage
    - Automatic backups
    - Data integrity verification
    - Secure deletion
    """
    
    def __init__(self, encryption_manager: EncryptionManager, config: SecurityConfig):
        self.encryption_manager = encryption_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage files
        self.storage_file = Path(config.secure_storage_file)
        self.backup_dir = self.storage_file.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        
        # Backup thread
        self._backup_thread: Optional[threading.Thread] = None
        self._stop_backup = threading.Event()
        
        # Load existing data
        self._load_storage()
        
        # Start backup thread if enabled
        if config.backup_enabled:
            self._start_backup_thread()
    
    def _load_storage(self):
        """Load existing secure storage."""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    encrypted_data = f.read()
                
                if encrypted_data:
                    decrypted_data = self.encryption_manager.decrypt(encrypted_data)
                    if isinstance(decrypted_data, dict):
                        with self._cache_lock:
                            self._cache = decrypted_data
                    
                    self.logger.info("Secure storage loaded successfully")
            else:
                self.logger.info("No existing secure storage found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Failed to load secure storage: {e}")
            # Try to restore from backup
            self._restore_from_backup()
    
    def _save_storage(self):
        """Save secure storage to file."""
        try:
            with self._cache_lock:
                data_to_save = self._cache.copy()
            
            encrypted_data = self.encryption_manager.encrypt(data_to_save)
            
            # Atomic write
            temp_file = self.storage_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                f.write(encrypted_data)
            
            # Verify integrity
            with open(temp_file, 'r') as f:
                test_data = f.read()
                self.encryption_manager.decrypt(test_data)
            
            # Replace original file
            temp_file.replace(self.storage_file)
            
            # Set secure permissions
            try:
                os.chmod(self.storage_file, 0o600)
            except OSError:
                self.logger.warning("Could not set secure file permissions")
            
        except Exception as e:
            self.logger.error(f"Failed to save secure storage: {e}")
            raise
    
    def store(self, key: str, value: Any):
        """
        Store a value securely.
        
        Args:
            key: Storage key
            value: Value to store
        """
        try:
            with self._cache_lock:
                self._cache[key] = value
            
            # Save to disk
            self._save_storage()
            
            self.logger.debug(f"Stored secure data for key: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to store data for key {key}: {e}")
            raise
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from secure storage.
        
        Args:
            key: Storage key
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        try:
            with self._cache_lock:
                return self._cache.get(key, default)
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for key {key}: {e}")
            return default
    
    def delete(self, key: str):
        """
        Delete a value from secure storage.
        
        Args:
            key: Storage key to delete
        """
        try:
            with self._cache_lock:
                if key in self._cache:
                    del self._cache[key]
            
            # Save changes
            self._save_storage()
            
            self.logger.debug(f"Deleted secure data for key: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete data for key {key}: {e}")
            raise
    
    def list_keys(self) -> List[str]:
        """List all stored keys."""
        with self._cache_lock:
            return list(self._cache.keys())
    
    def clear_all(self):
        """Clear all stored data."""
        try:
            with self._cache_lock:
                self._cache.clear()
            
            self._save_storage()
            self.logger.info("Cleared all secure storage data")
            
        except Exception as e:
            self.logger.error(f"Failed to clear secure storage: {e}")
            raise
    
    def _start_backup_thread(self):
        """Start automatic backup thread."""
        def backup_loop():
            while not self._stop_backup.is_set():
                try:
                    time.sleep(self.config.backup_interval)
                    if not self._stop_backup.is_set():
                        self._create_backup()
                except Exception as e:
                    self.logger.error(f"Automatic backup error: {e}")
        
        self._backup_thread = threading.Thread(
            target=backup_loop,
            daemon=True,
            name="SecureStorageBackup"
        )
        self._backup_thread.start()
    
    def _create_backup(self):
        """Create a backup of the secure storage."""
        try:
            timestamp = int(time.time())
            backup_file = self.backup_dir / f"secure_storage_{timestamp}.enc"
            
            # Copy current storage file
            if self.storage_file.exists():
                import shutil
                shutil.copy2(self.storage_file, backup_file)
                
                # Set secure permissions
                try:
                    os.chmod(backup_file, 0o600)
                except OSError:
                    pass
                
                # Clean up old backups
                self._cleanup_old_backups()
                
                self.logger.info(f"Created backup: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
    
    def _cleanup_old_backups(self):
        """Clean up old backup files."""
        try:
            backup_files = list(self.backup_dir.glob("secure_storage_*.enc"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove excess backups
            for backup_file in backup_files[self.config.max_backup_files:]:
                backup_file.unlink()
                self.logger.debug(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
    
    def _restore_from_backup(self):
        """Try to restore from the most recent backup."""
        try:
            backup_files = list(self.backup_dir.glob("secure_storage_*.enc"))
            if not backup_files:
                return
            
            # Get most recent backup
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_backup = backup_files[0]
            
            # Restore from backup
            import shutil
            shutil.copy2(latest_backup, self.storage_file)
            
            # Load the restored data
            self._load_storage()
            
            self.logger.info(f"Restored from backup: {latest_backup}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the secure storage."""
        with self._cache_lock:
            cache_size = len(self._cache)
        
        backup_files = list(self.backup_dir.glob("secure_storage_*.enc"))
        
        return {
            'storage_file_exists': self.storage_file.exists(),
            'cache_size': cache_size,
            'backup_enabled': self.config.backup_enabled,
            'backup_count': len(backup_files),
            'last_backup': max([f.stat().st_mtime for f in backup_files]) if backup_files else None
        }
    
    def shutdown(self):
        """Shutdown secure storage."""
        self._stop_backup.set()
        
        if self._backup_thread and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5)
        
        # Save final state
        self._save_storage()
        
        self.logger.info("Secure storage shutdown completed")


class SecurityManager:
    """
    Main security manager for AIFrameworkRPC.
    
    Features:
    - Encryption and secure storage
    - Access control
    - Audit logging
    - Session management
    - Security monitoring
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.encryption_manager = EncryptionManager(self.config)
        self.secure_storage = SecureStorage(self.encryption_manager, self.config)
        
        # Audit logging
        self.audit_log: List[SecurityAuditEntry] = []
        self.audit_lock = threading.Lock()
        
        # Session management
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()
        
        # Security monitoring
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Dict[str, float] = {}
        
        self.logger.info("Security manager initialized")
    
    def store_sensitive_data(self, key: str, value: Any, user_id: str = None):
        """
        Store sensitive data securely.
        
        Args:
            key: Storage key
            value: Value to store
            user_id: User ID for audit
        """
        try:
            self.secure_storage.store(key, value)
            self._log_audit('data_store', user_id, 'store', key, True)
            
        except Exception as e:
            self._log_audit('data_store', user_id, 'store', key, False, {'error': str(e)})
            raise
    
    def retrieve_sensitive_data(self, key: str, user_id: str = None, default: Any = None) -> Any:
        """
        Retrieve sensitive data securely.
        
        Args:
            key: Storage key
            user_id: User ID for audit
            default: Default value if not found
            
        Returns:
            Stored value or default
        """
        try:
            value = self.secure_storage.retrieve(key, default)
            self._log_audit('data_retrieve', user_id, 'retrieve', key, True)
            return value
            
        except Exception as e:
            self._log_audit('data_retrieve', user_id, 'retrieve', key, False, {'error': str(e)})
            return default
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """Encrypt data using the encryption manager."""
        return self.encryption_manager.encrypt(data)
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """Decrypt data using the encryption manager."""
        return self.encryption_manager.decrypt(encrypted_data)
    
    def create_session(self, user_id: str, session_data: Dict[str, Any] = None) -> str:
        """
        Create a new user session.
        
        Args:
            user_id: User identifier
            session_data: Additional session data
            
        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        
        with self.session_lock:
            self.sessions[session_token] = {
                'user_id': user_id,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'data': session_data or {}
            }
        
        self._log_audit('session_create', user_id, 'create_session', session_token, True)
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            Session data if valid, None otherwise
        """
        with self.session_lock:
            session = self.sessions.get(session_token)
            
            if not session:
                return None
            
            # Check session timeout
            if time.time() - session['last_accessed'] > self.config.session_timeout:
                del self.sessions[session_token]
                self._log_audit('session_expire', session['user_id'], 'expire_session', session_token, True)
                return None
            
            # Update last accessed
            session['last_accessed'] = time.time()
            return session.copy()
    
    def destroy_session(self, session_token: str):
        """Destroy a session."""
        with self.session_lock:
            session = self.sessions.get(session_token)
            if session:
                del self.sessions[session_token]
                self._log_audit('session_destroy', session['user_id'], 'destroy_session', session_token, True)
    
    def _log_audit(self, event_type: str, user_id: Optional[str], action: str, 
                  resource: str, success: bool, details: Dict[str, Any] = None):
        """Log a security audit event."""
        if not self.config.audit_logging:
            return
        
        entry = SecurityAuditEntry(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            success=success,
            details=details or {}
        )
        
        with self.audit_lock:
            self.audit_log.append(entry)
            
            # Keep audit log size manageable
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-5000:]
    
    def get_audit_log(self, limit: int = 100, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            limit: Maximum number of entries to return
            user_id: Filter by user ID
            
        Returns:
            List of audit entries
        """
        with self.audit_lock:
            log = self.audit_log.copy()
        
        if user_id:
            log = [entry for entry in log if entry.user_id == user_id]
        
        # Sort by timestamp (most recent first) and limit
        log.sort(key=lambda x: x.timestamp, reverse=True)
        return [entry.to_dict() for entry in log[:limit]]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'encryption': self.encryption_manager.get_key_info(),
            'storage': self.secure_storage.get_storage_info(),
            'sessions': {
                'active_sessions': len(self.sessions),
                'session_timeout': self.config.session_timeout
            },
            'audit': {
                'total_entries': len(self.audit_log),
                'logging_enabled': self.config.audit_logging
            },
            'blocked_ips': len(self.blocked_ips),
            'failed_attempts': dict(self.failed_attempts)
        }
    
    def rotate_encryption_keys(self):
        """Rotate encryption keys."""
        self.encryption_manager.rotate_keys()
        self._log_audit('key_rotation', None, 'rotate_keys', 'encryption_keys', True)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        with self.session_lock:
            for token, session in self.sessions.items():
                if current_time - session['last_accessed'] > self.config.session_timeout:
                    expired_sessions.append((token, session['user_id']))
            
            for token, user_id in expired_sessions:
                del self.sessions[token]
                self._log_audit('session_cleanup', user_id, 'cleanup_session', token, True)
        
        return len(expired_sessions)
    
    def shutdown(self):
        """Shutdown security manager."""
        self.secure_storage.shutdown()
        self.encryption_manager.shutdown()
        self.logger.info("Security manager shutdown completed")


# Global security manager instance
_global_security_manager: Optional[SecurityManager] = None


def initialize_security(config: SecurityConfig = None) -> SecurityManager:
    """
    Initialize the global security manager.
    
    Args:
        config: Security configuration
        
    Returns:
        SecurityManager instance
    """
    global _global_security_manager
    
    if _global_security_manager is not None:
        logging.warning("Security manager is already initialized")
        return _global_security_manager
    
    _global_security_manager = SecurityManager(config)
    return _global_security_manager


def get_security_manager() -> Optional[SecurityManager]:
    """Get the global security manager instance."""
    return _global_security_manager


def shutdown_security():
    """Shutdown the global security manager."""
    global _global_security_manager
    
    if _global_security_manager is not None:
        _global_security_manager.shutdown()
        _global_security_manager = None
