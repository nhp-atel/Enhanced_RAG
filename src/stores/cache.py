"""
Caching implementations with multiple backends and automatic eviction.
"""

import os
import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, asdict
from collections import OrderedDict
import threading

from ..interfaces import CacheInterface
from ..utils.errors import CacheError
from ..utils.logging import get_logger


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0


class FileCacheBackend(CacheInterface):
    """File-based cache implementation with LRU eviction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = Path(config.get('directory', './data/cache'))
        self.max_size_mb = config.get('max_size_mb', 1024)
        self.ttl_seconds = config.get('ttl_seconds', 3600)
        self.eviction_policy = config.get('eviction_policy', 'lru')
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata: Dict[str, Dict[str, Any]] = self._load_metadata()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger
        self.logger = get_logger("file_cache")
        
        # Initialize cache
        self._cleanup_expired_entries()
        self._enforce_size_limit()
        
        self.logger.info("File cache initialized", 
                        cache_dir=str(self.cache_dir),
                        max_size_mb=self.max_size_mb,
                        entry_count=len(self.metadata))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            try:
                # Check if key exists and not expired
                if not self._is_valid_entry(key):
                    return None
                
                # Load value from file
                file_path = self._get_file_path(key)
                if not file_path.exists():
                    self._remove_metadata(key)
                    return None
                
                # Deserialize value
                try:
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                except (pickle.PickleError, EOFError):
                    # Try JSON fallback
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            value = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to deserialize cache entry", key=key)
                        self._remove_entry(key)
                        return None
                
                # Update access metadata
                self._update_access_metadata(key)
                
                return value
                
            except Exception as e:
                self.logger.error("Error getting cache entry", key=key, error=str(e))
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            try:
                # Calculate expiration
                current_time = time.time()
                expires_at = None
                if ttl is not None:
                    expires_at = current_time + ttl
                elif self.ttl_seconds > 0:
                    expires_at = current_time + self.ttl_seconds
                
                # Serialize value
                file_path = self._get_file_path(key)
                serialized_successfully = False
                
                try:
                    # Try pickle first (supports more types)
                    with open(file_path, 'wb') as f:
                        pickle.dump(value, f)
                    serialized_successfully = True
                except (pickle.PickleError, TypeError):
                    # Fallback to JSON
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(value, f, default=str, ensure_ascii=False)
                        serialized_successfully = True
                    except (TypeError, ValueError) as e:
                        self.logger.warning("Failed to serialize cache value", 
                                          key=key, error=str(e))
                        return False
                
                if not serialized_successfully:
                    return False
                
                # Calculate size
                size_bytes = file_path.stat().st_size
                
                # Update metadata
                self.metadata[key] = {
                    'created_at': current_time,
                    'expires_at': expires_at,
                    'access_count': 0,
                    'last_accessed': current_time,
                    'size_bytes': size_bytes
                }
                
                # Save metadata
                self._save_metadata()
                
                # Enforce size limits
                self._enforce_size_limit()
                
                return True
                
            except Exception as e:
                self.logger.error("Error setting cache entry", key=key, error=str(e))
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self._lock:
            try:
                return self._remove_entry(key)
            except Exception as e:
                self.logger.error("Error deleting cache entry", key=key, error=str(e))
                return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob('cache_*'):
                    if cache_file.is_file():
                        cache_file.unlink()
                
                # Clear metadata
                self.metadata.clear()
                self._save_metadata()
                
                self.logger.info("Cache cleared")
                return True
                
            except Exception as e:
                self.logger.error("Error clearing cache", error=str(e))
                return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            return self._is_valid_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            current_time = time.time()
            total_size = sum(entry.get('size_bytes', 0) for entry in self.metadata.values())
            expired_count = sum(1 for entry in self.metadata.values() 
                              if entry.get('expires_at') and entry['expires_at'] < current_time)
            
            return {
                'backend': 'file',
                'entry_count': len(self.metadata),
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'hit_rate': self._calculate_hit_rate(),
                'expired_entries': expired_count,
                'cache_directory': str(self.cache_dir),
                'eviction_policy': self.eviction_policy
            }
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Hash the key to create safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f'cache_{key_hash}.dat'
    
    def _is_valid_entry(self, key: str) -> bool:
        """Check if cache entry is valid (exists and not expired)"""
        if key not in self.metadata:
            return False
        
        entry = self.metadata[key]
        current_time = time.time()
        
        # Check expiration
        if entry.get('expires_at') and entry['expires_at'] < current_time:
            self._remove_entry(key)
            return False
        
        return True
    
    def _update_access_metadata(self, key: str) -> None:
        """Update access metadata for LRU tracking"""
        if key in self.metadata:
            self.metadata[key]['access_count'] += 1
            self.metadata[key]['last_accessed'] = time.time()
    
    def _remove_entry(self, key: str) -> bool:
        """Remove cache entry and file"""
        try:
            # Remove file
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
            
            # Remove metadata
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
            
            return True
            
        except Exception as e:
            self.logger.error("Error removing cache entry", key=key, error=str(e))
            return False
    
    def _remove_metadata(self, key: str) -> None:
        """Remove only metadata (file already gone)"""
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
    
    def _cleanup_expired_entries(self) -> None:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.metadata.items():
            if entry.get('expires_at') and entry['expires_at'] < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            self.logger.info("Cleaned up expired entries", count=len(expired_keys))
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limits using eviction policy"""
        total_size = sum(entry.get('size_bytes', 0) for entry in self.metadata.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Calculate how much to remove (remove 20% extra to avoid frequent evictions)
        target_size = max_size_bytes * 0.8
        bytes_to_remove = total_size - target_size
        
        if self.eviction_policy == 'lru':
            self._evict_lru(bytes_to_remove)
        elif self.eviction_policy == 'lfu':
            self._evict_lfu(bytes_to_remove)
        else:
            self._evict_fifo(bytes_to_remove)
    
    def _evict_lru(self, bytes_to_remove: float) -> None:
        """Evict entries using Least Recently Used policy"""
        # Sort by last accessed time
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        bytes_removed = 0
        for key, entry in sorted_entries:
            if bytes_removed >= bytes_to_remove:
                break
            
            bytes_removed += entry.get('size_bytes', 0)
            self._remove_entry(key)
        
        self.logger.info("LRU eviction completed", 
                        bytes_removed=bytes_removed,
                        entries_removed=len([e for e in sorted_entries if bytes_removed >= e[1].get('size_bytes', 0)]))
    
    def _evict_lfu(self, bytes_to_remove: float) -> None:
        """Evict entries using Least Frequently Used policy"""
        # Sort by access count
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('access_count', 0)
        )
        
        bytes_removed = 0
        for key, entry in sorted_entries:
            if bytes_removed >= bytes_to_remove:
                break
            
            bytes_removed += entry.get('size_bytes', 0)
            self._remove_entry(key)
        
        self.logger.info("LFU eviction completed", bytes_removed=bytes_removed)
    
    def _evict_fifo(self, bytes_to_remove: float) -> None:
        """Evict entries using First In First Out policy"""
        # Sort by creation time
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('created_at', 0)
        )
        
        bytes_removed = 0
        for key, entry in sorted_entries:
            if bytes_removed >= bytes_to_remove:
                break
            
            bytes_removed += entry.get('size_bytes', 0)
            self._remove_entry(key)
        
        self.logger.info("FIFO eviction completed", bytes_removed=bytes_removed)
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning("Failed to load cache metadata", error=str(e))
        
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error("Failed to save cache metadata", error=str(e))
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder)"""
        # This would require tracking hits/misses in a real implementation
        return 0.0


class MemoryCacheBackend(CacheInterface):
    """In-memory cache implementation with LRU eviction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size_mb = config.get('max_size_mb', 256)
        self.ttl_seconds = config.get('ttl_seconds', 3600)
        self.eviction_policy = config.get('eviction_policy', 'lru')
        
        # Storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger
        self.logger = get_logger("memory_cache")
        
        self.logger.info("Memory cache initialized", 
                        max_size_mb=self.max_size_mb)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            current_time = time.time()
            if entry.expires_at and entry.expires_at < current_time:
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                return None
            
            # Update access info for LRU
            entry.access_count += 1
            entry.last_accessed = current_time
            
            # Move to end for LRU
            self.cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            try:
                # Calculate expiration
                current_time = time.time()
                expires_at = None
                if ttl is not None:
                    expires_at = current_time + ttl
                elif self.ttl_seconds > 0:
                    expires_at = current_time + self.ttl_seconds
                
                # Estimate size
                size_bytes = self._estimate_size(value)
                
                # Remove existing entry if present
                if key in self.cache:
                    old_entry = self.cache[key]
                    self.current_size_bytes -= old_entry.size_bytes
                    del self.cache[key]
                
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=current_time,
                    expires_at=expires_at,
                    access_count=0,
                    last_accessed=current_time,
                    size_bytes=size_bytes
                )
                
                # Add to cache
                self.cache[key] = entry
                self.current_size_bytes += size_bytes
                
                # Enforce size limits
                self._enforce_size_limit()
                
                return True
                
            except Exception as e:
                self.logger.error("Error setting cache entry", key=key, error=str(e))
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            current_time = time.time()
            
            # Check expiration
            if entry.expires_at and entry.expires_at < current_time:
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                return False
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'backend': 'memory',
                'entry_count': len(self.cache),
                'total_size_mb': self.current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'hit_rate': 0.0,  # Would need tracking
                'eviction_policy': self.eviction_policy
            }
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            # Use pickle to get a rough size estimate
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            return len(str(value)) * 2  # Rough estimate
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limits"""
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if self.current_size_bytes <= max_size_bytes:
            return
        
        # Remove oldest entries (LRU)
        while self.current_size_bytes > max_size_bytes * 0.8 and self.cache:
            key, entry = self.cache.popitem(last=False)  # Remove oldest
            self.current_size_bytes -= entry.size_bytes


def create_cache_backend(config: Dict[str, Any]) -> CacheInterface:
    """Factory function to create cache backend"""
    backend = config.get('backend', 'file').lower()
    
    if backend == 'file':
        return FileCacheBackend(config)
    elif backend == 'memory':
        return MemoryCacheBackend(config)
    elif backend == 'redis':
        # TODO: Implement Redis backend
        raise NotImplementedError("Redis cache backend not implemented yet")
    else:
        raise ValueError(f"Unknown cache backend: {backend}")


# Convenience wrapper with automatic key generation

class SmartCache:
    """Smart cache wrapper with automatic key generation and content hashing"""
    
    def __init__(self, backend: CacheInterface, prefix: str = ""):
        self.backend = backend
        self.prefix = prefix
        self.logger = get_logger("smart_cache")
    
    def get_or_compute(self, key_parts: List[str], compute_func: callable, ttl: Optional[int] = None) -> Any:
        """Get value from cache or compute it"""
        cache_key = self._generate_key(key_parts)
        
        # Try to get from cache
        cached_value = self.backend.get(cache_key)
        if cached_value is not None:
            self.logger.debug("Cache hit", key=cache_key)
            return cached_value
        
        # Compute value
        self.logger.debug("Cache miss, computing value", key=cache_key)
        computed_value = compute_func()
        
        # Store in cache
        self.backend.set(cache_key, computed_value, ttl)
        
        return computed_value
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern (for file backend)"""
        # This would require extending the interface to support pattern matching
        # For now, just return 0
        return 0
    
    def _generate_key(self, parts: List[str]) -> str:
        """Generate cache key from parts"""
        key_content = "|".join(str(part) for part in parts)
        key_hash = hashlib.md5(key_content.encode()).hexdigest()
        
        if self.prefix:
            return f"{self.prefix}:{key_hash}"
        return key_hash