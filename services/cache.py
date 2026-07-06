import os
import json
import hashlib
from typing import Any, Optional
from pathlib import Path

CACHE_DIR = Path(os.getenv("PAPERGUARD_CACHE_DIR", ".cache"))

def _get_cache_path(key: str, namespace: str) -> Path:
    """Generate a safe file path for the given key and namespace."""
    os.makedirs(CACHE_DIR / namespace, exist_ok=True)
    # Hash the key to avoid invalid filename characters
    safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / namespace / f"{safe_key}.json"

def get(key: str, namespace: str = "default") -> Optional[Any]:
    """Retrieve an item from the cache. Returns None if not found."""
    cache_path = _get_cache_path(key, namespace)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # If the cache file is corrupted, just return None
            return None
    return None

def set(key: str, data: Any, namespace: str = "default") -> None:
    """Store an item in the cache."""
    cache_path = _get_cache_path(key, namespace)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write cache for {key} in {namespace}: {e}")

def clear(namespace: Optional[str] = None) -> None:
    """Clear the cache. If namespace is provided, only clear that namespace."""
    if namespace:
        ns_dir = CACHE_DIR / namespace
        if ns_dir.exists():
            for f in ns_dir.glob("*.json"):
                f.unlink()
    else:
        for ns_dir in CACHE_DIR.glob("*"):
            if ns_dir.is_dir():
                for f in ns_dir.glob("*.json"):
                    f.unlink()
