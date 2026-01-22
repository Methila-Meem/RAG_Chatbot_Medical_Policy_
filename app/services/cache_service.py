import redis
import json
import hashlib
from typing import Optional
from app.config import get_settings

settings = get_settings()

class CacheService:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                decode_responses=True
            )
            self.redis_client.ping()
            self.enabled = True
        except:
            print("Redis not available, caching disabled")
            self.enabled = False
    
    def _generate_key(self, question: str) -> str:
        """Generate cache key from question"""
        return f"rag:{hashlib.md5(question.encode()).hexdigest()}"
    
    def get(self, question: str) -> Optional[dict]:
        """Get cached response"""
        if not self.enabled:
            return None
        
        try:
            key = self._generate_key(question)
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except:
            pass
        return None
    
    def set(self, question: str, response: dict):
        """Cache response"""
        if not self.enabled:
            return
        
        try:
            key = self._generate_key(question)
            self.redis_client.setex(
                key,
                settings.cache_expiry,
                json.dumps(response)
            )
        except:
            pass
    
    def is_connected(self) -> bool:
        """Check Redis connection"""
        if not self.enabled:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False