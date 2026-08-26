import os
import redis.asyncio as redis
from typing import Optional

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/2")

class RedisClient:
    _instance: Optional[redis.Redis] = None

    @classmethod
    def get_instance(cls) -> redis.Redis:
        if cls._instance is None:
            # connection_pool is created automatically by from_url
            # max_connections ensures we don't exhaust redis connections under load
            cls._instance = redis.from_url(
                REDIS_URL, 
                decode_responses=True, 
                max_connections=20
            )
        return cls._instance

    @classmethod
    async def close(cls):
        if cls._instance:
            await cls._instance.close()
            cls._instance = None

async def get_redis() -> redis.Redis:
    return RedisClient.get_instance()
