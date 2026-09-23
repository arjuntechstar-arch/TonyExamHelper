from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from app.core.config import get_settings


@lru_cache
def get_mongo_client() -> MongoClient:
    return MongoClient(get_settings().mongodb_url, serverSelectionTimeoutMS=5_000, tz_aware=True)


def get_database() -> Database:
    return get_mongo_client()[get_settings().mongodb_database]


def close_mongo_client() -> None:
    get_mongo_client.cache_clear()
