import psycopg2
from psycopg2.extras import RealDictCursor
from dev.config import DATABASE_URL

def get_conn():
    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )