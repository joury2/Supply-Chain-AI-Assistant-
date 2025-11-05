# app/services/knowledge_base_services/core/thread_safe_db.py
"""
Thread-safe SQLite connection manager for Streamlit
Fixes: SQLite objects created in a thread can only be used in that same thread
"""
import sqlite3
import threading
from typing import Optional
from contextlib import contextmanager  


class ThreadSafeDB:
    """
    Thread-safe SQLite connection manager
    Creates a new connection per thread automatically
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get connection for current thread
        Creates new connection if not exists
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Allow multi-thread access
            )
            self._local.connection.row_factory = sqlite3.Row
        
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """
        Context manager for safe cursor usage
        
        Usage:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def execute(self, query: str, params: tuple = ()):
        """
        Execute a query and return results
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: list):
        """
        Execute many queries with different parameters
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
    
    def close_all(self):
        """
        Close connection for current thread
        """
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Global instance (singleton)
_db_instances = {}


def get_thread_safe_db(db_path: str = "supply_chain.db") -> ThreadSafeDB:
    """
    Get thread-safe database instance (singleton per path)
    """
    if db_path not in _db_instances:
        _db_instances[db_path] = ThreadSafeDB(db_path)
    return _db_instances[db_path]