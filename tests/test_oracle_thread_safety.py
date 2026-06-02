"""Tests for H-6: thread-safety locking in oraclemcp.

Tests the locking mechanism only — no Oracle connection needed.
"""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def test_db_lock_is_threading_lock():
    from tools.oraclemcp.oraclemcp_fastmcp import _db_lock
    assert isinstance(_db_lock, type(threading.Lock()))


def test_db_lock_is_context_manager():
    from tools.oraclemcp.oraclemcp_fastmcp import _db_lock
    with _db_lock:
        pass


def test_lock_released_after_context():
    from tools.oraclemcp.oraclemcp_fastmcp import _db_lock
    with _db_lock:
        pass
    acquired = _db_lock.acquire(blocking=False)
    assert acquired, "Lock was not released after context manager exit"
    _db_lock.release()


def test_concurrent_dict_access_with_lock():
    results = {}
    errors = []
    lock = threading.Lock()

    def writer(key, value, delay):
        try:
            time.sleep(delay)
            with lock:
                results[key] = value
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(f"k{i}", f"v{i}", 0.001))
        for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"Errors during concurrent access: {errors}"
    assert len(results) == 20


def test_concurrent_get_db_connection_no_crash():
    from tools.oraclemcp.oraclemcp_fastmcp import get_db_connection
    errors = []

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    call_count = 0

    def fake_connect(**kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.002)
        return mock_conn

    with patch("tools.oraclemcp.oraclemcp_fastmcp.oracledb") as mock_oracle, \
         patch("tools.oraclemcp.oraclemcp_fastmcp.connection", None), \
         patch.dict("os.environ", {
             "USERID": "user/pass",
             "DB_HOST": "host",
             "DB_PORT": "1521",
             "DB_SERVICE_NAME": "svc",
         }):
        mock_oracle.DatabaseError = Exception
        mock_oracle.makedsn.return_value = "dsn"
        mock_oracle.connect.side_effect = fake_connect

        def attempt():
            try:
                get_db_connection()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=attempt)
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert not errors, f"Errors during concurrent get_db_connection: {errors}"
    assert call_count >= 1


def test_clear_cache_is_thread_safe():
    from tools.oraclemcp.oraclemcp_fastmcp import clear_cache
    errors = []

    with patch("tools.oraclemcp.oraclemcp_fastmcp.table_columns_cache", {"a": 1}), \
         patch("tools.oraclemcp.oraclemcp_fastmcp.schema_cache", {"b": 2}):

        def attempt():
            try:
                clear_cache({})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=attempt)
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    assert not errors, f"Errors during concurrent clear_cache: {errors}"
