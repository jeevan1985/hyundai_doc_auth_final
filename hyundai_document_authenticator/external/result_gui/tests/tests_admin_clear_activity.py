"""Tests for the admin activity clear endpoint.

Covers RBAC, valid selections, invalid selections, and redirect behavior.
"""
from __future__ import annotations

from typing import Any

import types
import pytest
from flask import Flask

from hyundai_document_authenticator.external.result_gui.app import create_app
from hyundai_document_authenticator.external.result_gui.db import Database


@pytest.fixture()
def app_admin(monkeypatch: pytest.MonkeyPatch) -> Flask:
    app = create_app()

    # Fake an authenticated admin user via monkeypatching current_user in routes
    class DummyAdmin:
        is_authenticated = True
        role = 'admin'
        id = 1
        username = 'admin'

    import hyundai_document_authenticator.external.result_gui.routes as routes_mod
    routes_mod.current_user = DummyAdmin()  # type: ignore

    # Ensure DB connection context works for system health
    def fake_get_conn(self):  # type: ignore[no-redef]
        class DummyConn:
            def cursor(self):
                class C:
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                    def execute(self, *a, **k): pass
                    def fetchone(self): return (1,)
                return C()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        class Ctx:
            def __enter__(self): return DummyConn()
            def __exit__(self, *a): return False
        return Ctx()

    monkeypatch.setattr(Database, "_get_conn", fake_get_conn)
    return app


@pytest.fixture()
def client_admin(app_admin: Flask):
    return app_admin.test_client()


def test_clear_activity_invalid_selection(monkeypatch: pytest.MonkeyPatch, client_admin) -> None:
    # Stub the clear method to ensure it is NOT called on invalid input
    called = {"count": 0}

    def fake_clear(self, days: int | None = None, today: bool = False) -> int:
        called["count"] += 1
        return 0

    monkeypatch.setattr(Database, "clear_activity_logs", fake_clear)

    resp = client_admin.post('/admin/activity/clear', data={"range_choice": "bad_value"}, follow_redirects=False)
    assert resp.status_code in (302, 303)
    # Ensure redirect back to /admin
    assert '/admin' in resp.headers.get('Location', '')
    # Ensure DB clear method was not invoked
    assert called["count"] == 0


def test_clear_activity_all(monkeypatch: pytest.MonkeyPatch, client_admin) -> None:
    # Stub the clear call and verify parameters
    calls: list[tuple[Any, ...]] = []

    def fake_clear(self, days: int | None = None, today: bool = False) -> int:
        calls.append((days, today))
        return 5

    monkeypatch.setattr(Database, "clear_activity_logs", fake_clear)

    resp = client_admin.post('/admin/activity/clear', data={"range_choice": "all"}, follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert '/admin' in resp.headers.get('Location', '')
    assert calls == [(None, False)]


def test_clear_activity_today(monkeypatch: pytest.MonkeyPatch, client_admin) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_clear(self, days: int | None = None, today: bool = False) -> int:
        calls.append((days, today))
        return 2

    monkeypatch.setattr(Database, "clear_activity_logs", fake_clear)

    resp = client_admin.post('/admin/activity/clear', data={"range_choice": "today"}, follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert '/admin' in resp.headers.get('Location', '')
    assert calls == [(None, True)]


def test_clear_activity_days_valid(monkeypatch: pytest.MonkeyPatch, client_admin) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_clear(self, days: int | None = None, today: bool = False) -> int:
        calls.append((days, today))
        return 12

    monkeypatch.setattr(Database, "clear_activity_logs", fake_clear)

    resp = client_admin.post('/admin/activity/clear', data={"range_choice": "days_7"}, follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert '/admin' in resp.headers.get('Location', '')
    assert calls == [(7, False)]


def test_clear_activity_days_out_of_range(monkeypatch: pytest.MonkeyPatch, client_admin) -> None:
    # Simulate DB raising ValueError for out-of-range days
    def fake_clear(self, days: int | None = None, today: bool = False) -> int:
        raise ValueError('days must be between 1 and 31')

    monkeypatch.setattr(Database, "clear_activity_logs", fake_clear)

    resp = client_admin.post('/admin/activity/clear', data={"range_choice": "days_99"}, follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert '/admin' in resp.headers.get('Location', '')


@pytest.fixture()
def app_viewer(monkeypatch: pytest.MonkeyPatch) -> Flask:
    app = create_app()

    # Fake a non-admin authenticated user
    class DummyViewer:
        is_authenticated = True
        role = 'viewer'
        id = 2
        username = 'viewer'

    import hyundai_document_authenticator.external.result_gui.routes as routes_mod
    routes_mod.current_user = DummyViewer()  # type: ignore

    monkeypatch.setattr(Database, "_get_conn", lambda self: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: False))
    return app


@pytest.fixture()
def client_viewer(app_viewer: Flask):
    return app_viewer.test_client()


def test_clear_activity_forbidden_for_viewer(client_viewer) -> None:
    resp = client_viewer.post('/admin/activity/clear', data={"range_choice": "all"})
    assert resp.status_code == 403
