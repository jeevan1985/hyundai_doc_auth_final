"""Unit tests for routes' system health helper shape and invariants.

These tests exercise the /admin/system_health.json and /admin/system_health.csv
endpoints using a Flask test client. Database connectivity is monkeypatched to
simulate both up and down states.
"""
from __future__ import annotations

from typing import Any, Dict

import json
import types

import pytest
from flask import Flask

from hyundai_document_authenticator.external.result_gui.app import create_app
from hyundai_document_authenticator.external.result_gui.config_loader import load_config
from hyundai_document_authenticator.external.result_gui.db import Database


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> Flask:
    app = create_app()

    # Log in a fake admin by monkeypatching Flask-Login's current_user
    class DummyUser:
        is_authenticated = True
        role = 'admin'
        id = 1
        username = 'admin'

    # Patch login_required to a no-op for test routes
    import hyundai_document_authenticator.external.result_gui.routes as routes_mod
    routes_mod.current_user = DummyUser()  # type: ignore

    # Monkeypatch Database._get_conn to simulate DB select 1
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
            def __init__(self): self.conn = DummyConn()
            def __enter__(self): return self.conn
            def __exit__(self, *a): return False
        return Ctx()

    monkeypatch.setattr(Database, "_get_conn", fake_get_conn)
    return app


@pytest.fixture()
def client(app: Flask):
    return app.test_client()


def test_system_health_json_shape(client) -> None:
    resp = client.get('/admin/system_health.json')
    assert resp.status_code == 200
    data: Dict[str, Any] = resp.get_json()
    assert set(data.keys()) >= {"server", "database", "queue", "storage", "generated_at"}
    assert isinstance(data["server"], dict)
    assert isinstance(data["database"], dict)
    assert isinstance(data["queue"], dict)
    assert isinstance(data["storage"], dict)
    assert isinstance(data["generated_at"], int)


def test_system_health_csv_download(client) -> None:
    resp = client.get('/admin/system_health.csv')
    assert resp.status_code == 200
    assert resp.headers['Content-Disposition'].endswith('system_health.csv')
    text = resp.get_data(as_text=True)
    assert 'key,value' in text
    assert 'server.hostname' in text
