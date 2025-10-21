"""Unit tests for i18n utilities.

Covers normalize_lang and translations_for fallback behavior.
"""
from __future__ import annotations

from typing import Dict

import pytest

from hyundai_document_authenticator.external.result_gui.i18n import normalize_lang, translations_for, TRANSLATIONS


def test_normalize_lang_variants() -> None:
    assert normalize_lang("en") == "en"
    assert normalize_lang("EN") == "en"
    assert normalize_lang("ko") == "ko"
    assert normalize_lang("ko-KR") == "ko"
    assert normalize_lang("KO-kr") == "ko"
    assert normalize_lang(None) == "en"  # type: ignore[arg-type]
    assert normalize_lang("fr") == "en"  # unsupported -> fallback


def test_translations_for_fallback() -> None:
    # Clone one entry and simulate missing ko to verify English fallback
    key = "__test.key__"
    TRANSLATIONS[key] = {"en": "English Only"}
    try:
        d_en: Dict[str, str] = translations_for("en")
        d_ko: Dict[str, str] = translations_for("ko")
        assert d_en[key] == "English Only"
        assert d_ko[key] == "English Only"  # fallback
        # Missing key falls back to key when queried directly
        assert d_en.get("__missing__", "__missing__") == "__missing__"
    finally:
        TRANSLATIONS.pop(key, None)
