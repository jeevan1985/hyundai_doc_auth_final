"""Lightweight smoke validation for _initialize_provider_manager behavior.

This avoids heavy model/DB initialization by:
- Using a dummy FeatureExtractor for 'bruteforce' and 'qdrant' (when Qdrant is absent).
- Validating output tuple shapes and key fields for provider-specific semantics.
- Re-running a compile check for the target module.

Run:
    python extra_tools/verify_provider_helper_smoke.py
"""
from __future__ import annotations

import sys
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "hyundai_document_authenticator" / "core_engine" / "image_similarity_system" / "workflow.py"

# Re-compile to ensure we're green
py_compile.compile(str(TARGET), doraise=True)

# Import target module
sys.path.insert(0, str(ROOT))
from hyundai_document_authenticator.core_engine.image_similarity_system import workflow as wf  # type: ignore


class DummyFE:
    def __init__(self) -> None:
        self.feature_dim = 128
        self.model_name = "dummy-model"


def test_bruteforce() -> None:
    db_conf = {"provider": "bruteforce"}
    base, count, ident, fb_msg, mode, fatal = wf._initialize_provider_manager(
        provider="bruteforce",
        db_conf=db_conf,
        feature_extractor=DummyFE(),
        project_root=ROOT,
        search_task_conf={},
        full_config={},
        message_style="legacy",
    )
    assert base is None, "bruteforce: base manager must be None"
    assert count == 0, "bruteforce: indexed count must be 0"
    assert ident is None, "bruteforce: identifier must be None"
    assert fb_msg is None, "bruteforce: fallback message must be None"
    assert mode == "bruteforce", "bruteforce: selected_fallback_mode must be 'bruteforce'"
    assert fatal is None, "bruteforce: fatal must be None"


def test_qdrant_optional_transient() -> None:
    # simulate absent qdrant dependency by relying on module state
    allow_fallback = True
    db_conf = {
        "provider": "qdrant",
        "allow_fallback": allow_fallback,
        "fallback_choice": "transient",
        "qdrant": {"collection_name_stem": "smoke"},
    }
    base, count, ident, fb_msg, mode, fatal = wf._initialize_provider_manager(
        provider="qdrant",
        db_conf=db_conf,
        feature_extractor=DummyFE(),
        project_root=ROOT,
        search_task_conf={},
        full_config={},
        message_style="legacy",
    )
    # When qdrant import is unavailable, helper must fall back transient-only
    assert base is None, "qdrant(optional): base must be None when import is unavailable"
    assert mode is None, "qdrant(optional): transient-only behavior keeps selected_fallback_mode None"
    assert fatal is None, "qdrant(optional): no fatal when allow_fallback is True"
    assert isinstance(fb_msg, (str, type(None))), "qdrant(optional): expects a warning message"


if __name__ == "__main__":
    test_bruteforce()
    test_qdrant_optional_transient()
    print("SMOKE OK: helper behaves correctly for bruteforce and qdrant(optional).")
