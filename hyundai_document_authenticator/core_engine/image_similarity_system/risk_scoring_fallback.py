"""Fallback risk scoring utilities for base (non-Pro) deployments.

This module provides safe, no-op implementations of the risk scoring
functions used by TIF search workflows. It is intended to be imported only when
advanced scoring is not provisioned. The functions here must never raise and
should preserve output schema stability by returning neutral values.

Behavior
- compute_threshold_match_count: returns 0 to neutralize CSV column when the
  feature is unavailable.
- compute_fraud_probability: returns "Not Implemented" to indicate the feature
  is disabled without affecting flow or outputs.

The main workflow attempts to import from a premium module (risk_scoring) and
falls back to this module if unavailable. Additionally, the workflow guards
with in-module stubs if both modules are absent, ensuring the base edition can
remove this file entirely without breaking.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def compute_threshold_match_count(top_docs: List[Dict[str, float]], threshold: float) -> int:
    """Return a neutral threshold match count when risk scoring is disabled.

    This fallback intentionally returns 0 to preserve CSV/JSON output shape
    while clearly indicating that advanced matching metrics are not active.

    Args:
        top_docs (List[Dict[str, float]]): Aggregated top documents, each with a
            "score" key. Ignored in the fallback implementation.
        threshold (float): Score threshold used by premium scoring. Ignored.

    Returns:
        int: Always 0 in the fallback implementation.
    """
    return 0


def compute_fraud_probability(
    top_docs: List[Dict[str, float]],
    non_authentic_classes: List[str],
    image_auth_map: Optional[Dict[str, str]],
    similar_doc_flag_threshold: float,
) -> str:
    """Return a neutral fraud probability label when scoring is disabled.

    The fallback returns "Not Implemented" to convey that risk scoring is
    not provisioned in the current edition. This is stable for CSV/JSON/DB
    payloads and should not alter control flow in the pipeline.

    Args:
        top_docs (List[Dict[str, float]]): Aggregated top documents. Ignored.
        non_authentic_classes (List[str]): Classes considered non-authentic.
            Ignored.
        image_auth_map (Optional[Dict[str, str]]): Optional image authenticity
            mapping. Ignored.
        similar_doc_flag_threshold (float): Score threshold. Ignored.

    Returns:
        str: Always "Not Implemented" in the fallback implementation.
    """
    return "Not Implemented"
