"""
Risk scoring utilities for the Image Similarity System.

This module centralizes small, pure computations related to risk/flagging so they can
be reused consistently across the workflow and easily unit tested. Implementations
mirror the current behavior exactly and do not change thresholds, semantics, or
output values.

Functions
- compute_threshold_match_count: Count per-query top document scores passing a threshold.
- compute_fraud_probability: Decide fraud probability label from similarity and authenticity evidence.

Notes
- Threshold constants are configured in constants.py. Callers must pass the value they
  currently use (e.g., constants.SIMILAR_DOC_FLAG_THRESHOLD for CSV threshold_match_count,
  or YAML-configured threshold for fraud probability decisions).
"""
from __future__ import annotations

from typing import Dict, List, Optional


def compute_threshold_match_count(top_docs: List[Dict[str, float]], threshold: float) -> int:
    """Count how many top document entries meet or exceed a score threshold.

    This is a pure function without side effects. It preserves legacy behavior by:
    - Treating missing/None/invalid scores as 0.0
    - Using a >= comparison against the provided threshold

    Args:
        top_docs: A list of entries like {"document": str, "score": float}.
        threshold: The numeric threshold to compare scores against.

    Returns:
        The number of entries with score >= threshold.

    Raises:
        None: All conversion errors are handled defensively by coercing to 0.0.
    """
    count = 0
    for td in top_docs or []:
        try:
            score = float(td.get("score") or 0.0)
            if score >= float(threshold):
                count += 1
        except Exception:
            # Be defensive: treat any issue as 0.0 and continue
            continue
    return int(count)


def compute_fraud_probability(
    top_docs: List[Dict[str, float]],
    non_authentic_classes: List[str],
    image_auth_map: Optional[Dict[str, str]],
    similar_doc_flag_threshold: float,
) -> str:
    """Compute the fraud probability label given similarity and authenticity evidence.

    Behavior (kept identical to legacy rules):
    - has_high_similarity := any(score >= similar_doc_flag_threshold for score in top_docs)
    - has_non_authentic_photo := any(class in non_authentic_classes) when image_auth_map provided, else False
    - Return "Very_High" if both true; "High" if either; otherwise "No"

    Args:
        top_docs: Per-query aggregated top documents, with fields {"document", "score"}.
        non_authentic_classes: Labels that, when detected on any photo, count as non-authentic.
        image_auth_map: Optional mapping from stable photo ID to predicted class name.
        similar_doc_flag_threshold: Threshold for the similarity condition.

    Returns:
        A string label among {"Very_High", "High", "No"}.

    Exceptions:
        None: Any conversion errors are handled defensively.
    """
    # Similarity check
    try:
        has_high_similarity = any(
            float((td or {}).get("score") or 0.0) >= float(similar_doc_flag_threshold)
            for td in (top_docs or [])
        )
    except Exception:
        has_high_similarity = False

    # Authenticity check
    has_non_authentic_photo = False
    if image_auth_map:
        try:
            na_set = set(non_authentic_classes or [])
            for _pid, cls in (image_auth_map or {}).items():
                if cls in na_set:
                    has_non_authentic_photo = True
                    break
        except Exception:
            has_non_authentic_photo = False

    if has_high_similarity and has_non_authentic_photo:
        return "Very_High"
    if has_high_similarity or has_non_authentic_photo:
        return "High"
    return "No"
