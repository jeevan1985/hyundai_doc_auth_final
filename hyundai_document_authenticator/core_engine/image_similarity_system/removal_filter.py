"""
Removal filter utilities for Image Similarity outputs.

This module provides a single source of truth for column/key suppression rules
used when preparing CSV headers, summary JSON, and DB payload decisions. The
semantics are kept identical to legacy behavior and controlled by the active
removal set constructed from configuration.

Functions:
- effective_removal_set: Merge user-provided and default removal lists to a set.
- filter_per_query_entry: Remove keys from a per-query JSON entry, including the
  special rule that if 'top_similar_docs' is removed then 'top_docs' is also
  removed from the entry.
- filter_per_query_list: Apply filter_per_query_entry on a list while preserving
  ordering and unrelated keys.
- filter_csv_headers: Compute final CSV headers (base + extra) honoring the
  removal set and preserving input order.
- decide_global_top_docs_arg: Compute the argument for CSV writers given removal
  set and save flag.
- decide_sim_img_check_arg: Decide sim_img_check payload availability given
  removal set and privacy gating.

Notes:
- threshold_match_count participates in the same removal filtering semantics for
  CSV as any other column. Its calculation still occurs within the workflow
  logic; this module only suppresses its appearance in the CSV headers if the
  key is removed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


def effective_removal_set(remove_columns: Optional[List[str]], default_remove: List[str]) -> Set[str]:
    """Return the effective set of columns/keys to remove from outputs.

    Args:
        remove_columns: Optional list from configuration (search_task.remove_columns_from_results).
        default_remove: Default removal list (from constants.DEFAULT_REMOVE_COLUMNS_FROM_RESULTS).

    Returns:
        A set with the union of provided and default removals. Order is irrelevant.
    """
    user_list = list(remove_columns or [])
    default_list = list(default_remove or [])
    return set(user_list + default_list)


def filter_per_query_entry(entry: Dict[str, Any], removal_set: Set[str]) -> Dict[str, Any]:
    """Filter keys from one per-query entry for the summary JSON.

    Behavior:
    - Remove any keys that appear in removal_set.
    - Special case: if 'top_similar_docs' is in removal_set, also remove the
      internal 'top_docs' key to avoid leaking that information to JSON.

    Args:
        entry: The original per-query entry dict.
        removal_set: Set of keys to remove.

    Returns:
        A shallow-copied and filtered dict.
    """
    e = dict(entry or {})
    for k in list(e.keys()):
        if k in removal_set:
            e.pop(k, None)
    if 'top_similar_docs' in removal_set:
        e.pop('top_docs', None)
    return e


def filter_per_query_list(entries: List[Dict[str, Any]], removal_set: Set[str]) -> List[Dict[str, Any]]:
    """Filter a list of per-query entries for the summary JSON.

    Args:
        entries: List of per-query dict entries.
        removal_set: Set of keys to remove.

    Returns:
        Filtered list preserving original ordering.
    """
    return [filter_per_query_entry(e, removal_set) for e in (entries or [])]


def filter_csv_headers(base_headers: List[str], extra_headers: List[str], removal_set: Set[str]) -> List[str]:
    """Return final CSV headers honoring removal semantics.

    The order of 'base_headers' followed by 'extra_headers' is preserved while
    dropping any header name that appears in removal_set.

    Args:
        base_headers: Fixed base headers in their canonical order.
        extra_headers: Additional headers configured externally (key enrichment).
        removal_set: Set of headers to suppress.

    Returns:
        Filtered header list with preserved ordering.
    """
    filtered_base = [h for h in (base_headers or []) if h not in (removal_set or set())]
    filtered_extra = [h for h in (extra_headers or []) if h not in (removal_set or set())]
    return filtered_base + filtered_extra


def decide_global_top_docs_arg(global_names: List[str], save_flag: bool, removal_set: Set[str]) -> Optional[List[str]]:
    """Decide the argument to pass for the 'global_top_docs' CSV column.

    Behavior (must match legacy):
    - If 'global_top_docs' is in removal_set, return None (column omitted).
    - Else return global_names if save_flag is True; otherwise an empty list [].

    Args:
        global_names: Global top document names from the run aggregation.
        save_flag: search_task.save_global_top_docs boolean flag.
        removal_set: Active removal set.

    Returns:
        None to omit the column entirely, or a list (possibly empty) to include it.
    """
    if 'global_top_docs' in (removal_set or set()):
        return None
    return list(global_names or []) if bool(save_flag) else []


def decide_sim_img_check_arg(
    sim_checks_map: Optional[Dict[str, Any]],
    removal_set: Set[str],
    privacy_allowed: bool,
) -> Optional[Dict[str, Any]]:
    """Decide the per-query sim_img_check payload based on removal and privacy.

    Behavior (must match legacy):
    - If 'sim_img_check' is in removal_set OR privacy_allowed is False, return None.
    - Otherwise, return the provided map or an empty dict {} if None.

    Args:
        sim_checks_map: Map built by build_sim_img_checks_map or equivalent.
        removal_set: Active removal set.
        privacy_allowed: Whether privacy gating allows including this payload.

    Returns:
        None to omit, or a dict payload (possibly empty).
    """
    if (not privacy_allowed) or ('sim_img_check' in (removal_set or set())):
        return None
    return dict(sim_checks_map or {})
