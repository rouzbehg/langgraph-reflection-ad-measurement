from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Iterable

try:
    from langsmith import traceable
except ImportError:
    traceable = None

try:
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    get_current_run_tree = None


def traced(name: str, run_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if traceable is None:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper

        return decorator
    return traceable(name=name, run_type=run_type)  # type: ignore[return-value]


def attach_run_metadata(metadata: Dict[str, Any], tags: Iterable[str]) -> None:
    if get_current_run_tree is None:
        return
    run_tree = get_current_run_tree()
    if run_tree is None:
        return

    extra = dict(getattr(run_tree, "extra", {}) or {})
    existing_metadata = dict(extra.get("metadata", {}) or {})
    existing_metadata.update(metadata)
    extra["metadata"] = existing_metadata
    run_tree.extra = extra

    existing_tags = list(getattr(run_tree, "tags", []) or [])
    for tag in tags:
        if tag not in existing_tags:
            existing_tags.append(tag)
    run_tree.tags = existing_tags
