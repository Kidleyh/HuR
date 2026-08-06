"""Validation of VBench worker records before HuR aggregation."""

from __future__ import annotations

from typing import Any, Set, Sequence, Tuple

from .schema import HumanAnomalyInput

PersonFrameKey = Tuple[int, int]


def _valid_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_worker_results(
    entries: Sequence[HumanAnomalyInput], frame_results: Sequence[Any]
) -> None:
    """Require an exact one-to-one worker result for every manifest entry."""
    if not entries:
        raise ValueError("No valid person-frame entries were produced from stitching results")

    expected: Set[PersonFrameKey] = {
        (entry.frame_index, entry.logical_track_id) for entry in entries
    }
    if len(expected) != len(entries):
        raise ValueError("Input manifest contains duplicate person-frame keys")

    actual: Set[PersonFrameKey] = set()
    for index, result in enumerate(frame_results):
        if not isinstance(result, dict):
            raise ValueError(f"Worker result at index {index} is not a JSON object")
        frame_index = result.get("frame_index")
        logical_track_id = result.get("logical_track_id")
        if not _valid_id(frame_index) or not _valid_id(logical_track_id):
            raise ValueError(
                f"Worker result at index {index} has invalid frame_index or logical_track_id"
            )
        key = (frame_index, logical_track_id)
        if key in actual:
            raise ValueError(f"Worker output contains duplicate person-frame key: {key}")
        actual.add(key)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if len(frame_results) != len(entries) or missing or extra:
        raise ValueError(
            "Worker result keys do not match input manifest: "
            f"expected={len(entries)}, actual={len(frame_results)}, "
            f"missing={missing}, extra={extra}"
        )

    if not any(
        isinstance(result.get("human"), dict)
        and result["human"].get("scored") is True
        for result in frame_results
    ):
        raise ValueError("Worker produced no scored person frames")
