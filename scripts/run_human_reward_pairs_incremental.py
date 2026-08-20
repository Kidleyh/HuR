#!/usr/bin/env python3
"""Score selected GT/render pairs with an atomic checkpoint after each pair."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrolabe.scorers.video.human_reward import HumanRewardModel
from astrolabe.scorers.video.human_reward.pair_evaluation import (
    build_pair_frame_evaluation,
    build_pair_score_summary,
)
from scripts import run_human_reward_pairs as base

LOGGER = logging.getLogger("human_reward_pairs_incremental")
SCHEMA_VERSION = "1.0"
FULL_RESULT_FILENAME = "human_reward_pairs_incremental_full.json"
SCORES_RESULT_FILENAME = "human_reward_pairs_incremental_scores.json"
FRAME_EVALUATION_FILENAME = (
    "human_reward_pair_incremental_frame_evaluation.json"
)
PAIR_EVALUATION_FILENAME = "human_reward_pair_incremental_evaluation.json"
PROGRESS_FILENAME = "human_reward_pairs_incremental_progress.json"
GENERATED_FILENAMES = (
    FULL_RESULT_FILENAME,
    SCORES_RESULT_FILENAME,
    FRAME_EVALUATION_FILENAME,
    PAIR_EVALUATION_FILENAME,
    PROGRESS_FILENAME,
)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read incremental checkpoint {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Incremental checkpoint root must be an object: {path}")
    return data


def _load_resume_pairs(
    output_dir: Path,
    *,
    selection_manifest: Optional[Path],
    input_root: Path,
    selected_pairs: Sequence[base.VideoPair],
) -> List[Dict[str, Any]]:
    full_path = output_dir / FULL_RESULT_FILENAME
    if not full_path.is_file():
        return []
    checkpoint = _load_json(full_path)
    expected_manifest = (
        str(selection_manifest.expanduser().resolve())
        if selection_manifest is not None else None
    )
    if checkpoint.get("selection_manifest") != expected_manifest:
        raise ValueError(
            "Existing checkpoint selection_manifest does not match this run"
        )
    if checkpoint.get("input_dir") != str(input_root.resolve()):
        raise ValueError("Existing checkpoint input_dir does not match this run")
    completed = checkpoint.get("pairs")
    if not isinstance(completed, list):
        raise ValueError("Existing checkpoint pairs must be a list")
    expected_names = [pair.name for pair in selected_pairs[:len(completed)]]
    actual_names = [
        pair.get("name") if isinstance(pair, dict) else None for pair in completed
    ]
    if actual_names != expected_names:
        raise ValueError(
            "Existing checkpoint is not an ordered prefix of selected pairs"
        )
    return completed


def _build_full_checkpoint(
    *,
    input_root: Path,
    selection_manifest: Optional[Path],
    requested_pair_count: int,
    completed_pairs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "incremental_per_pair",
        "input_dir": str(input_root.resolve()),
        "selection_manifest": (
            str(selection_manifest.expanduser().resolve())
            if selection_manifest is not None else None
        ),
        "requested_pair_count": requested_pair_count,
        "completed_pair_count": len(completed_pairs),
        "pair_count": len(completed_pairs),
        "video_count": 2 * len(completed_pairs),
        "complete": len(completed_pairs) == requested_pair_count,
        "pairs": list(completed_pairs),
    }


def write_incremental_checkpoint(
    *,
    output_dir: Path,
    input_root: Path,
    selection_manifest: Optional[Path],
    requested_pair_count: int,
    completed_pairs: Sequence[Dict[str, Any]],
    tie_epsilon: float,
) -> Dict[str, Any]:
    """Atomically replace each generated artifact after one pair completes."""
    full = _build_full_checkpoint(
        input_root=input_root,
        selection_manifest=selection_manifest,
        requested_pair_count=requested_pair_count,
        completed_pairs=completed_pairs,
    )
    scores = base.build_scores_result(full)
    frame_evaluation = build_pair_frame_evaluation(
        full, tie_epsilon=tie_epsilon
    )
    frame_evaluation.update({
        "mode": "incremental_per_pair",
        "requested_pair_count": requested_pair_count,
        "completed_pair_count": len(completed_pairs),
        "complete": len(completed_pairs) == requested_pair_count,
        "selection_manifest": full["selection_manifest"],
    })
    evaluation = build_pair_score_summary(frame_evaluation)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if full["complete"] else "running",
        "requested_pair_count": requested_pair_count,
        "completed_pair_count": len(completed_pairs),
        "remaining_pair_count": requested_pair_count - len(completed_pairs),
        "last_completed_pair": (
            completed_pairs[-1]["name"] if completed_pairs else None
        ),
        "dataset_metrics": evaluation["dataset_metrics"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        (FULL_RESULT_FILENAME, full),
        (SCORES_RESULT_FILENAME, scores),
        (FRAME_EVALUATION_FILENAME, frame_evaluation),
        (PAIR_EVALUATION_FILENAME, evaluation),
        (PROGRESS_FILENAME, progress),
    ):
        base.write_json_atomic(output_dir / filename, payload)
    return progress


def _selected_input(args: Any) -> Tuple[Path, List[base.VideoPair], Optional[Path]]:
    if args.selection_manifest:
        manifest = Path(args.selection_manifest).expanduser().resolve()
        input_root, pairs = base.load_selected_video_pairs(manifest)
        return input_root, pairs, manifest
    input_root = Path(args.input_dir).expanduser().resolve()
    return input_root, base.discover_video_pairs(input_root), None


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an ordered-prefix checkpoint in the same output directory",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    if args.tie_epsilon < 0:
        parser.error("--tie-epsilon must be non-negative")
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output must be a directory path: {output_dir}")
    input_root, pairs, selection_manifest = _selected_input(args)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        parser.error("No video pairs were selected")

    existing_generated = [
        output_dir / name for name in GENERATED_FILENAMES
        if (output_dir / name).exists()
    ]
    if existing_generated and not args.resume:
        parser.error(
            "Incremental output already exists; choose a new --output directory "
            "or pass --resume: " + str(output_dir)
        )
    if args.resume and existing_generated and not (
        output_dir / FULL_RESULT_FILENAME
    ).is_file():
        parser.error(
            "Cannot resume because the full incremental checkpoint is missing: "
            + str(output_dir / FULL_RESULT_FILENAME)
        )
    completed_pairs = (
        _load_resume_pairs(
            output_dir,
            selection_manifest=selection_manifest,
            input_root=input_root,
            selected_pairs=pairs,
        ) if args.resume else []
    )
    if len(completed_pairs) > len(pairs):
        parser.error("Checkpoint contains more pairs than selected by this run")

    model = HumanRewardModel(base._config_from_args(args))
    requested = len(pairs)
    LOGGER.info(
        "Incremental scoring: completed=%d requested=%d output=%s",
        len(completed_pairs), requested, output_dir,
    )
    for index in range(len(completed_pairs), requested):
        pair = pairs[index]
        LOGGER.info("Scoring pair %d/%d: %s", index + 1, requested, pair.name)
        results = model.score_batch([pair.positive, pair.negative])
        paired = base.build_paired_result(input_root, [pair], results)["pairs"][0]
        if args.visualization_dir:
            base.write_pair_visualizations(
                [pair], results, Path(args.visualization_dir)
            )
        completed_pairs.append(paired)
        progress = write_incremental_checkpoint(
            output_dir=output_dir,
            input_root=input_root,
            selection_manifest=selection_manifest,
            requested_pair_count=requested,
            completed_pairs=completed_pairs,
            tie_epsilon=args.tie_epsilon,
        )
        LOGGER.info(
            "Checkpointed %d/%d pairs; current accuracy file=%s",
            progress["completed_pair_count"], requested,
            output_dir / PAIR_EVALUATION_FILENAME,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
