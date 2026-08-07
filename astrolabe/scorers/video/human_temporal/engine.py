"""Frame-batched top-down RTMPose inference using existing HuR person boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

import cv2
import numpy as np

from .metrics import analyze_person_temporal
from .schema import HumanTemporalConfig


def keypoint_name_to_index(dataset_meta: Mapping[str, Any]) -> Dict[str, int]:
    """Read MMPose dataset metadata without assuming fixed COCO indices."""
    name_to_id = dataset_meta.get("keypoint_name2id")
    if isinstance(name_to_id, Mapping):
        return {str(name): int(index) for name, index in name_to_id.items()}
    id_to_name = dataset_meta.get("keypoint_id2name")
    if isinstance(id_to_name, Mapping):
        return {str(name): int(index) for index, name in id_to_name.items()}
    keypoint_info = dataset_meta.get("keypoint_info")
    if isinstance(keypoint_info, Mapping):
        parsed = {}
        for fallback_index, info in keypoint_info.items():
            if isinstance(info, Mapping) and "name" in info:
                parsed[str(info["name"])] = int(info.get("id", fallback_index))
        if parsed:
            return parsed
    raise ValueError("RTMPose dataset_meta does not define keypoint names")


def _extract_pose(sample: Any) -> Dict[str, List[Any]]:
    if sample is None:
        return {"keypoints_xy": [], "keypoint_scores": []}
    if isinstance(sample, Mapping):
        keypoints = sample.get("keypoints_xy", sample.get("keypoints", []))
        scores = sample.get("keypoint_scores", [])
    else:
        instances = getattr(sample, "pred_instances", None)
        if instances is None:
            return {"keypoints_xy": [], "keypoint_scores": []}
        keypoints = getattr(instances, "keypoints", [])
        scores = getattr(instances, "keypoint_scores", [])
    keypoints_array = np.asarray(keypoints, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    while keypoints_array.ndim > 2:
        keypoints_array = keypoints_array[0]
    while scores_array.ndim > 1:
        scores_array = scores_array[0]
    if (
        keypoints_array.ndim != 2
        or keypoints_array.shape[-1:] != (2,)
        or scores_array.ndim != 1
        or len(keypoints_array) != len(scores_array)
    ):
        return {"keypoints_xy": [], "keypoint_scores": []}
    return {
        "keypoints_xy": keypoints_array.astype(float).tolist(),
        "keypoint_scores": scores_array.astype(float).tolist(),
    }


def _frame_to_person_refs(
    persons: Sequence[Mapping[str, Any]],
) -> Dict[int, List[tuple[int, int]]]:
    refs: Dict[int, List[tuple[int, int]]] = {}
    for person in persons:
        logical_id = int(person["logical_track_id"])
        for person_frame_index, frame in enumerate(person["frames"]):
            refs.setdefault(int(frame["frame_index"]), []).append(
                (logical_id, person_frame_index)
            )
    for frame_refs in refs.values():
        frame_refs.sort()
    return refs


class HumanTemporalEngine:
    """Load one RTMPose model and score multiple videos using HuR boxes only."""

    def __init__(
        self,
        config: HumanTemporalConfig,
        device: str = "cuda:0",
        *,
        model_loader: Optional[Callable[..., Any]] = None,
        inference_fn: Optional[Callable[..., Sequence[Any]]] = None,
    ) -> None:
        self.config = config
        if model_loader is None or inference_fn is None:
            try:
                from mmpose.apis import inference_topdown, init_model
            except ImportError as error:
                raise ImportError(
                    "Human Temporal requires MMPose in the active environment"
                ) from error
            model_loader = model_loader or init_model
            inference_fn = inference_fn or inference_topdown
        self.model = model_loader(
            str(config.pose_config), str(config.pose_checkpoint), device=device
        )
        self._inference = inference_fn
        self.keypoint_name_to_index = keypoint_name_to_index(
            getattr(self.model, "dataset_meta", {})
        )

    def score_video(
        self, video_path: Path, persons: Sequence[MutableMapping[str, Any]]
    ) -> None:
        """Attach original-coordinate poses and temporal metrics in place."""
        source = Path(video_path).expanduser().resolve()
        people = {int(person["logical_track_id"]): person for person in persons}
        frame_refs = _frame_to_person_refs(persons)
        if not frame_refs:
            return
        maximum_frame = max(frame_refs)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Human Temporal video cannot be opened: {source}")
        decoded_targets = set()
        frame_index = 0
        try:
            while frame_index <= maximum_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                refs = frame_refs.get(frame_index, [])
                if refs:
                    bboxes = [
                        people[logical_id]["frames"][person_frame_index]["bbox_xyxy"]
                        for logical_id, person_frame_index in refs
                    ]
                    pose_results = list(self._inference(
                        self.model, frame, bboxes=bboxes, bbox_format="xyxy"
                    ))
                    if len(pose_results) != len(refs):
                        raise RuntimeError(
                            "RTMPose result count does not match HuR bbox count: "
                            f"{len(pose_results)} != {len(refs)} at frame {frame_index}"
                        )
                    for reference, pose_result in zip(refs, pose_results):
                        logical_id, person_frame_index = reference
                        people[logical_id]["frames"][person_frame_index][
                            "human_pose"
                        ] = _extract_pose(pose_result)
                    decoded_targets.add(frame_index)
                frame_index += 1
        finally:
            capture.release()
        missing = sorted(set(frame_refs) - decoded_targets)
        if missing:
            raise RuntimeError(
                f"Human Temporal could not decode requested frames: {missing[:10]}"
            )
        for person in persons:
            person.setdefault("temporal", {})["human"] = analyze_person_temporal(
                person, self.keypoint_name_to_index, self.config
            )

    def close(self) -> None:
        self.model = None
