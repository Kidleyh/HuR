"""Frame-batched top-down RTMPose engines for existing face and hand boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence

import cv2

from .engine import _extract_pose, _frame_to_person_refs
from .hand_metrics import (
    aggregate_hand_temporal, analyze_hand_side, associate_hands_to_wrists,
)
from .head_metrics import analyze_head_temporal
from .schema import HandTemporalConfig, PartTemporalConfig


class _PartPoseEngine:
    def __init__(self, config, device="cuda:0", *, model_loader=None, inference_fn=None):
        self.config = config
        if model_loader is None or inference_fn is None:
            try:
                from mmpose.apis import inference_topdown, init_model
            except ImportError as error:
                raise ImportError("Part Temporal requires MMPose") from error
            model_loader = model_loader or init_model
            inference_fn = inference_fn or inference_topdown
        self.model = model_loader(str(config.pose_config), str(config.pose_checkpoint), device=device)
        self._inference = inference_fn

    def _run_video(self, video_path, persons, category, pose_key):
        people = {int(person["logical_track_id"]): person for person in persons}
        refs = _frame_to_person_refs(persons)
        if not refs:
            return
        capture = cv2.VideoCapture(str(Path(video_path).expanduser().resolve()))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Temporal video cannot be opened: {video_path}")
        maximum = max(refs, default=-1)
        frame_index = 0
        try:
            while frame_index <= maximum:
                ok, image = capture.read()
                if not ok:
                    break
                targets = []
                bboxes = []
                for logical_id, person_frame_index in refs.get(frame_index, []):
                    frame = people[logical_id]["frames"][person_frame_index]
                    for part_index, part in enumerate(frame.get(category, [])):
                        targets.append((logical_id, person_frame_index, part_index))
                        bboxes.append(part["bbox_xyxy"])
                if bboxes:
                    results = list(self._inference(self.model, image, bboxes=bboxes, bbox_format="xyxy"))
                    if len(results) != len(targets):
                        raise RuntimeError(f"RTMPose {category} result count mismatch")
                    for target, result in zip(targets, results):
                        logical_id, person_frame_index, part_index = target
                        people[logical_id]["frames"][person_frame_index][category][part_index][pose_key] = _extract_pose(result)
                frame_index += 1
        finally:
            capture.release()
        missing = sorted(index for index in refs if index >= frame_index)
        if missing:
            raise RuntimeError(
                f"Temporal video ended before requested frame {missing[0]}: "
                f"{video_path}"
            )

    def close(self):
        self.model = None


class HeadTemporalEngine(_PartPoseEngine):
    def score_video(self, video_path: Path, persons: Sequence[MutableMapping[str, Any]]) -> None:
        self._run_video(video_path, persons, "faces", "face_pose")
        for person in persons:
            observations = []
            for frame in person["frames"]:
                faces = [face for face in frame.get("faces", []) if face.get("face_pose")]
                if not faces:
                    continue
                face = max(faces, key=lambda item: float(item.get("detector_score", 0.0)))
                observations.append({"frame_index": frame["frame_index"], **face})
            person.setdefault("temporal", {})["head"] = analyze_head_temporal(observations, self.config)


class HandTemporalEngine(_PartPoseEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.body_name_to_index: Dict[str, int] = {}

    def score_video(self, video_path: Path, persons: Sequence[MutableMapping[str, Any]]) -> None:
        self._run_video(video_path, persons, "hands", "hand_pose")
        for person in persons:
            human = person.get("temporal", {}).get("human", {})
            name_to_index = human.get("keypoint_name_to_index", self.body_name_to_index)
            by_side: Dict[str, List[Dict[str, Any]]] = {"left": [], "right": []}
            for frame in person["frames"]:
                association = associate_hands_to_wrists(frame, name_to_index, self.config)
                frame["hand_association"] = association
                for side, index in association.items():
                    if index is None:
                        continue
                    hand = frame["hands"][index]
                    hand["side"] = side
                    by_side[side].append({"frame_index": frame["frame_index"], **hand})
            left = analyze_hand_side(by_side["left"], self.config)
            right = analyze_hand_side(by_side["right"], self.config)
            person.setdefault("temporal", {})["hand"] = aggregate_hand_temporal(left, right)
