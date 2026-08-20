"""Lazy in-process adapter from HuR logical tracks to official GVHMR APIs."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence

import cv2
import numpy as np

from .metrics import analyze_3d_temporal
from .schema import (
    GVHMRSequence,
    GVHMRTemporalConfig,
    failed_human_3d_result,
)


def _is_cuda_failure(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return any(token in text for token in (
        "cuda out of memory", "cuda error", "device-side assert", "cudnn",
    ))


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _OfficialGVHMRBackend:
    """Narrow adapter around the public GVHMR demo model/preprocessors."""

    def __init__(self, config: GVHMRTemporalConfig, device: str) -> None:
        if not device.startswith("cuda"):
            raise ValueError("Official GVHMR inference currently requires a CUDA device")
        self.config = config
        self.device = device
        root_text = str(config.gvhmr_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            import hydra
            import torch
            from hydra import compose, initialize_config_module
            from hmr4d.configs import register_store_gvhmr
            from hmr4d.utils.preproc import Extractor, VitPoseExtractor
        except ImportError as error:
            raise RuntimeError(
                "GVHMR dependencies are unavailable; install the official GVHMR "
                "runtime in the active environment"
            ) from error
        self._torch = torch
        self._hydra = hydra
        torch.cuda.set_device(torch.device(device))
        with _working_directory(config.gvhmr_root):
            register_store_gvhmr()
            with initialize_config_module(
                version_base="1.3", config_module="hmr4d.configs"
            ):
                cfg = compose(config_name="demo", overrides=["static_cam=True"])
            self.model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            self.model.load_pretrained_model(str(config.checkpoint))
            self.model = self.model.eval().to(device)
            self.vitpose = VitPoseExtractor(tqdm_leave=False)
            self.extractor = Extractor(tqdm_leave=False)
            self.vitpose.pose = self.vitpose.pose.to(device)
            self.extractor.extractor = self.extractor.extractor.to(device)

    def infer(
        self,
        rgb_frames: np.ndarray,
        bbox_xyxy: np.ndarray,
        frame_indices: np.ndarray,
        width: int,
        height: int,
    ) -> GVHMRSequence:
        torch = self._torch
        from hmr4d.utils.geo.hmr_cam import (
            estimate_K,
            get_bbx_xys_from_xyxy,
        )
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.preproc.vitfeat_extractor import get_batch

        torch.cuda.set_device(torch.device(self.device))
        boxes = torch.as_tensor(bbox_xyxy, dtype=torch.float32)
        bbx_xys = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
        images, _ = get_batch(
            rgb_frames, bbx_xys, img_ds=1.0, path_type="np"
        )
        with _working_directory(self.config.gvhmr_root):
            kp2d = self.vitpose.extract(images, bbx_xys)
            features = self.extractor.extract_video_features(images, bbx_xys)
        length = len(frame_indices)
        rotation = torch.eye(3).repeat(length, 1, 1)
        data = {
            "length": torch.tensor(length),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": estimate_K(width, height).repeat(length, 1, 1),
            "cam_angvel": compute_cam_angvel(rotation),
            "f_imgseq": features,
        }
        with torch.no_grad():
            prediction = self.model.predict(data, static_cam=True)
            params_gpu = {
                key: value.to(self.device)[None]
                for key, value in prediction["smpl_params_global"].items()
            }
            joints = self.model.pipeline.endecoder.fk_v2(**params_gpu)[0]
        params = {
            key: value.detach().cpu().numpy()
            for key, value in prediction["smpl_params_global"].items()
        }
        return GVHMRSequence(
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            joints_3d=joints.detach().cpu().numpy(),
            root_translation=params["transl"],
            smpl_params=params,
        )

    def close(self) -> None:
        self.model = None
        self.vitpose = None
        self.extractor = None


class GVHMRTemporalEngine:
    """Attach Human Temporal V2 results to person-centric HuR results."""

    def __init__(
        self,
        config: GVHMRTemporalConfig,
        device: str = "cuda:0",
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self._backend = (backend_factory or _OfficialGVHMRBackend)(config, device)

    @staticmethod
    def _read_needed_frames(
        video_path: Path, needed: Sequence[int]
    ) -> Dict[int, np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Video cannot be opened for GVHMR: {video_path}")
        targets = set(int(index) for index in needed)
        frames: Dict[int, np.ndarray] = {}
        index = 0
        try:
            while targets:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in targets:
                    frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    targets.remove(index)
                index += 1
        finally:
            capture.release()
        if targets:
            raise RuntimeError(
                f"Video ended before requested GVHMR frames: {sorted(targets)[:10]}"
            )
        return frames

    def score_video(
        self,
        video_path: Path,
        persons: Sequence[MutableMapping[str, Any]],
        *,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        needed = sorted({
            int(frame["frame_index"])
            for person in persons for frame in person.get("frames", [])
        })
        decoded = self._read_needed_frames(Path(video_path), needed)
        for person in persons:
            person_frames = sorted(
                person.get("frames", []), key=lambda item: item["frame_index"]
            )
            temporal = person.setdefault("temporal", {})
            try:
                if not person_frames:
                    raise ValueError("Logical person has no observed frames")
                frame_indices = np.asarray(
                    [frame["frame_index"] for frame in person_frames],
                    dtype=np.int64,
                )
                boxes = np.asarray(
                    [frame["bbox_xyxy"] for frame in person_frames],
                    dtype=np.float32,
                )
                images = np.stack([decoded[int(index)] for index in frame_indices])
                sequence = self._backend.infer(
                    images, boxes, frame_indices, width, height
                )
                temporal["human_3d"] = analyze_3d_temporal(
                    sequence,
                    fps=fps,
                    total_observed_frames=len(person_frames),
                    min_valid_joints=self.config.min_valid_joints,
                )
            except Exception as error:
                if _is_cuda_failure(error):
                    raise
                temporal["human_3d"] = failed_human_3d_result(
                    len(person_frames), error
                )

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()
        self._backend = None
