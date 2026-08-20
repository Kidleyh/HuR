from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import filter_small_clear_faces as faces


def _face(qualified=True):
    return {"qualified": qualified}


def test_area_ratio_and_original_crop_measurements():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[10:30, 20:60] = np.indices((20, 40)).sum(axis=0)[..., None] % 2 * 255
    result = faces.measure_face(
        image,
        {"bbox_xyxy": [20.0, 10.0, 60.0, 30.0], "detector_score": 0.9},
        faces.FilterThresholds(
            area_threshold=0.05,
            min_face_short_side=20,
            laplacian_threshold=0,
            tenengrad_threshold=0,
        ),
    )
    assert result["width_px"] == 40.0
    assert result["height_px"] == 20.0
    assert result["short_side_px"] == 20.0
    assert result["area_ratio"] == pytest.approx(0.04)
    assert result["qualified"] is True


def test_tile_origins_and_bbox_mapping_cover_trailing_edge():
    assert faces.tile_origins(250, 100, 0.2) == [0, 80, 150]
    mapped = faces.map_detection_to_image(
        {"label": 0, "bbox_xyxy": [5, 10, 40, 50], "detector_score": 0.8},
        x_offset=150,
        y_offset=30,
        image_width=180,
        image_height=100,
    )
    assert mapped == {
        "bbox_xyxy": [155.0, 40.0, 180.0, 80.0],
        "detector_score": 0.8,
    }
    assert faces.map_detection_to_image(
        {"label": 1, "bbox_xyxy": [0, 0, 10, 10], "detector_score": 1.0},
        0, 0, 100, 100,
    ) is None


def test_global_nms_deduplicates_whole_image_and_tile_faces():
    class Detector:
        def detect(self, image, score_threshold):
            return [
                {"label": 0, "bbox_xyxy": [10, 10, 40, 40],
                 "detector_score": 0.9},
                {"label": 1, "bbox_xyxy": [60, 60, 80, 80],
                 "detector_score": 0.99},
            ]

    detections = faces.detect_faces(
        np.zeros((80, 80, 3), dtype=np.uint8),
        Detector(),
        detector_threshold=0.1,
        tiled_inference=True,
        tile_size=1024,
        tile_overlap=0.15,
        whole_image_detection=True,
        nms_iou_threshold=0.5,
    )
    assert detections == [{
        "bbox_xyxy": [10.0, 10.0, 40.0, 40.0],
        "detector_score": 0.9,
    }]


def test_nms_keeps_best_overlapping_box_and_separate_box():
    result = faces.non_maximum_suppression([
        {"bbox_xyxy": [0, 0, 20, 20], "detector_score": 0.7},
        {"bbox_xyxy": [1, 1, 21, 21], "detector_score": 0.9},
        {"bbox_xyxy": [50, 50, 70, 70], "detector_score": 0.6},
    ], 0.5)
    assert [item["detector_score"] for item in result] == [0.9, 0.6]


def test_clarity_metrics_are_higher_for_sharp_checkerboard_than_blur():
    rows, columns = np.indices((64, 64))
    checker = (((rows // 8 + columns // 8) % 2) * 255).astype(np.uint8)
    checker = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(checker, (15, 15), 5)
    sharp_laplacian, sharp_tenengrad = faces.clarity_metrics(checker)
    blur_laplacian, blur_tenengrad = faces.clarity_metrics(blurred)
    assert sharp_laplacian > blur_laplacian
    assert sharp_tenengrad > blur_tenengrad


def test_image_and_video_selected_logic_uses_qualified_frame_count():
    frames = [
        {"frame_index": 0, "faces": [_face(True)]},
        {"frame_index": 5, "faces": [_face(False)]},
        {"frame_index": 10, "faces": [_face(True)]},
    ]
    image = faces.build_media_result(
        Path("image.jpg"), "image", 100, 100, frames[:1],
        min_qualified_frames=3,
    )
    video = faces.build_media_result(
        Path("video.mp4"), "video", 100, 100, frames,
        min_qualified_frames=2,
    )
    insufficient = faces.build_media_result(
        Path("video.mp4"), "video", 100, 100, frames[:2],
        min_qualified_frames=2,
    )
    assert image["selected"] is True
    assert video["selected"] is True
    assert video["qualified_frame_count"] == 2
    assert insufficient["selected"] is False


def _measured_face(bbox, qualified):
    return {
        "bbox_xyxy": bbox,
        "detector_score": 0.91,
        "area_ratio": 0.004,
        "short_side_px": 40.0,
        "laplacian_variance": 150.0,
        "tenengrad": 1400.0,
        "qualified": qualified,
    }


def test_visualization_draws_qualified_green_and_unqualified_yellow():
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    rendered = faces.draw_face_visualization(image, {
        "frame_index": 0,
        "faces": [
            _measured_face([20, 50, 80, 120], True),
            _measured_face([140, 50, 210, 120], False),
        ],
    })
    assert tuple(rendered[120, 50]) == faces.QUALIFIED_COLOR
    assert tuple(rendered[120, 175]) == faces.UNQUALIFIED_COLOR
    assert np.count_nonzero(image) == 0


def test_visualization_paths_preserve_relative_layout(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "visualizations"
    first = faces.visualization_path_for_media(
        input_root / "a" / "sample.mp4", input_root, output_root, "video"
    )
    second = faces.visualization_path_for_media(
        input_root / "b" / "sample.mp4", input_root, output_root, "video"
    )
    assert first == output_root / "a" / "sample.mp4"
    assert second == output_root / "b" / "sample.mp4"
    assert first != second


def test_image_visualization_is_readable_and_keeps_dimensions(tmp_path):
    source = tmp_path / "source.jpg"
    destination = tmp_path / "out" / "source.jpg"
    original = np.zeros((72, 128, 3), dtype=np.uint8)
    assert cv2.imwrite(str(source), original)
    result = {
        "media_type": "image",
        "frames": [{"frame_index": 0, "faces": [
            _measured_face([20, 20, 70, 60], True)
        ]}],
    }
    assert faces.write_image_visualization(source, result, destination) == destination
    rendered = cv2.imread(str(destination))
    assert rendered is not None
    assert rendered.shape == original.shape


def test_video_visualization_keeps_frames_dimensions_and_fps(tmp_path):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "out" / "source.mp4"
    width, height, fps = 128, 72, 12.0
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()
    result = {
        "media_type": "video",
        "frames": [{"frame_index": 0, "faces": [
            _measured_face([20, 20, 70, 60], True)
        ]}],
    }
    assert faces.write_video_visualization(source, result, destination) == destination
    capture = cv2.VideoCapture(str(destination))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(fps, abs=0.2)
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
