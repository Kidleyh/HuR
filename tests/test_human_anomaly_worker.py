import numpy as np
import pytest

from scripts.vbench_human_anomaly_worker import _process_person


ENTRY = {
    "frame_index": 0,
    "logical_track_id": 4,
    "source_track_id": 9,
    "bbox_xyxy": [1.0, 1.0, 15.0, 15.0],
}
FRAME = np.zeros((20, 20, 3), dtype=np.uint8)


class Analyzer:
    def __init__(self, fail_first_hand=False):
        self.fail_first_hand = fail_first_hand
        self.hand_calls = 0

    def smart_cut(self, frame, bbox):
        return frame

    def predict_batch(self, category, crops):
        if category == "hand":
            self.hand_calls += 1
            if self.fail_first_hand and self.hand_calls == 1:
                raise ValueError("bad hand crop")
        return [[0.1, 0.9]]


class Detector:
    def __init__(self, parts=None, error=None):
        self.parts = parts or []
        self.error = error

    def detect(self, image):
        if self.error is not None:
            raise self.error
        return self.parts


def part(label, offset):
    return {
        "label": label,
        "bbox_xyxy": [offset, offset, offset + 2, offset + 2],
        "detector_score": 0.8,
    }


def test_human_result_survives_face_hand_detection_failure():
    output = _process_person(
        Analyzer(), Detector(error=ValueError("detector rejected crop")), FRAME, ENTRY
    )
    assert output["human"]["scored"] is True
    assert output["human"]["scores"] == [0.1, 0.9]
    assert output["faces"] == [] and output["hands"] == []
    assert output["failures"][0]["stage"] == "face_hand_detection"
    assert "face_hand_detection" in output["failure_reason"]


def test_one_hand_failure_preserves_human_face_and_other_hand():
    output = _process_person(
        Analyzer(fail_first_hand=True),
        Detector(parts=[part(0, 1), part(1, 3), part(1, 5)]),
        FRAME,
        ENTRY,
    )
    assert output["human"]["scored"] is True
    assert len(output["faces"]) == 1
    assert len(output["hands"]) == 1
    assert output["failures"][0]["stage"] == "hand_scoring"
    assert output["person_abnormal"] is False


@pytest.mark.parametrize(
    "message", ["CUDA out of memory", "cuDNN execution failed", "device-side assert"]
)
def test_cuda_failures_are_raised(message):
    with pytest.raises(RuntimeError, match=message):
        _process_person(Analyzer(), Detector(error=RuntimeError(message)), FRAME, ENTRY)


def test_missing_face_and_hand_remain_neutral():
    output = _process_person(Analyzer(), Detector(parts=[]), FRAME, ENTRY)
    assert output["human"]["scored"] is True
    assert output["faces"] == [] and output["hands"] == []
    assert output["person_abnormal"] is False
    assert output["failures"] == []
    assert output["failure_reason"] is None
