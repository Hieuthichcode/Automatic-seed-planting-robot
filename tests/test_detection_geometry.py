import numpy as np
import pytest

import detect_stream


@pytest.mark.parametrize(
    ("image_shape", "model_shape", "box"),
    [
        ((1086, 1448), (480, 480), (103, 207, 901, 807)),
        ((480, 640), (480, 480), (50, 40, 500, 420)),
        ((900, 500), (640, 640), (40, 100, 430, 800)),
        ((720, 1280), (512, 512), (200, 80, 1100, 650)),
    ],
)
def test_letterbox_coordinates_round_trip(image_shape, model_shape, box):
    image = np.zeros((*image_shape, 3), dtype=np.uint8)
    padded, ratio, pad = detect_stream.letterbox(image, model_shape)
    assert padded.shape[:2] == model_shape

    x1, y1, x2, y2 = box
    model_box = [
        ((x1 + x2) / 2) * ratio + pad[0],
        ((y1 + y2) / 2) * ratio + pad[1],
        (x2 - x1) * ratio,
        (y2 - y1) * ratio,
    ]
    restored = detect_stream.scale_coords(
        model_box, ratio, pad, image_shape
    )

    assert restored == pytest.approx(box, abs=1)


def test_camera_inference_keeps_its_source_frame(monkeypatch):
    source = np.full((8, 12, 3), 37, dtype=np.uint8)
    expected_output = np.zeros((1, 7), dtype=np.float32)

    monkeypatch.setattr(
        detect_stream,
        "infer_on_image",
        lambda frame: (expected_output, 0.5, (2, 3), 0.01),
    )
    result = detect_stream._infer_camera_frame(source)

    assert result[0] is source
    assert result[1] is expected_output
    assert result[2:] == (0.5, (2, 3), 0.01)


def test_tflite_input_is_converted_from_bgr_to_rgb():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    tensor = detect_stream._prepare_model_input(image)

    assert tensor.shape == (1, 2, 3, 3)
    assert tensor[0, 0, 0] == pytest.approx(
        np.array([30, 20, 10], dtype=np.float32) / 255.0
    )


def test_yolov8_output_is_transposed_to_one_row_per_prediction():
    raw_output = np.zeros((1, 6, 8400), dtype=np.float32)
    raw_output[0, 4, 123] = 0.75

    output = detect_stream._decode_model_output(raw_output)

    assert output.shape == (8400, 6)
    assert output[123, 4] == pytest.approx(0.75)


def test_yolov8_detection_uses_class_confidence_without_objectness():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = np.array([
        [200 / 640, 280 / 640, 200 / 640, 200 / 640, 0.1, 0.9]
    ], dtype=np.float32)
    previous_thresholds = detect_stream.get_detection_thresholds()

    try:
        detect_stream.set_detection_thresholds(0.4, 0.45)
        annotated, counts = detect_stream.postprocess_and_draw_result(
            output,
            r=1.0,
            pad=(0, 80),
            frame=frame,
            input_shape=(640, 640),
            output_has_objectness=False,
        )
    finally:
        detect_stream.set_detection_thresholds(*previous_thresholds)

    assert counts == {
        "plants_detected": 1,
        "disease_count": 0,
        "healthy_count": 1,
    }
    assert annotated[100, 100].tolist() == [0, 255, 0]
