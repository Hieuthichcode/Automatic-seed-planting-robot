"""Static checks for the main dashboard button interaction styles."""

import re
from pathlib import Path


MAP_HTML = Path(__file__).resolve().parents[1] / "map.html"


def _button_tag(html, button_id):
    match = re.search(
        rf'<button\s+[^>]*id="{re.escape(button_id)}"[^>]*>',
        html,
        flags=re.IGNORECASE,
    )
    assert match is not None, f"Missing button #{button_id}"
    return match.group(0)


def test_main_buttons_share_hover_behavior_without_inline_backgrounds():
    html = MAP_HTML.read_text(encoding="utf-8")

    assert "button:not(:disabled):hover" in html
    assert "background-color: var(--button-hover-bg)" in html

    expected_classes = {
        "btnPause": "button-warning",
        "btnStop": "button-danger",
        "btnDetection": "button-success",
        "btnLidar": "button-purple",
        "btnArm": "button-cyan",
        "btnDepth": "button-blue",
        "btnConfig": "button-warning",
    }
    for button_id, class_name in expected_classes.items():
        tag = _button_tag(html, button_id)
        assert class_name in tag
        assert "background-color" not in tag
