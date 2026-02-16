from __future__ import annotations

from c4.stages.axis_ocr import _expand_axis_roi_x


def test_expand_axis_roi_x_left_padding_clamped():
    x0, x1 = 300, 380
    new_x0, new_x1, pad_left, pad_right = _expand_axis_roi_x(x0, x1, frame_w=1000)

    assert 40 <= (x0 - new_x0) <= 140
    assert pad_left == (x0 - new_x0)
    assert 10 <= pad_right <= 40
    assert new_x1 == x1 + pad_right
