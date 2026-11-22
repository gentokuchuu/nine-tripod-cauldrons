# SPDX-License-Identifier: GPL-3.0
import pytest

from tuxemon.prepare import FrameFit, gpu_render_scale


def test_same_aspect_fills_window_without_borders():
    fit = FrameFit.compute((1280, 720), (768, 432))
    assert fit.stretch == pytest.approx(1280 / 768)
    assert (fit.offset_x, fit.offset_y) == (0, 0)


def test_whole_number_ratio():
    fit = FrameFit.compute((1536, 864), (768, 432))
    assert fit.stretch == 2
    assert fit.window_to_screen((1535, 863)) == (767, 431)


def test_wider_window_gets_side_borders():
    fit = FrameFit.compute((2560, 1080), (768, 432))
    assert fit.stretch == pytest.approx(1080 / 432)
    assert fit.offset_x == round((2560 / fit.stretch - 768) / 2)
    assert fit.offset_y == 0


def test_window_to_screen_maps_centre_and_corners():
    fit = FrameFit.compute((1280, 720), (768, 432))
    assert fit.window_to_screen((640, 360)) == (384, 216)
    assert fit.window_to_screen((0, 0)) == (0, 0)
    assert fit.window_to_screen((1279, 719)) == (767, 431)


def test_window_to_screen_ignores_clicks_on_borders():
    fit = FrameFit.compute((2560, 1080), (768, 432))
    assert fit.window_to_screen((0, 540)) is None
    assert fit.window_to_screen((2559, 540)) is None


def test_window_to_screen_keeps_points_just_inside_the_frame():
    fit = FrameFit.compute((2560, 1080), (768, 432))
    left = fit.offset_x * fit.stretch
    right = (fit.offset_x + 768) * fit.stretch
    assert fit.window_to_screen((left + 1, 540)) == (0, 216)
    assert fit.window_to_screen((right - 1, 540)) == (767, 216)


def test_handle_window_resized_refits_gpu_display(monkeypatch):
    from dataclasses import replace

    from tuxemon import prepare

    class FakeRenderer:
        logical_size = (768, 432)
        scale = (1.0, 1.0)
        viewport = None

        def set_viewport(self, rect):
            self.viewport = rect

    renderer = FakeRenderer()
    context = replace(
        prepare.DISPLAY_CONTEXT,
        resolution=(768, 432),
        gpu_upscaling=True,
        window_size=(1280, 720),
    )
    monkeypatch.setattr(prepare, "DISPLAY_CONTEXT", context)
    monkeypatch.setattr(prepare, "_gpu_renderer", renderer)

    prepare.handle_window_resized((2560, 1080))

    fit = FrameFit.compute((2560, 1080), (768, 432))
    assert context.window_size == (2560, 1080)
    assert renderer.logical_size == (0, 0)
    assert renderer.scale == (fit.stretch, fit.stretch)
    assert tuple(renderer.viewport) == (fit.offset_x, 0, 768, 432)
    assert context.window_to_screen((1280, 540)) == (384, 216)
    assert context.window_to_screen((0, 540)) is None


def test_handle_window_resized_ignored_in_cpu_mode(monkeypatch):
    from dataclasses import replace

    from tuxemon import prepare

    context = replace(prepare.DISPLAY_CONTEXT, window_size=(1280, 720))
    monkeypatch.setattr(prepare, "DISPLAY_CONTEXT", context)
    monkeypatch.setattr(prepare, "_gpu_renderer", None)

    prepare.handle_window_resized((1600, 900))

    assert context.window_size == (1280, 720)


@pytest.mark.parametrize(
    "requested, expected",
    [
        pytest.param(-2, 1, id="negative"),
        pytest.param(0, 1, id="zero"),
        pytest.param(1, 1, id="one"),
        pytest.param(3, 3, id="three"),
        pytest.param(8, 8, id="max"),
        pytest.param(20, 8, id="above_max"),
    ],
)
def test_gpu_render_scale_is_clamped(requested: int, expected: int):
    assert gpu_render_scale(requested) == expected
