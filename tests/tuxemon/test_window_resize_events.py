# SPDX-License-Identifier: GPL-3.0
from dataclasses import replace

import pygame as pg
import pytest

from tuxemon import prepare
from tuxemon.platform.input_manager import InputManager
from tuxemon.platform.platform_pygame.events import PygameEventQueueHandler


@pytest.fixture(scope="module", autouse=True)
def pygame_setup_teardown():
    pg.init()
    yield
    pg.quit()


class FakeWindow:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size


class FakeRenderer:
    def __init__(self) -> None:
        self.logical_size = (768, 432)
        self.scale = (1.0, 1.0)
        self.viewport = pg.Rect(0, 0, 0, 0)

    def set_viewport(self, rect: pg.Rect) -> None:
        self.viewport = pg.Rect(rect)

    def get_viewport(self) -> pg.Rect:
        return pg.Rect(self.viewport)


@pytest.fixture
def gpu_display(monkeypatch: pytest.MonkeyPatch):
    window = FakeWindow((1280, 720))
    renderer = FakeRenderer()
    context = replace(
        prepare.DISPLAY_CONTEXT,
        resolution=(768, 432),
        gpu_upscaling=True,
        window_size=(1280, 720),
    )
    monkeypatch.setattr(prepare, "DISPLAY_CONTEXT", context)
    monkeypatch.setattr(prepare, "_gpu_renderer", renderer)
    monkeypatch.setattr(prepare, "_gpu_window", window)
    prepare.handle_window_resized((1280, 720))
    return window, renderer, context


def test_sync_refits_after_window_resize(gpu_display):
    window, renderer, context = gpu_display
    window.size = (1600, 600)

    prepare.sync_gpu_display()

    fit = prepare.FrameFit.compute((1600, 600), (768, 432))
    assert context.window_size == (1600, 600)
    assert renderer.get_viewport() == pg.Rect(fit.offset_x, 0, 768, 432)
    assert context.window_to_screen((10, 300)) is None


def test_sync_restores_viewport_reset_by_sdl(gpu_display):
    window, renderer, context = gpu_display
    renderer.set_viewport(pg.Rect(0, 0, 960, 540))

    prepare.sync_gpu_display()

    assert renderer.get_viewport() == pg.Rect(0, 0, 768, 432)


def test_sync_does_nothing_when_unchanged(gpu_display, monkeypatch):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(prepare, "handle_window_resized", calls.append)

    prepare.sync_gpu_display()

    assert calls == []


def test_sync_ignored_in_cpu_mode(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(prepare, "handle_window_resized", calls.append)
    monkeypatch.setattr(prepare, "_gpu_renderer", None)
    monkeypatch.setattr(prepare, "_gpu_window", None)

    prepare.sync_gpu_display()

    assert calls == []


def test_live_input_handles_resize_events(monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[int, int]] = []
    monkeypatch.setattr(prepare, "handle_window_resized", seen.append)
    pg.event.clear()
    pg.event.post(pg.event.Event(pg.WINDOWSIZECHANGED, x=1280, y=720))

    list(PygameEventQueueHandler().process_events())

    assert seen == [(1280, 720)]


class FakeRecorder:
    _is_playing_back = True
    _is_recording = False

    def next_playback_event(self):
        return None


class FakeEventQueue:
    def __init__(self) -> None:
        self.window_syncs = 0
        self.filters = []

    def sync_window_size(self) -> None:
        self.window_syncs += 1

    def set_event_filter(self, filter_func) -> None:
        self.filters.append(filter_func)


def test_playback_still_syncs_window_size():
    manager = InputManager.__new__(InputManager)
    manager.recorder = FakeRecorder()
    manager.event_queue = FakeEventQueue()

    assert list(manager.process_events()) == []
    assert manager.event_queue.window_syncs == 1
