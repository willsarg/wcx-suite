"""config.py — the validated VRAM safety margin."""
from __future__ import annotations

import pytest

from wcx_suite import config


def test_margin_defaults(monkeypatch):
    monkeypatch.delenv(config.MARGIN_ENV, raising=False)
    assert config.margin_gb() == config.DEFAULT_MARGIN_GB


def test_margin_explicit_value_overrides_env(monkeypatch):
    monkeypatch.setenv(config.MARGIN_ENV, "5")
    assert config.margin_gb(2.5) == 2.5   # explicit wins over the env


def test_margin_from_env(monkeypatch):
    monkeypatch.setenv(config.MARGIN_ENV, "3")
    assert config.margin_gb() == 3.0


def test_margin_rejects_non_numbers():
    with pytest.raises(ValueError):
        config.margin_gb("not-a-number")


def test_margin_rejects_negative():
    with pytest.raises(ValueError):
        config.margin_gb(-1)
