"""cli.py — the `wcx-suite` entry point (barebones: the `system` view)."""
from __future__ import annotations

from wcx_suite import cli


def test_main_system_prints_vram(monkeypatch, capsys):
    monkeypatch.setattr(cli.system, "read_limits",
                        lambda: cli.system.GPULimits("NVIDIA GeForce RTX 2070",
                                                     total_gb=8.0, wall_gb=8.0,
                                                     free_gb=7.0, used_gb=1.0))
    rc = cli.main(["system"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RTX 2070" in out
    assert "8.0 GB" in out          # the VRAM wall
    assert "7.0 GB" in out          # free now


def test_main_system_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(cli.system, "read_limits", lambda: None)
    rc = cli.main(["system"])
    assert rc == 1
    assert "no NVIDIA GPU" in capsys.readouterr().out


def test_main_no_command_shows_help(capsys):
    rc = cli.main([])
    assert rc == 1
    assert "wcx-suite" in capsys.readouterr().out


def test_main_calibrate_prints_overhead(monkeypatch, capsys):
    monkeypatch.setattr(cli.probe, "calibrate",
                        lambda: cli.probe.Calibration(device="RTX 2070", overhead_gb=0.42))
    rc = cli.main(["calibrate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RTX 2070" in out and "0.42 GB" in out


def test_main_calibrate_fails_without_cuda(monkeypatch, capsys):
    monkeypatch.setattr(cli.probe, "calibrate", lambda: None)
    rc = cli.main(["calibrate"])
    assert rc == 1
    assert "calibration failed" in capsys.readouterr().out


# UTF-8 stdout so Windows' legacy cp1252 console doesn't crash on glyphs (−, —).
class _FakeStream:
    def __init__(self, encoding="cp1252", raises=None):
        self.encoding = encoding
        self._raises = raises
        self.reconfigured = None

    def reconfigure(self, **kwargs):
        if self._raises is not None:
            raise self._raises
        self.reconfigured = kwargs
        self.encoding = kwargs.get("encoding", self.encoding)


def test_ensure_utf8_reconfigures_legacy_codepage():
    s = _FakeStream("cp1252")
    cli._ensure_utf8(s)
    assert s.reconfigured == {"encoding": "utf-8", "errors": "replace"}


def test_ensure_utf8_skips_utf8_stream():
    s = _FakeStream("utf-8")
    cli._ensure_utf8(s)
    assert s.reconfigured is None


def test_ensure_utf8_noop_without_reconfigure():
    import io
    cli._ensure_utf8(io.StringIO())   # no reconfigure attr → must not raise


def test_ensure_utf8_swallows_failure():
    s = _FakeStream("cp1252", raises=ValueError("nope"))
    cli._ensure_utf8(s)
    assert s.reconfigured is None
