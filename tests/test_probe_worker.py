"""probe_worker.py — the subprocess dispatch + JSON contract (GPU modes verified on hardware)."""
from __future__ import annotations

import json

from wcx_suite import probe_worker


def test_worker_unknown_mode(capsys):
    rc = probe_worker.main(["bogus"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False and "unknown mode" in out["error"]


def test_worker_no_mode(capsys):
    rc = probe_worker.main([])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False


def test_worker_runs_mode_and_emits_json(monkeypatch, capsys):
    monkeypatch.setitem(probe_worker._MODES, "fake", lambda: {"ok": True, "x": 1})
    rc = probe_worker.main(["fake"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out == {"ok": True, "x": 1}


def test_worker_catches_mode_exception(monkeypatch, capsys):
    def boom():
        raise RuntimeError("cuda exploded")
    monkeypatch.setitem(probe_worker._MODES, "boom", boom)
    rc = probe_worker.main(["boom"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False and "cuda exploded" in out["error"]
