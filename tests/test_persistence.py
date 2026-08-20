from pathlib import Path

import pytest

from tamis.persistence import atomic_write_bytes


def test_atomic_write_bytes_writes_the_file(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_bytes(path, b"hello")
    assert path.read_bytes() == b"hello"


def test_atomic_write_bytes_overwrites_existing_content(tmp_path):
    path = tmp_path / "data.json"
    path.write_bytes(b"old")
    atomic_write_bytes(path, b"new")
    assert path.read_bytes() == b"new"


def test_atomic_write_bytes_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_bytes(path, b"hello")
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_bytes_does_not_corrupt_the_original_if_the_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    path.write_bytes(b"original")

    def failing_write_bytes(self, data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", failing_write_bytes)

    with pytest.raises(OSError):
        atomic_write_bytes(path, b"new")

    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".tamis_write_*"))
