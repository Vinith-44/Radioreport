"""Tests for the ReXGradient streaming fetcher on a local zstd-compressed tar split into parts."""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rexgradient_fetch import (ConcatParts, extract_wanted, member_rel,  # noqa: E402
                                        wanted_from_metadata, zstd_reader)


def _compress(data: bytes) -> bytes:
    try:
        from compression import zstd

        return zstd.compress(data)
    except ImportError:
        import zstandard

        return zstandard.ZstdCompressor().compress(data)


def _make_archive(n_files: int = 30) -> tuple[bytes, dict[str, bytes]]:
    import random
    rng = random.Random(0)  # incompressible bytes, so the compressed parts are realistically large
    files = {f"P{i}/A{i}/studies/S{i}/series/X{i}/instances/U{i}.png": rng.randbytes(700 + 37 * i)
             for i in range(n_files)}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for rel, data in files.items():
            ti = tarfile.TarInfo("deid_png/" + rel)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return _compress(buf.getvalue()), files


def _split(blob: bytes, n: int) -> dict[str, bytes]:
    k = len(blob) // n + 1
    return {f"part{i:02d}": blob[i * k:(i + 1) * k] for i in range(n)}


def test_member_rel_and_wanted():
    assert member_rel("./deid_png/P1/x.png") == "P1/x.png"
    meta = {"k": {"ImagePath": ["../deid_png/P1/A/studies/S/series/X/instances/U.png"]}}
    assert wanted_from_metadata(meta) == {"P1/A/studies/S/series/X/instances/U.png"}


def test_extract_only_wanted_across_parts(tmp_path):
    blob, files = _make_archive()
    parts = _split(blob, 4)                       # byte splits, like deid_png.partXX
    wanted = {r for i, r in enumerate(files) if i % 3 == 0}
    raw = ConcatParts(lambda p: io.BytesIO(parts[p]), list(parts))
    st = extract_wanted(zstd_reader(raw), wanted, tmp_path)
    assert st["found"] == len(wanted) and st["written"] == len(wanted) and not st["missing"]
    for r in wanted:
        assert (tmp_path / r).read_bytes() == files[r]
    assert sum(1 for _ in tmp_path.rglob("*.png")) == len(wanted)
    # Re-run: nothing re-written.
    raw = ConcatParts(lambda p: io.BytesIO(parts[p]), list(parts))
    st2 = extract_wanted(zstd_reader(raw), wanted, tmp_path)
    assert st2["written"] == 0 and st2["already_present"] == len(wanted)


def test_missing_reported(tmp_path):
    blob, files = _make_archive(5)
    raw = ConcatParts(lambda p: io.BytesIO(blob), ["p0"])
    st = extract_wanted(zstd_reader(raw), set(files) | {"NOT/THERE.png"}, tmp_path)
    assert st["missing"] == ["NOT/THERE.png"]


class _Flaky(io.BytesIO):
    """Raises a connection error once in the middle of the stream."""
    failed = False

    def read(self, n=-1):
        if not _Flaky.failed and self.tell() > 200:
            _Flaky.failed = True
            raise ConnectionError("simulated drop")
        return super().read(min(n, 97) if n and n > 0 else n)


def test_concat_parts_retries_at_same_offset(tmp_path, monkeypatch):
    import src.data.rexgradient_fetch as rf

    monkeypatch.setattr(rf.time, "sleep", lambda s: None)
    blob, files = _make_archive(12)
    parts = _split(blob, 3)
    _Flaky.failed = False
    raw = ConcatParts(lambda p: _Flaky(parts[p]), list(parts))
    st = extract_wanted(zstd_reader(raw), set(files), tmp_path)
    assert _Flaky.failed and st["found"] == len(files)
    for r, data in files.items():
        assert (tmp_path / r).read_bytes() == data
