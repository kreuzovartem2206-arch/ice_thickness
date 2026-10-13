import zipfile
from pathlib import Path

import numpy as np
import pytest

import src.daily_multisat_l2_sit as m


def s3_name(start, created, cycle, orbit, timeliness, baseline="004", mission="S3A"):
    return (
        f"{mission}_SR_2_LAN_SI_{start}_{start}_{created}_"
        f"3000_{cycle}_{orbit}______LN3_O_{timeliness}_{baseline}.SEN3"
    )


def test_radar_freeboard_is_corrected_for_snow():
    fi = m.radar_to_ice_freeboard(np.array([0.10]), np.array([0.20]), np.array([300.0]))
    expected = 0.10 + 0.20 * ((1 + 0.51 * 0.3) ** 1.5 - 1)
    assert np.allclose(fi, expected)
    assert fi[0] > 0.10


def test_one_version_kept_per_pass_with_nt_preferred():
    nr = m.Product("1", s3_name("20260914T100000", "20260914T120000", "130", "216", "NR"))
    nt = m.Product("2", s3_name("20260914T100000", "20260920T000000", "130", "216", "NT"))
    st = m.Product("3", s3_name("20260914T100000", "20260915T000000", "130", "216", "ST"))
    other_nr = m.Product("4", s3_name("20260914T120000", "20260914T140000", "130", "217", "NR"))
    s3b = m.Product("5", s3_name("20260914T100000", "20260914T120000", "130", "216", "NR", mission="S3B"))
    unknown = m.Product("6", "standard_measurement.nc")

    chosen = m.choose_s3_timeliness([nr, nt, st, other_nr, s3b, unknown])

    assert sorted(p.id for p in chosen) == ["2", "4", "5", "6"]


def test_newer_baseline_wins_for_same_timeliness():
    old = m.Product("1", s3_name("20260914T100000", "20260920T000000", "130", "216", "NT", "004"))
    new = m.Product("2", s3_name("20260914T100000", "20260901T000000", "130", "216", "NT", "005"))
    assert [p.id for p in m.choose_s3_timeliness([old, new])] == ["2"]


def test_local_discovery_drops_duplicate_versions(tmp_path):
    for t in ("NR", "NT"):
        (tmp_path / s3_name("20260914T100000", "20260914T120000", "130", "216", t)).mkdir()
    found = m.discover_s3(tmp_path)
    assert len(found) == 1 and "_NT_" in found[0].name


def test_sea_ice_class_taken_from_flag_meanings():
    attrs = {"flag_values": np.array([0, 1, 2, 3]), "flag_meanings": "undefined lead floe open_ocean"}
    assert m.sea_ice_class_values(attrs) == [2]
    assert m.sea_ice_class_values({}) is None
    assert m.sea_ice_class_values({"flag_values": [0, 1], "flag_meanings": "ocean land"}) is None


def test_scalar_density_is_broadcast():
    frame = m.make_frame(
        time=np.array(["2026-09-14T10:00"], dtype="datetime64[ns]"),
        lat=np.array([80.0]),
        lon=np.array([100.0]),
        freeboard=np.array([0.2]),
        snow_depth=np.array([0.1]),
        snow_density=np.array(300.0),
        ice_density=np.array(917.0),
        source="x",
        sensor="CryoSat-2",
    )
    assert len(frame) == 1


class FakeResponse:
    def __init__(self, status, body=b""):
        self.status_code = status
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body


class FakeAuth:
    def __init__(self):
        self.forced = 0

    def token(self, force_refresh=False):
        self.forced += force_refresh
        return "t2" if force_refresh else "t1"


def zip_bytes(tmp_path, stem):
    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(f"{stem}.SEN3/standard_measurement.nc", b"x")
    return src.read_bytes()


def test_download_retries_with_fresh_token_after_401(tmp_path, monkeypatch):
    stem = "S3A_TEST"
    body = zip_bytes(tmp_path, stem)
    responses = [FakeResponse(401), FakeResponse(200, body)]
    monkeypatch.setattr(m.requests, "get", lambda *a, **k: responses.pop(0))
    auth = FakeAuth()

    out = m.download_s3_product(m.Product("1", f"{stem}.SEN3"), tmp_path / "raw", auth)

    assert out.name == f"{stem}.SEN3" and auth.forced == 1


def test_interrupted_download_leaves_no_archive(tmp_path, monkeypatch):
    class Broken(FakeResponse):
        def iter_content(self, chunk_size):
            yield b"partial"
            raise ConnectionError("cut")

    monkeypatch.setattr(m.requests, "get", lambda *a, **k: Broken(200))
    raw = tmp_path / "raw"
    with pytest.raises(ConnectionError):
        m.download_s3_product(m.Product("1", "S3A_TEST.SEN3"), raw, FakeAuth())
    assert not (raw / "S3A_TEST.zip").exists()


def test_corrupt_archive_is_fetched_again(tmp_path, monkeypatch):
    stem = "S3A_TEST"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / f"{stem}.zip").write_bytes(b"not a zip")
    body = zip_bytes(tmp_path, stem)
    monkeypatch.setattr(m.requests, "get", lambda *a, **k: FakeResponse(200, body))

    out = m.download_s3_product(m.Product("1", f"{stem}.SEN3"), raw, FakeAuth())

    assert out.exists()
