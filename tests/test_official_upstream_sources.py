from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from navier_cfd.datasets.providers import upstream
from navier_cfd.datasets.providers.upstream import (
    OFFICIAL_DATASET_SOURCES,
    OfficialDatasetAccessError,
    OfficialDatasetManager,
    OfficialDatasetSource,
)


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _Response(BytesIO):
    def __init__(self, payload: bytes, *, status: int = 200, content_type: str = "application/octet-stream"):
        super().__init__(payload)
        self.status = status
        self.headers = _Headers(
            {
                "Content-Length": str(len(payload)),
                "Content-Type": content_type,
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def test_official_registry_covers_non_hub_catalog_sources() -> None:
    assert {
        "airfrans",
        "apebench",
        "drivaerml",
        "drivaernetpp",
        "eagle",
        "scalarflow",
        "shapenet_car",
    } <= set(OFFICIAL_DATASET_SOURCES)
    assert OFFICIAL_DATASET_SOURCES["drivaerml"].repository == "neashton/drivaerml"
    assert OFFICIAL_DATASET_SOURCES["eagle"].checksums["step"].startswith("ac04d3")


def test_http_probe_and_download_write_provenance(monkeypatch, tmp_path: Path) -> None:
    payload = b"official-upstream-bytes"

    def fake_urlopen(request, timeout=None):
        del timeout
        method = request.get_method()
        return _Response(b"" if method == "HEAD" else payload)

    monkeypatch.setattr(upstream, "urlopen", fake_urlopen)
    source = OfficialDatasetSource(
        dataset_id="fixture",
        backend="direct_http",
        homepage="https://raw.githubusercontent.com/example/project/main/data.bin",
        license="test",
        public=True,
        artifacts={"sample": "https://raw.githubusercontent.com/example/project/main/data.bin"},
    )
    monkeypatch.setitem(OFFICIAL_DATASET_SOURCES, "fixture", source)

    manager = OfficialDatasetManager()
    probe = manager.probe("fixture")
    assert probe.reachable is True
    assert probe.entries[0]["status"] == 200

    result = manager.download("fixture", tmp_path, artifacts=["sample"])
    saved = Path(result.files[0])
    assert saved.read_bytes() == payload
    assert Path(result.manifest_path).exists()
    assert result.sha256[saved.name]


def test_download_rejects_untrusted_hosts(tmp_path: Path) -> None:
    manager = OfficialDatasetManager()
    with pytest.raises(OfficialDatasetAccessError, match="untrusted upstream host"):
        manager.download(
            "drivaernetpp",
            tmp_path,
            official_url="https://example.invalid/private-file.zip",
        )


def test_safe_zip_extraction_blocks_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "blocked")
    with pytest.raises(OfficialDatasetAccessError, match="Unsafe archive member"):
        upstream._extract_archive(archive, tmp_path / "out")


def test_huggingface_official_file_is_selected_without_snapshot(monkeypatch, tmp_path: Path) -> None:
    manager = OfficialDatasetManager()
    official = tmp_path / "force_mom_all.csv"
    official.write_text("run,drag\n1,0.3\n", encoding="utf-8")
    calls = []

    def fake_download_file(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return official

    monkeypatch.setattr(manager.huggingface, "download_file", fake_download_file)
    result = manager.download("drivaerml", tmp_path, artifacts=["force_mom_all"])
    assert calls[0][0] == "neashton/drivaerml"
    assert calls[0][1] == "force_mom_all.csv"
    assert result.files == (str(official),)


def test_apebench_probe_checks_the_real_optional_dependency(monkeypatch) -> None:
    manager = OfficialDatasetManager()
    monkeypatch.setattr(upstream.importlib.util, "find_spec", lambda name: SimpleNamespace(name=name))
    probe = manager.probe("apebench")
    assert probe.reachable is True
    assert probe.entries[0]["module"] == "apebench"


class _FakeFTP:
    def __init__(self, host, timeout=None):
        self.host = host
        self.timeout = timeout
        self.payload = b"scalarflow-official"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login(self, username, password):
        assert username == password == "m1521788"

    def mlsd(self):
        return iter([("reconstruction_000.tar", {"size": str(len(self.payload)), "type": "file"})])

    def retrbinary(self, command, callback):
        assert command == "RETR reconstruction_000.tar"
        callback(self.payload)


def test_scalarflow_uses_published_official_ftp_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(upstream, "FTP", _FakeFTP)
    manager = OfficialDatasetManager()
    probe = manager.probe("scalarflow")
    assert probe.reachable is True
    assert probe.entries[0]["path"] == "reconstruction_000.tar"
    result = manager.download(
        "scalarflow",
        tmp_path,
        artifacts=["reconstruction_000.tar"],
        max_bytes=1024,
    )
    assert Path(result.files[0]).read_bytes() == b"scalarflow-official"
