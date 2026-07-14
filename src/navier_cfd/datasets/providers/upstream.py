from __future__ import annotations

from dataclasses import asdict, dataclass, field
from ftplib import FTP
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

from ..huggingface import HuggingFaceDatasetManager


class OfficialDatasetAccessError(RuntimeError):
    """Raised when an official upstream dataset cannot be probed or staged safely."""


@dataclass(frozen=True)
class OfficialDatasetSource:
    dataset_id: str
    backend: str
    homepage: str
    license: str
    public: bool
    repository: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    checksums: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OfficialUpstreamProbe:
    dataset_id: str
    backend: str
    reachable: bool
    homepage: str
    entries: tuple[Mapping[str, Any], ...]
    authenticated: bool = False
    error_category: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialDownloadResult:
    dataset_id: str
    backend: str
    destination: str
    files: tuple[str, ...]
    source_urls: tuple[str, ...]
    sha256: Mapping[str, str]
    extracted_paths: tuple[str, ...]
    manifest_path: str
    auth_source: str = "anonymous"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OFFICIAL_DATASET_SOURCES: dict[str, OfficialDatasetSource] = {
    "airfrans": OfficialDatasetSource(
        dataset_id="airfrans",
        backend="direct_http",
        homepage="https://github.com/Extrality/airfrans_lib",
        license="ODbL-1.0",
        public=True,
        artifacts={
            "processed": "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip",
            "openfoam": "https://data.isir.upmc.fr/extrality/NeurIPS_2022/OF_dataset.zip",
        },
        notes=(
            "The official airfrans package can also download and load the collection.",
            "Both official archives are large; probe before downloading on small machines.",
        ),
    ),
    "drivaernetpp": OfficialDatasetSource(
        dataset_id="drivaernetpp",
        backend="dataverse",
        homepage="https://dataverse.harvard.edu/dataverse/DrivAerNet",
        license="CC-BY-NC-4.0",
        public=True,
        repository="Mohamedelrefaie/DrivAerNet",
        artifacts={
            "train_split": "https://raw.githubusercontent.com/Mohamedelrefaie/DrivAerNet/main/train_val_test_splits/train_design_ids.txt",
            "validation_split": "https://raw.githubusercontent.com/Mohamedelrefaie/DrivAerNet/main/train_val_test_splits/val_design_ids.txt",
            "test_split": "https://raw.githubusercontent.com/Mohamedelrefaie/DrivAerNet/main/train_val_test_splits/test_design_ids.txt",
        },
        notes=(
            "The complete CFD collection is distributed through Harvard Dataverse and Globus.",
            "Use dataverse_file_id or an official direct URL to stage a selected CFD file.",
        ),
    ),
    "drivaerml": OfficialDatasetSource(
        dataset_id="drivaerml",
        backend="huggingface",
        homepage="https://huggingface.co/datasets/neashton/drivaerml",
        license="CC-BY-SA-4.0",
        public=True,
        repository="neashton/drivaerml",
        artifacts={
            "force_mom_all": "force_mom_all.csv",
            "force_mom_constref_all": "force_mom_constref_all.csv",
            "geo_parameters_all": "geo_parameters_all.csv",
        },
        notes=(
            "Use explicit run paths to selectively download VTK, STL, CSV, or image files.",
            "The full collection is approximately 31 TB; snapshot download is not the default.",
        ),
    ),
    "scalarflow": OfficialDatasetSource(
        dataset_id="scalarflow",
        backend="ftp",
        homepage="https://ge.in.tum.de/publications/2019-scalarflow-eckert/",
        license="Research dataset; verify the official terms before redistribution",
        public=True,
        repository="dataserv.ub.tum.de",
        artifacts={},
        notes=(
            "Official FTP host, username, and password are all published by TUM mediaTUM.",
            "Individual reconstructions are several gigabytes; select files explicitly.",
        ),
    ),
    "shapenet_car": OfficialDatasetSource(
        dataset_id="shapenet_car",
        backend="direct_http",
        homepage="http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip",
        license="Research dataset; verify the original distribution terms",
        public=True,
        artifacts={
            "dataset": "http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip",
        },
        notes=(
            "This is the original ShapeNet-Car CFD benchmark archive used by downstream loaders.",
            "The archive contains parameter tarballs that must be extracted before training.",
        ),
    ),
    "eagle": OfficialDatasetSource(
        dataset_id="eagle",
        backend="direct_http",
        homepage="https://datasets.liris.cnrs.fr/eagle-version1",
        license="Research dataset; verify the official dataset page",
        public=True,
        repository="eagle-dataset/EagleMeshTransformer",
        artifacts={
            "clusters": "https://datasets.liris.cnrs.fr/eagle-version1/eagle_clusters.tar.gz",
            "spline": "https://datasets.liris.cnrs.fr/eagle-version1/spline.tar.gz",
            "step": "https://datasets.liris.cnrs.fr/eagle-version1/step.tar.gz",
            "triangular": "https://datasets.liris.cnrs.fr/eagle-version1/triangular.tar.gz",
        },
        checksums={
            "clusters": "f1bbc1dc22b0fbc57a5f8d0243d85f6471c43585fb0ecc7409de19996d3de12c",
            "spline": "f73cb9a443011646fb944e0a634a0d91c20b3d71a8b4d89d55486f9e99bdca78",
            "step": "ac04d3efb539a80d8538fb8214228652b482ab149fc7cc9ecf0b6d119e3b1be7",
            "triangular": "59a2ae96ca5ade7d3772e58b302c4132e1ee003ac239b7e38973ceb480a979e6",
        },
        notes=(
            "Official archives contain NPZ trajectories and triangle connectivity files.",
            "Published SHA-256 checksums are verified after complete downloads.",
        ),
    ),
    "apebench": OfficialDatasetSource(
        dataset_id="apebench",
        backend="procedural",
        homepage="https://github.com/tum-pbs/apebench",
        license="MIT",
        public=True,
        repository="tum-pbs/apebench",
        notes=(
            "APEBench generates official deterministic trajectories through its installed Python API.",
        ),
    ),
}


_ALLOWED_HTTP_HOSTS = {
    "data.isir.upmc.fr",
    "dataverse.harvard.edu",
    "datasets.liris.cnrs.fr",
    "huggingface.co",
    "raw.githubusercontent.com",
    "www.nobuyuki-umetani.com",
}


def _error_category(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code in {401, 403}:
            return "authentication"
        if exc.code == 404:
            return "not_found"
        return "http"
    message = str(exc).lower()
    if any(token in message for token in ("timed out", "timeout", "connection", "network")):
        return "network"
    return "upstream_error"


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise OfficialDatasetAccessError(f"Unsupported upstream URL scheme: {parsed.scheme!r}")
    if parsed.hostname not in _ALLOWED_HTTP_HOSTS:
        raise OfficialDatasetAccessError(
            f"Refusing untrusted upstream host {parsed.hostname!r}; pass an official allow-listed URL"
        )


def _http_probe(url: str, *, timeout: float) -> dict[str, Any]:
    _validate_http_url(url)
    request = Request(url, method="HEAD", headers={"User-Agent": "navier-cfd-upstream/0.8"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return {
                "url": url,
                "status": getattr(response, "status", 200),
                "size": response.headers.get("Content-Length"),
                "content_type": response.headers.get("Content-Type"),
            }
    except HTTPError as exc:
        if exc.code not in {405, 501}:
            raise
    request = Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "navier-cfd-upstream/0.8"},
    )
    with urlopen(request, timeout=timeout) as response:
        response.read(1)
        return {
            "url": url,
            "status": getattr(response, "status", 200),
            "size": response.headers.get("Content-Range") or response.headers.get("Content-Length"),
            "content_type": response.headers.get("Content-Type"),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_http(
    url: str,
    destination: Path,
    *,
    timeout: float,
    max_bytes: int | None,
    overwrite: bool,
) -> Path:
    _validate_http_url(url)
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "navier-cfd-upstream/0.8"})
    written = 0
    try:
        with urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            length = response.headers.get("Content-Length")
            if max_bytes is not None and length and int(length) > max_bytes:
                raise OfficialDatasetAccessError(
                    f"Official file is {int(length)} bytes, above max_bytes={max_bytes}"
                )
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if max_bytes is not None and written > max_bytes:
                    raise OfficialDatasetAccessError(
                        f"Download exceeded max_bytes={max_bytes}; partial file was removed"
                    )
                output.write(block)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise OfficialDatasetAccessError(f"Unsafe archive member path: {member}")
    return target


def _extract_archive(path: Path, destination: Path) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                _safe_target(destination, member.filename)
            archive.extractall(destination)
            extracted.extend(member.filename for member in archive.infolist() if not member.is_dir())
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            for member in members:
                _safe_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise OfficialDatasetAccessError(
                        f"Archive links are not extracted for safety: {member.name}"
                    )
            archive.extractall(destination, members=members)
            extracted.extend(member.name for member in members if member.isfile())
    else:
        raise OfficialDatasetAccessError(f"Not a recognized ZIP or TAR archive: {path}")
    return tuple(extracted)


def _artifact_filename(url: str, fallback: str) -> str:
    name = PurePosixPath(urlparse(url).path).name
    return name or fallback


class OfficialDatasetManager:
    """Probe and stage real files from official dataset distribution channels.

    This manager never bypasses authentication, license acceptance, registration, or storage
    limits. It downloads selected official artifacts, writes a provenance manifest, and leaves
    canonical parsing to the provider-specific/local NAVIER-CFD loaders.
    """

    def __init__(
        self,
        *,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.huggingface = HuggingFaceDatasetManager(token=token, endpoint=endpoint)

    def source(self, dataset_id: str) -> OfficialDatasetSource:
        try:
            return OFFICIAL_DATASET_SOURCES[dataset_id]
        except KeyError as exc:
            raise KeyError(
                f"No official upstream source for {dataset_id!r}; available: "
                f"{sorted(OFFICIAL_DATASET_SOURCES)}"
            ) from exc

    def probe(
        self,
        dataset_id: str,
        *,
        timeout: float = 20.0,
        max_entries: int = 50,
        revision: str | None = None,
    ) -> OfficialUpstreamProbe:
        source = self.source(dataset_id)
        try:
            if source.backend == "huggingface":
                assert source.repository is not None
                probe = self.huggingface.probe(
                    source.repository,
                    revision=revision,
                    max_files=max_entries,
                )
                return OfficialUpstreamProbe(
                    dataset_id=dataset_id,
                    backend=source.backend,
                    reachable=probe.reachable,
                    homepage=source.homepage,
                    entries=tuple({"path": item} for item in probe.files),
                    authenticated=probe.authenticated,
                    error_category=probe.error_category,
                    message=probe.message,
                )
            if source.backend == "ftp":
                with FTP("dataserv.ub.tum.de", timeout=timeout) as ftp:
                    ftp.login("m1521788", "m1521788")
                    entries = []
                    try:
                        for name, facts in ftp.mlsd():
                            if name in {".", ".."}:
                                continue
                            entries.append(
                                {
                                    "path": name,
                                    "size": facts.get("size"),
                                    "type": facts.get("type"),
                                }
                            )
                            if len(entries) >= max_entries:
                                break
                    except Exception:
                        entries = [{"path": name} for name in ftp.nlst()[:max_entries]]
                return OfficialUpstreamProbe(
                    dataset_id=dataset_id,
                    backend=source.backend,
                    reachable=True,
                    homepage=source.homepage,
                    entries=tuple(entries),
                )
            if source.backend == "procedural":
                available = importlib.util.find_spec("apebench") is not None
                return OfficialUpstreamProbe(
                    dataset_id=dataset_id,
                    backend=source.backend,
                    reachable=available,
                    homepage=source.homepage,
                    entries=({"module": "apebench", "installed": available},),
                    error_category=None if available else "dependency",
                    message=None if available else "Install navier-cfd[apebench]",
                )
            urls = tuple(source.artifacts.values()) or (source.homepage,)
            entries = tuple(_http_probe(url, timeout=timeout) for url in urls[:max_entries])
            return OfficialUpstreamProbe(
                dataset_id=dataset_id,
                backend=source.backend,
                reachable=True,
                homepage=source.homepage,
                entries=entries,
            )
        except Exception as exc:
            return OfficialUpstreamProbe(
                dataset_id=dataset_id,
                backend=source.backend,
                reachable=False,
                homepage=source.homepage,
                entries=(),
                error_category=_error_category(exc),
                message=str(exc),
            )

    def download(
        self,
        dataset_id: str,
        destination: str | Path,
        *,
        artifacts: Sequence[str] | None = None,
        revision: str | None = None,
        dataverse_file_id: int | None = None,
        official_url: str | None = None,
        extract: bool = False,
        max_bytes: int | None = None,
        timeout: float = 120.0,
        overwrite: bool = False,
        verify_checksum: bool = True,
    ) -> OfficialDownloadResult:
        source = self.source(dataset_id)
        root = Path(destination).expanduser() / dataset_id
        root.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        urls: list[str] = []
        expected_checksums: dict[str, str] = {}
        auth_source = "anonymous"

        if source.backend == "procedural":
            raise OfficialDatasetAccessError(
                "APEBench is procedural; use APEBenchDatasetManager.load to generate official trajectories"
            )
        if source.backend == "huggingface":
            assert source.repository is not None
            requested = tuple(artifacts or source.artifacts.keys())
            if not requested:
                raise OfficialDatasetAccessError("Specify one or more Hugging Face artifact paths")
            for item in requested:
                filename = source.artifacts.get(item, item)
                path = self.huggingface.download_file(
                    source.repository,
                    filename,
                    revision=revision,
                    local_dir=root,
                )
                files.append(path)
                urls.append(
                    f"https://huggingface.co/datasets/{source.repository}/resolve/"
                    f"{revision or 'main'}/{filename}"
                )
            auth_source = self.huggingface.auth.source
        elif source.backend == "ftp":
            requested = tuple(artifacts or ())
            if not requested:
                raise OfficialDatasetAccessError(
                    "ScalarFlow FTP staging requires explicit remote artifact names from probe()"
                )
            with FTP("dataserv.ub.tum.de", timeout=timeout) as ftp:
                ftp.login("m1521788", "m1521788")
                for remote in requested:
                    name = PurePosixPath(remote).name
                    path = root / name
                    if path.exists() and not overwrite:
                        files.append(path)
                        urls.append(f"ftp://dataserv.ub.tum.de/{remote}")
                        continue
                    temporary = path.with_suffix(path.suffix + ".part")
                    written = 0
                    try:
                        with temporary.open("wb") as output:
                            def write(block: bytes) -> None:
                                nonlocal written
                                written += len(block)
                                if max_bytes is not None and written > max_bytes:
                                    raise OfficialDatasetAccessError(
                                        f"FTP download exceeded max_bytes={max_bytes}"
                                    )
                                output.write(block)
                            ftp.retrbinary(f"RETR {remote}", write)
                        temporary.replace(path)
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        raise
                    files.append(path)
                    urls.append(f"ftp://dataserv.ub.tum.de/{remote}")
        else:
            requested_urls: list[tuple[str, str]] = []
            if source.backend == "dataverse" and dataverse_file_id is not None:
                requested_urls.append(
                    (
                        str(dataverse_file_id),
                        f"https://dataverse.harvard.edu/api/access/datafile/{dataverse_file_id}",
                    )
                )
            if official_url is not None:
                requested_urls.append(("official", official_url))
            for item in tuple(artifacts or ()):
                if item in source.artifacts:
                    requested_urls.append((item, source.artifacts[item]))
                elif item.startswith(("http://", "https://")):
                    requested_urls.append(("official", item))
                else:
                    raise OfficialDatasetAccessError(
                        f"Unknown artifact {item!r}; choose one of {sorted(source.artifacts)}"
                    )
            if not requested_urls:
                if source.backend == "dataverse":
                    raise OfficialDatasetAccessError(
                        "DrivAerNet++ CFD staging requires dataverse_file_id, official_url, or an "
                        "explicit official metadata artifact"
                    )
                requested_urls = list(source.artifacts.items())
            for key, url in requested_urls:
                filename = _artifact_filename(url, f"{dataset_id}-{key}")
                path = _download_http(
                    url,
                    root / filename,
                    timeout=timeout,
                    max_bytes=max_bytes,
                    overwrite=overwrite,
                )
                files.append(path)
                urls.append(url)
                if key in source.checksums:
                    expected_checksums[path.name] = source.checksums[key]

        checksums = {path.name: _sha256(path) for path in files}
        if verify_checksum:
            for filename, expected in expected_checksums.items():
                actual = checksums[filename]
                if actual.lower() != expected.lower():
                    raise OfficialDatasetAccessError(
                        f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
                    )

        extracted_paths: list[str] = []
        if extract:
            extraction_root = root / "extracted"
            for path in files:
                try:
                    extracted_paths.extend(_extract_archive(path, extraction_root / path.stem))
                except OfficialDatasetAccessError:
                    raise
                except (tarfile.TarError, zipfile.BadZipFile) as exc:
                    raise OfficialDatasetAccessError(str(exc)) from exc

        manifest = {
            "dataset_id": dataset_id,
            "backend": source.backend,
            "homepage": source.homepage,
            "license": source.license,
            "files": [str(path) for path in files],
            "source_urls": urls,
            "sha256": checksums,
            "expected_sha256": expected_checksums,
            "extracted_paths": extracted_paths,
            "auth_source": auth_source,
            "notes": list(source.notes),
        }
        manifest_path = root / "navier-upstream-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return OfficialDownloadResult(
            dataset_id=dataset_id,
            backend=source.backend,
            destination=str(root),
            files=tuple(str(path) for path in files),
            source_urls=tuple(urls),
            sha256=checksums,
            extracted_paths=tuple(extracted_paths),
            manifest_path=str(manifest_path),
            auth_source=auth_source,
        )


__all__ = [
    "OFFICIAL_DATASET_SOURCES",
    "OfficialDatasetAccessError",
    "OfficialDatasetManager",
    "OfficialDatasetSource",
    "OfficialDownloadResult",
    "OfficialUpstreamProbe",
]
