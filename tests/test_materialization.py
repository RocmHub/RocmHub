"""Managed Agent-local materialization and consent contract tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from rocmhub.core.errors import InsufficientDiskSpaceError, ModelMaterializationError
from rocmhub.core.types import ModelSpec
from rocmhub.forge.base import MaterializationMode
from rocmhub.forge.materializer import ModelMaterializer
from rocmhub.server.orchestrator.models import JobCreateRequest

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(
        model_id=MODEL,
        requested_revision=REVISION,
        commit_sha=REVISION,
        architecture="Qwen2ForCausalLM",
        parameter_count=500_000_000,
        weights_format="safetensors",
        remote_code_required=False,
    )


def downloaded_snapshot(cache_dir: Path, revision: str = REVISION) -> str:
    snapshot = cache_dir / f"models--{MODEL.replace('/', '--')}" / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (snapshot / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fixture-weight-bytes")
    return str(snapshot)


def full_job(**changes: object) -> dict[str, object]:
    job: dict[str, object] = {
        "job_type": "PREPARE_MODEL_FOR_AMD",
        "model_id": MODEL,
        "revision": REVISION,
        "materialization_mode": "FULL_WEIGHTS",
        "weights_consent": True,
    }
    job.update(changes)
    return job


def test_full_materialization_requires_explicit_consent() -> None:
    with pytest.raises(ValidationError, match="weights_consent"):
        JobCreateRequest.model_validate(full_job(weights_consent=False))


def test_prepare_contract_requires_immutable_revision() -> None:
    with pytest.raises(ValidationError, match="immutable 40-character"):
        JobCreateRequest.model_validate(full_job(revision="main"))


def test_metadata_only_request_is_default_and_rejects_legacy_download_flag() -> None:
    base = {"job_type": "PREPARE_MODEL_FOR_AMD", "model_id": MODEL, "revision": REVISION}
    req = JobCreateRequest.model_validate(base)
    assert req.materialization_mode.value == "METADATA_ONLY"
    assert req.weights_consent is False
    with pytest.raises(ValidationError, match="metadata-only"):
        JobCreateRequest.model_validate({**base, "allow_full_weights": True})


def test_first_managed_materialization_is_complete_and_provenanced(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]))) as download:
        result = materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    assert download.call_args.kwargs["revision"] == REVISION
    assert result.has_weights and not result.cached
    assert result.cache_status == "MISS_DOWNLOADED"
    assert result.integrity == "FULL_SHA256_MANIFEST_VERIFIED"
    assert result.materialized_bytes > 0
    assert (tmp_path / "models" / next((tmp_path / "models").iterdir()).name / ".rocmhub-cache.json").exists()


def test_identical_materialization_reuses_verified_cache(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]))):
        first = materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    with patch("rocmhub.forge.materializer.snapshot_download") as download:
        second = ModelMaterializer(cache_dir=tmp_path, managed_cache=True).materialize(
            spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS
        )
    assert second.cached and second.cache_status == "HIT_VERIFIED"
    assert second.local_path == first.local_path
    download.assert_not_called()


def test_corrupted_or_incomplete_cache_is_never_a_hit(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]))):
        materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    entry = next((tmp_path / "models").iterdir())
    (entry / ".rocmhub-cache.json").write_text('{"state":"DOWNLOADING"}', encoding="utf-8")
    with patch("rocmhub.forge.materializer.snapshot_download") as download:
        with pytest.raises(ModelMaterializationError, match="integrity verification"):
            materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    download.assert_not_called()


def test_interrupted_download_keeps_recoverable_stage_and_retry_publishes(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    def interrupt(**kwargs: object) -> str:
        downloaded_snapshot(Path(kwargs["cache_dir"]))
        raise RuntimeError("interrupted")
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=interrupt):
        with pytest.raises(ModelMaterializationError):
            materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    assert list((tmp_path / "staging").iterdir())
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]))):
        result = materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    assert result.cache_status == "MISS_DOWNLOADED" and result.has_weights


def test_cancellation_before_publish_does_not_create_complete_entry(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(
        cache_dir=tmp_path, managed_cache=True, cancellation_check=lambda: True
    )
    with patch("rocmhub.forge.materializer.snapshot_download") as download:
        with pytest.raises(ModelMaterializationError, match="cancelled"):
            materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    download.assert_not_called()
    assert not (tmp_path / "models" / ModelMaterializer._cache_key(MODEL, REVISION)).exists()


def test_insufficient_disk_refuses_materialization(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("shutil.disk_usage", return_value=type("Usage", (), {"free": 1024})()):
        with patch("rocmhub.forge.materializer.snapshot_download") as download:
            with pytest.raises(InsufficientDiskSpaceError):
                materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    download.assert_not_called()


def test_concurrent_identical_jobs_share_atomic_materialization(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    count = 0
    count_lock = threading.Lock()
    def download(**kwargs: object) -> str:
        nonlocal count
        with count_lock:
            count += 1
        time.sleep(0.05)
        return downloaded_snapshot(Path(kwargs["cache_dir"]))
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=download):
        results: list[object] = []
        threads = [threading.Thread(target=lambda: results.append(materializer.materialize(
            spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS
        ))) for _ in range(2)]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
    assert count == 1
    assert {result.cache_status for result in results} == {"MISS_DOWNLOADED", "HIT_VERIFIED"}


def test_revision_change_gets_distinct_cache_identity(tmp_path: Path, spec: ModelSpec) -> None:
    other = spec.model_copy(update={"commit_sha": "a" * 40})
    assert ModelMaterializer._cache_key(spec.model_id, spec.commit_sha) != ModelMaterializer._cache_key(other.model_id, other.commit_sha)


def test_cache_manifest_has_no_absolute_path(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]))):
        materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    manifest_path = next((tmp_path / "models").iterdir()) / ".rocmhub-cache.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(raw)
    assert manifest["state"] == "COMPLETE"
    assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])
    assert str(tmp_path) not in raw


def test_config_or_tokenizer_corruption_is_detected_by_selected_hash(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]))):
        result = materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    (Path(result.local_path) / "config.json").write_text('{"model_type":"evil"}', encoding="utf-8")
    with pytest.raises(ModelMaterializationError, match="integrity verification"):
        materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)


def test_model_size_limit_is_checked_before_network_request(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True, max_cache_bytes=1024)
    with patch("rocmhub.forge.materializer.snapshot_download") as download:
        with pytest.raises(InsufficientDiskSpaceError, match="cache limit"):
            materializer.materialize(spec, required_disk_bytes=4096, mode=MaterializationMode.FULL_WEIGHTS)
    download.assert_not_called()


def test_full_materialization_without_weights_cannot_publish(tmp_path: Path, spec: ModelSpec) -> None:
    def no_weights(**kwargs: object) -> str:
        snapshot = Path(kwargs["cache_dir"]) / f"models--{MODEL.replace('/', '--')}" / "snapshots" / REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{}', encoding="utf-8")
        (snapshot / "tokenizer.json").write_text('{}', encoding="utf-8")
        return str(snapshot)

    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=no_weights):
        with pytest.raises(ModelMaterializationError, match="No valid weight files"):
            materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    assert not (tmp_path / "models" / ModelMaterializer._cache_key(MODEL, REVISION)).exists()


def test_forged_cache_identity_does_not_reuse_other_revision(tmp_path: Path, spec: ModelSpec) -> None:
    materializer = ModelMaterializer(cache_dir=tmp_path, managed_cache=True)
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]))):
        materializer.materialize(spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    newer_spec = spec.model_copy(update={"commit_sha": "a" * 40})
    with patch("rocmhub.forge.materializer.snapshot_download", side_effect=lambda **kw: downloaded_snapshot(Path(kw["cache_dir"]), "a" * 40)) as download:
        newer = materializer.materialize(newer_spec, required_disk_bytes=1024, mode=MaterializationMode.FULL_WEIGHTS)
    assert download.call_args.kwargs["revision"] == "a" * 40
    assert newer.cached is False
    assert len(list((tmp_path / "models").iterdir())) == 2
