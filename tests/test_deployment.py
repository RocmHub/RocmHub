"""Deployment configuration contracts that do not require a cloud provider."""

from __future__ import annotations

from pathlib import Path

from rocmhub.server.config import ServerConfig


def test_platform_port_and_durable_data_directory_are_environment_driven(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setenv("PORT", "4567")  # type: ignore[attr-defined]
    monkeypatch.setenv("ROCMHUB_DATA_DIR", str(tmp_path / "railway-volume"))  # type: ignore[attr-defined]

    config = ServerConfig()

    assert config.port == 4567
    assert config.data_dir == (tmp_path / "railway-volume").resolve()
    assert config.db_path == (tmp_path / "railway-volume" / "jobs.db").resolve()
    assert config.artifact_storage_dir == (tmp_path / "railway-volume" / "agent-artifacts").resolve()
