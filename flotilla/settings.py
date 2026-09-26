"""Process settings read from environment variables (secrets and wiring)."""

from __future__ import annotations

import os
import socket
import ssl
from dataclasses import dataclass, field
from urllib.parse import urlparse

from flotilla.util import parse_labels, parse_list


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def ssl_context(ca_file: str | None) -> ssl.SSLContext | bool:
    """TLS verification setting for outgoing calls (custom CA if given)."""
    if ca_file:
        return ssl.create_default_context(cafile=ca_file)
    return True


@dataclass
class CoordinatorSettings:
    host: str = "0.0.0.0"
    port: int = 8800
    config_path: str = "config/flotilla.yaml"
    data_dir: str | None = None
    cluster_name: str = "flotilla"
    # Keys clients (Open WebUI, scripts) must send. Empty = no auth.
    api_keys: list[str] = field(default_factory=list)
    # Shared secret between the coordinator and agents.
    cluster_token: str | None = None
    tls_cert: str | None = None
    tls_key: str | None = None
    ca_file: str | None = None
    trust_env_proxy: bool = False
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "CoordinatorSettings":
        return cls(
            host=_env("FLOTILLA_HOST", "0.0.0.0"),
            port=_env_int("FLOTILLA_PORT", 8800),
            config_path=_env("FLOTILLA_CONFIG", "config/flotilla.yaml"),
            data_dir=_env("FLOTILLA_DATA_DIR"),
            cluster_name=_env("FLOTILLA_CLUSTER_NAME", "flotilla"),
            api_keys=parse_list(_env("FLOTILLA_API_KEYS")),
            cluster_token=_env("FLOTILLA_CLUSTER_TOKEN"),
            tls_cert=_env("FLOTILLA_TLS_CERT"),
            tls_key=_env("FLOTILLA_TLS_KEY"),
            ca_file=_env("FLOTILLA_CA_FILE"),
            trust_env_proxy=_env_bool("FLOTILLA_TRUST_ENV_PROXY"),
            log_level=_env("FLOTILLA_LOG_LEVEL", "info").lower(),
        )


@dataclass
class AgentSettings:
    coordinator_url: str | None = None
    cluster_token: str | None = None
    node_name: str = ""
    host: str = "0.0.0.0"
    port: int = 8801
    # URL the coordinator uses to reach this agent. When empty, the
    # coordinator uses the address the heartbeat came from plus `port`.
    advertise_url: str | None = None
    # Port the coordinator should use when it derives the URL from the
    # heartbeat's source address (the published host port, if remapped).
    advertise_port: int | None = None
    backend: str = "ollama"              # "ollama" or "openai"
    backend_url: str = "http://127.0.0.1:11434"
    backend_api_key: str | None = None
    max_concurrency: int = 2
    queue_timeout: float = 300.0
    request_timeout: float = 600.0
    heartbeat_interval: float = 10.0
    labels: dict[str, str] = field(default_factory=dict)
    pull_models: list[str] = field(default_factory=list)
    static_models: list[str] = field(default_factory=list)
    include_remote_models: bool = False
    tls_cert: str | None = None
    tls_key: str | None = None
    ca_file: str | None = None
    trust_env_proxy: bool = False
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "AgentSettings":
        advertise = _env("FLOTILLA_ADVERTISE_URL")
        name = _env("FLOTILLA_NODE_NAME")
        if not name:
            host = urlparse(advertise).hostname if advertise else None
            name = host or socket.gethostname()
        return cls(
            coordinator_url=(_env("FLOTILLA_COORDINATOR_URL") or "").rstrip("/") or None,
            cluster_token=_env("FLOTILLA_CLUSTER_TOKEN"),
            node_name=name,
            host=_env("FLOTILLA_HOST", "0.0.0.0"),
            port=_env_int("FLOTILLA_PORT", 8801),
            advertise_url=advertise.rstrip("/") if advertise else None,
            advertise_port=_env_int("FLOTILLA_ADVERTISE_PORT", 0) or None,
            backend=_env("FLOTILLA_BACKEND", "ollama").lower(),
            backend_url=_env("FLOTILLA_BACKEND_URL", "http://127.0.0.1:11434").rstrip("/"),
            backend_api_key=_env("FLOTILLA_BACKEND_API_KEY"),
            max_concurrency=_env_int("FLOTILLA_MAX_CONCURRENCY", 2),
            queue_timeout=_env_float("FLOTILLA_QUEUE_TIMEOUT", 300.0),
            request_timeout=_env_float("FLOTILLA_REQUEST_TIMEOUT", 600.0),
            heartbeat_interval=_env_float("FLOTILLA_HEARTBEAT_INTERVAL", 10.0),
            labels=parse_labels(_env("FLOTILLA_NODE_LABELS")),
            pull_models=parse_list(_env("FLOTILLA_PULL_MODELS")),
            static_models=parse_list(_env("FLOTILLA_MODELS")),
            include_remote_models=_env_bool("FLOTILLA_INCLUDE_REMOTE_MODELS"),
            tls_cert=_env("FLOTILLA_TLS_CERT"),
            tls_key=_env("FLOTILLA_TLS_KEY"),
            ca_file=_env("FLOTILLA_CA_FILE"),
            trust_env_proxy=_env_bool("FLOTILLA_TRUST_ENV_PROXY"),
            log_level=_env("FLOTILLA_LOG_LEVEL", "info").lower(),
        )
