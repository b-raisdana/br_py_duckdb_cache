import base64
import hashlib
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

_CONFIG_INSTANCE: "Config | None" = None
_CONFIG_OVERRIDES: dict[str, Any] = {}
_MODULE_STATE: dict[str, Any] = {}


class Config(BaseSettings):  # type: ignore[explicit-any]
    """Runtime settings. Any field can be overridden via a `DLF_<FIELD_NAME>` env var
    (or a `.env` file), validated against its declared type/bounds on load and on
    every later `app_config.<field> = ...` assignment."""

    model_config = SettingsConfigDict(
        env_prefix="DLF_",
        env_file=".env",
        extra="ignore",
        validate_assignment=True,
    )

    root_path: Path = _ROOT_PATH

    timeframes: tuple[str, ...] = (
        "1min",
        "5min",
        "15min",
        "1h",
        "4h",
        "1D",
        "1W",
    )

    under_process_market: str = ""
    under_process_symbol: str = ""
    under_process_exchange: str = ""
    environment: str = "development"

    @property
    def path_of_logs(self) -> Path:
        return self.root_path / "logs"


def configure(**overrides: Any) -> None:
    """Override default config values before first access.

    Must be called before accessing `app_config` for the first time.
    Example:
        from config import configure
        configure(timeframes=("1min", "5min", "1h"))
    """
    global _CONFIG_OVERRIDES
    if _CONFIG_INSTANCE is not None:
        raise RuntimeError("Cannot configure after app_config has been accessed")
    _CONFIG_OVERRIDES.update(overrides)


def _create_config() -> Config:
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = Config(**_CONFIG_OVERRIDES)
        _CONFIG_OVERRIDES.clear()
    return _CONFIG_INSTANCE


def _initialize_module() -> None:
    """Initialize module-level attributes on first access."""
    if not _MODULE_STATE:
        config = _create_config()
        config_as_json = config.model_dump_json()
        config_digest = str.translate(
            base64.b64encode(hashlib.md5(config_as_json.encode("utf-8")).digest()).decode("ascii"),
            {
                ord("+"): "",
                ord("/"): "",
                ord("="): "",
            },
        )
        config_log_dir = config.path_of_logs / "config"
        config_log_dir.mkdir(parents=True, exist_ok=True)
        dump_filename = config_log_dir / f"Config.{config_digest}.json"
        if not dump_filename.exists():
            dump_filename.write_text(config_as_json, encoding="utf-8")
        _MODULE_STATE.update(
            {
                "config_as_json": config_as_json,
                "config_digest": config_digest,
                "config_log_dir": config_log_dir,
                "dump_filename": dump_filename,
                "id": config_digest,
            }
        )


class _ConfigProxy:
    """Lazy proxy that creates Config on first attribute access."""

    def __getattr__(self, name: str) -> Any:
        _initialize_module()
        if name in _MODULE_STATE:
            return _MODULE_STATE[name]
        return getattr(_create_config(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_ConfigProxy__dict", "_ConfigProxy__weakref__"):
            super().__setattr__(name, value)
        else:
            _initialize_module()
            setattr(_create_config(), name, value)

    def __dir__(self) -> list[str]:
        _initialize_module()
        return list(set(dir(_create_config())) | set(_MODULE_STATE.keys()))


app_config = _ConfigProxy()

# These will be initialized on first access via _initialize_module
# config_as_json, config_digest, config_log_dir, dump_filename, BASE_TIMEFRAME
# are now accessed as app_config.config_as_json, etc.
