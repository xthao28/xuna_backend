"""Process configuration from `XUNA_*` environment variables only.

No secret has a default value; secrets never appear in source or logs.
"""

import os
from dataclasses import dataclass
from enum import StrEnum


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class Settings:
    environment: Environment
    database_url: str
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        database_url = source.get("XUNA_DATABASE_URL", "")
        if not database_url:
            raise RuntimeError("XUNA_DATABASE_URL is required")
        return cls(
            environment=Environment(source.get("XUNA_ENV", Environment.LOCAL)),
            database_url=database_url,
            log_level=source.get("XUNA_LOG_LEVEL", "INFO"),
        )
