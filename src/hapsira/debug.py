import logging
import os

__all__ = [
    "DEBUG",
    "LOGLEVEL",
    "get_environ_switch",
    "logger",
]


def get_environ_switch(name: str, default: bool) -> bool:
    """
    Helper for parsing environment variables
    """

    value = os.environ.get(name, "1" if default else "0")

    if value.strip().lower() in ("true", "1", "yes"):
        return True
    if value.strip().lower() in ("false", "0", "no"):
        return False

    raise ValueError(f'can not convert value "{value:s}" to bool')


DEBUG = get_environ_switch("HAPSIRA_DEBUG", default=False)
LOGLEVEL = os.environ.get("HAPSIRA_LOGLEVEL", "WARNING")
if LOGLEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"):
    raise ValueError(f'Unknown loglevel "{LOGLEVEL:s}"')

logger = logging.getLogger("hapsira")
if LOGLEVEL != "NOTSET":
    logger.setLevel(logging.DEBUG if DEBUG else getattr(logging, LOGLEVEL))

logger.debug("debug mode: %s", "on" if DEBUG else "off")
logger.debug("logging level: %s", LOGLEVEL)
