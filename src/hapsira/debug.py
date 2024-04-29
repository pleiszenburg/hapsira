import logging

from hapsira.settings import settings

__all__ = [
    "logger",
]

logger = logging.getLogger("hapsira")

if settings["LOGLEVEL"].value != "NOTSET":
    logger.setLevel(
        logging.DEBUG
        if settings["DEBUG"].value
        else getattr(logging, settings["LOGLEVEL"].value)
    )

logger.debug("logging level: %s", logging.getLevelName(logger.level))
logger.debug("debug mode: %s", "on" if settings["DEBUG"].value else "off")
