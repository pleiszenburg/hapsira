import os
from typing import Any, Generator, Optional, Type

__all__ = [
    "Setting",
    "Settings",
    "settings",
]


def _str2bool(value: str) -> bool:
    """
    Helper for parsing environment variables
    """

    if value.strip().lower() in ("true", "1", "yes", "y"):
        return True
    if value.strip().lower() in ("false", "0", "no", "n"):
        return False

    raise ValueError(f'can not convert value "{value:s}" to bool')


class Setting:
    """
    Holds one setting settable by user before sub-module import
    """

    def __init__(self, name: str, default: Any, options: Optional[tuple[Any]] = None):
        self._name = name
        self._type = type(default)
        self._value = default
        self._options = options
        self._check_env()

    def _check_env(self):
        """
        Check for environment variables
        """
        value = os.environ.get(f"HAPSIRA_{self._name:s}")
        if value is None:
            return
        if self._type is bool:
            value = _str2bool(value)
        self.value = value  # Run through setter for checks!

    @property
    def name(self) -> str:
        """
        Return name of setting
        """
        return self._name

    @property
    def type_(self) -> Type:
        """
        Return type of setting
        """
        return self._type

    @property
    def options(self) -> Optional[tuple[Any]]:
        """
        Return options for value
        """
        return self._options

    @property
    def value(self) -> Any:
        """
        Change value of setting
        """
        return self._value

    @value.setter
    def value(self, new_value: Any):
        """
        Return value of setting
        """
        if not isinstance(new_value, self._type):
            raise TypeError(
                f'"{repr():s}" has type "{repr():s}", expected type"{repr():s}"'
            )
        if self._options is not None and new_value not in self._options:
            raise ValueError(
                f'value "{repr(new_value):s}" not a valid option, valid options are "{repr(self._options):s}"'
            )
        self._value = new_value


class Settings:
    """
    Holds settings settable by user before sub-module import
    """

    def __init__(self):
        self._settings = {}
        self._add(
            Setting(
                "DEBUG",
                False,
            )
        )
        self._add(
            Setting(
                "LOGLEVEL",
                "NOTSET",
                options=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"),
            )
        )
        self._add(
            Setting(
                "TARGET",
                "cpu",
                options=("cpu", "parallel", "cuda"),
            )
        )
        self._add(
            Setting(
                "INLINE",
                self["TARGET"].value == "cuda",
            )
        )
        self._add(
            Setting(
                "NOPYTHON",
                True,
            )
        )

    def _add(self, setting: Setting):
        """
        Add new setting
        """
        self._settings[setting.name] = setting

    def __getitem__(self, name: str) -> Setting:
        """
        Return setting by name
        """
        if name not in self._settings.keys():
            raise KeyError(
                f'setting "{name:s}" unknown, possible settings are {repr(list(self._settings.keys())):s}'
            )
        return self._settings[name]

    def keys(self) -> Generator:
        """
        Generator of all setting names
        """
        return (name for name in self._settings.keys())


settings = Settings()
