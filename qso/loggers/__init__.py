from os import PathLike
import pathlib

from pprint import pformat
from orjson import dumps
from tqdm import tqdm
from typing import Any, Callable


class Logger():
    """
    This class is the base class for all logging utilities provided in this
    package.
    """

    def __init__(self, **metadata) -> None:
        assert "iterations" not in metadata, (
            "The `iterations` keyword is reserved for the per-iteration "
            "data.")

        self.state: dict[str, Any] = metadata
        self.state['iterations'] = []

        self.step_hooks: "list[Callable[[Logger], None]]" = []

    def save_json(self, path: str | PathLike, overwrite: bool = False):
        """
        Save the current log to a JSON file.

        Parameters
        ---
        - `path` (`str | os.PathLike`): The location to save the log to.
        - `overwrite` (`bool`): If `True` and the file already exists, it will
          be overwritten. If `False` and the file already exists, a
          `FileExistsError` will be raised. By default, this is `False`.
        """
        if pathlib.Path(path).exists() and not overwrite:
            raise FileExistsError(path)

        with open(path, 'wb') as f:
            f.write(dumps(self.state, default=repr))

    def log_step(self, state: dict[str, Any], **kwargs):
        self.state['iterations'].append(state | kwargs)

        for callable in self.step_hooks:
            callable(self)

    def __getitem__(self, idx: str) -> Any:
        return self.state[idx]

    def __setitem__(self, idx: str, val: Any):
        self.state[idx] = val

    def __delitem__(self, idx: str):
        del self.state[idx]

    def __contains__(self, idx: Any) -> bool:
        return idx in self.state

    def register_hook(self, callable: "Callable[[Logger], None]"):
        self.step_hooks.append(callable)


class PrettyPrint(Logger):
    """
    This class also prints out information at each iteration.
    """

    def __init__(self, **metadata) -> None:
        super().__init__(**metadata)

        assert 'steps' in metadata, "Expected to find the `n_steps` keyword in `metadata`."
        self.bar = tqdm(total=metadata['steps'])

    def log_step(self, state: dict[str, Any], **kwargs):
        """
        Parameters
        ---
        - `state` (`dict[str, Any]`): The contents of this dictionary are
          pretty printed at each step in addition to being added to the log
          (`self.state`).
        - `**kwargs` (`dict[str, Any]`): Data to be stored in the log
          (`self.state`).
        """
        super().log_step(state, **kwargs)

        self.bar.write(pformat(state, sort_dicts=False))
        self.bar.update()
