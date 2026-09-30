"""Content-addressed dumping of `pandera_validate` return values for debugging.

Layout (a component is a path segment, ``.`` joins segments, ``=`` joins a class
to its method and a nested path to its leaf)::

    src/abc/def/ghi.py::Abc.jkl   -> src.abc.def.ghi.Abc=jkl.<hash>.parquet
    src/abc/def/jkl.py::mno      -> src.abc.def.jkl.<hash>.parquet
    src/abc/def/ghi.py::stu      -> src.abc.def.ghi.stu.3.<hash>.parquet
                                      src.abc.def.ghi.stu.<hash>.json
    src/abc/def/pqr.py::mno      -> src.abc.def.pqr.mno.d=f=3.<hash>.parquet
                                      src.abc.def.pqr.mno.<hash>.json

``=`` is used where the layout spec writes ``->``: ``>`` is an illegal
character in Windows filenames, which would make every class-method dump
impossible to create on that platform.

A single DataFrame/ndarray return produces only its Parquet dump. A composite
return produces one JSON descriptor for the whole output plus one Parquet dump
per DataFrame/ndarray found anywhere inside it, the frames being replaced in the
JSON by a lightweight reference to the Parquet file that holds them. Component
characters outside ``[A-Za-z0-9._=-]`` (``<locals>`` of a nested function, a
quoted dict key, a non-ASCII identifier) become ``_``.
"""

from __future__ import annotations

import inspect
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from types import FunctionType
from typing import cast

import numpy as np
import pandas as pd
from br_py_log_n_profile import log_w

from helper.content_hash import content_hash
from helper.dump_folder import get_dump_folder
from helper.repo_root import find_repo_root_or_parent

Array = np.ndarray[tuple[int, ...], np.dtype[np.generic]]  # type: ignore[explicit-any]
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
PARQUET_SUFFIX = ".parquet"
JSON_SUFFIX = ".json"
COMPONENT_SEPARATOR = "="
REFERENCE_KEY = "output_dump_reference"
NDARRAY_COLUMN_PREFIX = "value"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._=-]")


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _safe_name_part(text: str) -> str:
    """Reduce one path component to characters that are legal in a filename everywhere."""
    return _UNSAFE_NAME_CHARS.sub("_", text) or "_"


def namespace_path(source_file: Path) -> str:
    """``/repo/src/abc/def/ghi.py`` -> ``src.abc.def.ghi``.

    The path is taken relative to the Git root owning ``source_file``, so a
    library installed elsewhere cannot influence the namespace.
    """
    root = find_repo_root_or_parent(source_file)
    resolved = source_file.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        relative = Path(resolved.name)
    parts = (*(_safe_name_part(part) for part in relative.parts[:-1]), _safe_name_part(relative.stem))
    return ".".join(parts)


def build_base_name(source_file: Path, qualname: str) -> str:
    """``namespace`` for a module-level function, ``namespace.Class=method`` for a method."""
    namespace = namespace_path(source_file)
    owner, _, name = qualname.rpartition(".")
    if not owner:
        return f"{namespace}.{_safe_name_part(name)}"
    return f"{namespace}.{_safe_name_part(owner)}{COMPONENT_SEPARATOR}{_safe_name_part(name)}"


def _child_name(base_name: str, path: tuple[str, ...]) -> str:
    """``base`` for a single component, ``base.d=f=3`` for a nested one."""
    return f"{base_name}.{COMPONENT_SEPARATOR.join(_safe_name_part(component) for component in path)}"


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _frame_from_array(array: Array) -> pd.DataFrame:
    """Tabular view of a 1-D or 2-D array, keeping values and shape."""
    if array.ndim == 1:
        return pd.DataFrame({NDARRAY_COLUMN_PREFIX: array})
    return pd.DataFrame(array, columns=[f"{NDARRAY_COLUMN_PREFIX}{index}" for index in range(array.shape[1])])


def _frame_with_flat_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Store a real index as columns, matching the flat on-disk convention of parquet_normalization."""
    if isinstance(frame.index, pd.RangeIndex):
        return frame
    try:
        return frame.reset_index()
    except ValueError:
        return frame


def _extractable_frame(value: pd.DataFrame | Array) -> pd.DataFrame:
    if isinstance(value, np.ndarray):
        return _frame_from_array(value)
    return _frame_with_flat_index(value)


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Walking the returned value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Extracted:
    path: tuple[str, ...]
    frame: pd.DataFrame


def _iter_components[Value](value: dict[str, Value] | list[Value] | tuple[Value, ...]) -> Iterator[tuple[str, Value]]:
    """Child components of a composite, as ``(path component, value)`` pairs.

    A dict key contributes its name, a tuple/list element its 1-based index.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
        return
    for position, item in enumerate(value, start=1):
        yield str(position), item


def _collect_frames[Value](value: Value, path: tuple[str, ...]) -> list[_Extracted]:
    if isinstance(value, (pd.DataFrame, np.ndarray)) and value.ndim >= 1:
        return [_Extracted(path, _extractable_frame(value))]
    if not isinstance(value, (dict, list, tuple)):
        return []
    found: list[_Extracted] = []
    for component, item in _iter_components(value):
        found.extend(_collect_frames(item, (*path, component)))
    return found


def _json_scalar[Value](value: Value) -> JsonValue:
    """JSON-safe stand-in for a leaf that is not stored in its own Parquet file."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date, time)):
        return value.isoformat()
    return _json_other(value)


def _json_other[Value](value: Value) -> JsonValue:
    """Fallback for arrays too high-dimensional to dump, and for exotic leaves."""
    if isinstance(value, np.ndarray):
        return cast(JsonValue, value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (set, frozenset)):
        return [repr(item) for item in sorted(value, key=repr)]
    return repr(value)


def _reference(value: pd.DataFrame | Array, parquet: Path | None) -> JsonValue:
    return {
        REFERENCE_KEY: {
            "parquet": None if parquet is None else parquet.name,
            "type": type(value).__name__,
            "shape": list(value.shape),
        }
    }


def _json_value[Value](value: Value, path: tuple[str, ...], references: dict[tuple[str, ...], Path]) -> JsonValue:
    """Same structure as ``value``, with every frame replaced by a reference."""
    if isinstance(value, (pd.DataFrame, np.ndarray)) and value.ndim >= 1:
        return _reference(value, references.get(path))
    if isinstance(value, dict):
        return {str(key): _json_value(item, (*path, str(key)), references) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, (*path, str(position)), references) for position, item in enumerate(value, start=1)]
    return _json_scalar(value)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write(folder: Path, base_name: str, suffix: str, payload: bytes) -> Path:
    path = folder / f"{base_name}.{content_hash(payload)}{suffix}"
    path.write_bytes(payload)
    return path


def _write_frame(folder: Path, base_name: str, frame: pd.DataFrame) -> Path:
    return _write(folder, base_name, PARQUET_SUFFIX, _parquet_payload(frame))


def _write_json(folder: Path, base_name: str, value: JsonValue) -> Path:
    payload = json.dumps(value, sort_keys=True, indent=2, default=repr).encode("utf-8")
    return _write(folder, base_name, JSON_SUFFIX, payload)


def _source_file(func: FunctionType) -> Path:
    return Path(inspect.getsourcefile(func) or inspect.getfile(func))


def _write_frames(folder: Path, base_name: str, frames: list[_Extracted]) -> dict[tuple[str, ...], Path]:
    """One Parquet dump per frame; a frame no encoder accepts is reported and skipped."""
    written: dict[tuple[str, ...], Path] = {}
    for item in frames:
        try:
            written[item.path] = _write_frame(folder, _child_name(base_name, item.path), item.frame)
        except Exception as error:
            log_w(f"{base_name}.{'.'.join(item.path)}: parquet dump failed: {error!r}")
    return written


def _dump[Output](func: FunctionType, result: Output) -> tuple[Path, ...]:
    source_file = _source_file(func)
    folder = get_dump_folder(source_file)
    base_name = build_base_name(source_file, func.__qualname__)
    if isinstance(result, (pd.DataFrame, np.ndarray)) and result.ndim >= 1:
        return (_write_frame(folder, base_name, _extractable_frame(result)),)
    references = _write_frames(folder, base_name, _collect_frames(result, ()))
    return (*references.values(), _write_json(folder, base_name, _json_value(result, (), references)))


def dump_function_output[Output](func: FunctionType, result: Output) -> tuple[Path, ...]:
    """Write the content-addressed dump of ``result`` returned by ``func``.

    Returns the files written, in order. A dumping failure (unwritable folder,
    an unserializable leaf) is logged and swallowed: debugging must never change
    what the decorated function returns or raises.
    """
    try:
        return _dump(func, result)
    except Exception as error:
        log_w(f"{func.__qualname__}: output dump failed: {error!r}")
        return ()
