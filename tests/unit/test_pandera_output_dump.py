"""Output-dump behaviour of @pandera_validate(dump_output=True).

Every test decorates functions defined in a throw-away module written inside a
throw-away Git repository, which is the only way to exercise the real code
path: the layout is derived from the source file of the *decorated* function.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from br_pre_commit.src.br_pandera.dump_folder import configure_pandera_dump_folder, configured_dump_folder
from br_pre_commit.src.helper.content_hash import HASH_LENGTH, content_hash, is_content_hash
from br_pre_commit.src.helper.output_dump import build_base_name, namespace_path

SOURCE_HEADER = """\
import numpy as np
import pandas as pd

from br_pre_commit import pandera_validate
"""
_MODULE_IDS = itertools.count()
_LIBRARY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_dump_folder():
    configure_pandera_dump_folder(None)
    yield
    configure_pandera_dump_folder(None)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in for the Git repository of the project importing this library."""
    (tmp_path / ".git").mkdir()
    return tmp_path


def _write_module(root: Path, relative_path: str, body: str) -> Path:
    source_file = root / relative_path
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(f"{SOURCE_HEADER}{body}", encoding="utf-8")
    return source_file


def _load(source_file: Path) -> ModuleType:
    """Import the throw-away module the way an importing project would."""
    module_name = f"dumped_{source_file.stem}_{next(_MODULE_IDS)}"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _dump_files(project: Path) -> list[Path]:
    folder = project / "logs" / "output_dump"
    return sorted(folder.iterdir()) if folder.is_dir() else []


def _names(project: Path) -> list[str]:
    return [path.name for path in _dump_files(project)]


def _hashes(names: list[str]) -> list[str]:
    return [name.split(".")[-2] for name in names]


def _json_of(project: Path, name: str) -> object:
    return json.loads((project / "logs" / "output_dump" / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Disabled / unchanged behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dump_disabled_creates_nothing(project: Path):
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2]})
""",
    )
    module = _load(source_file)

    frame = module.load()

    assert frame["a"].tolist() == [1, 2]
    assert _dump_files(project) == []
    assert not (project / "logs").exists()


@pytest.mark.unit
def test_disabled_dump_leaves_return_value_untouched(project: Path):
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2]})
""",
    )
    module = _load(source_file)

    frame = module.load()

    assert list(frame.columns) == ["a"]
    assert frame.index.tolist() == [0, 1]


@pytest.mark.unit
def test_dump_failure_does_not_break_the_call(project: Path):
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2]})
""",
    )
    module = _load(source_file)
    (project / "logs").mkdir()
    (project / "logs" / "output_dump").write_text("not a directory", encoding="utf-8")

    assert module.load()["a"].tolist() == [1, 2]
    assert (project / "logs" / "output_dump").read_text(encoding="utf-8") == "not a directory"


# ---------------------------------------------------------------------------
# Folder resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_folder_is_repo_root_logs_output_dump(project: Path):
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})
""",
    )
    module = _load(source_file)

    module.load()

    assert [path.parent for path in _dump_files(project)] == [project / "logs" / "output_dump"]


@pytest.mark.unit
def test_dump_follows_the_decorated_file_not_the_library_or_cwd(project: Path, monkeypatch: pytest.MonkeyPatch):
    """PyPI install outside the caller's repo: the dump belongs to the importing project."""
    monkeypatch.chdir(Path(__file__).parent)
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})
""",
    )
    module = _load(source_file)

    module.load()

    assert _names(project)[0].startswith("src.app.model.load.")
    assert not (_LIBRARY_ROOT / "logs" / "output_dump").exists()


@pytest.mark.unit
def test_custom_absolute_folder(project: Path, tmp_path_factory: pytest.TempPathFactory):
    absolute = tmp_path_factory.mktemp("absolute_dumps")
    configure_pandera_dump_folder(absolute)
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})
""",
    )
    module = _load(source_file)

    module.load()

    assert [path.parent for path in absolute.iterdir()] == [absolute]
    assert _dump_files(project) == []


@pytest.mark.unit
def test_custom_relative_folder_resolves_against_repo_root(project: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(Path(__file__).parent)
    configure_pandera_dump_folder("var/debug_frames")
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})
""",
    )
    module = _load(source_file)

    module.load()

    dumped = project / "var" / "debug_frames"
    assert [path.parent for path in dumped.iterdir()] == [dumped]
    assert _dump_files(project) == []


@pytest.mark.unit
def test_configuration_applies_to_later_dumps(project: Path):
    configure_pandera_dump_folder("var/first")
    source_file = _write_module(
        project,
        "src/app/model.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})
""",
    )
    module = _load(source_file)
    module.load()
    configure_pandera_dump_folder("var/second")
    module.load()

    assert len(list((project / "var" / "first").iterdir())) == 1
    assert len(list((project / "var" / "second").iterdir())) == 1


@pytest.mark.unit
def test_configurator_accepts_str_and_path():
    configure_pandera_dump_folder("var/frames")
    assert configured_dump_folder() == Path("var/frames")

    configure_pandera_dump_folder(Path("var") / "other")
    assert configured_dump_folder() == Path("var/other")

    configure_pandera_dump_folder(None)
    assert configured_dump_folder() is None


# ---------------------------------------------------------------------------
# Repository-root detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_namespace_of_a_nested_file(project: Path):
    source_file = _write_module(project, "src/abc/def/ghi.py", "")

    assert namespace_path(source_file) == "src.abc.def.ghi"


@pytest.mark.unit
def test_repo_root_ignores_the_process_working_directory(
    project: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    source_file = _write_module(project, "src/abc/def/ghi.py", "")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / ".git").mkdir()
    monkeypatch.chdir(elsewhere)

    assert namespace_path(source_file) == "src.abc.def.ghi"


@pytest.mark.unit
def test_inner_repository_wins_over_the_outer_one(project: Path):
    outer = _write_module(project, "outer.py", "")
    inner_root = project / "packages" / "inner"
    inner_root.mkdir(parents=True)
    (inner_root / ".git").mkdir()
    inner = _write_module(inner_root, "src/app.py", "")

    assert namespace_path(outer) == "outer"
    assert namespace_path(inner) == "src.app"


@pytest.mark.unit
def test_file_outside_any_repository_uses_its_own_directory(tmp_path: Path):
    source_file = _write_module(tmp_path / "scripts", "tool.py", "")

    assert namespace_path(source_file) == "tool"


@pytest.mark.unit
def test_repo_root_detection_is_stable_across_calls(project: Path):
    from br_pre_commit.src.helper.repo_root import find_repo_root

    first = find_repo_root(project / "src" / "app" / "model.py")
    second = find_repo_root(project / "src" / "app" / "model.py")

    assert first == project.resolve()
    assert second == first


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_function_base_name(project: Path):
    source_file = _write_module(project, "src/abc/def/jkl.py", "")

    assert build_base_name(source_file, "mno") == "src.abc.def.jkl.mno"


@pytest.mark.unit
def test_method_base_name(project: Path):
    source_file = _write_module(project, "src/abc/def/ghi.py", "")

    assert build_base_name(source_file, "Abc.jkl") == "src.abc.def.ghi.Abc=jkl"


@pytest.mark.unit
def test_nested_function_name_stays_filename_legal(project: Path):
    source_file = _write_module(project, "src/abc/def/ghi.py", "")

    base = build_base_name(source_file, "outer.<locals>.inner")

    assert base == "src.abc.def.ghi.outer._locals_=inner"
    assert not set(base) & set('<>:"/\\|?*')


# ---------------------------------------------------------------------------
# Single frame returns
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_dataframe_writes_only_parquet(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def jkl() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2]})
""",
    )
    module = _load(source_file)

    module.jkl()

    names = _names(project)
    assert len(names) == 1
    assert names[0].startswith("src.abc.def.ghi.jkl.")
    assert names[0].endswith(".parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(_dump_files(project)[0]), pd.DataFrame({"a": [1, 2]}))


@pytest.mark.unit
def test_single_ndarray_writes_only_parquet(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def jkl():
    return np.arange(6).reshape(3, 2)
""",
    )
    module = _load(source_file)

    module.jkl()

    names = _names(project)
    assert len(names) == 1
    assert names[0].startswith("src.abc.def.ghi.jkl.")
    assert names[0].endswith(".parquet")
    assert pd.read_parquet(_dump_files(project)[0]).to_numpy().tolist() == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.unit
def test_one_dimensional_ndarray_round_trips(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def jkl():
    return np.arange(4)
""",
    )
    module = _load(source_file)

    module.jkl()

    assert pd.read_parquet(_dump_files(project)[0]).to_numpy().tolist() == [[0], [1], [2], [3]]


@pytest.mark.unit
def test_class_method_dump(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

class Abc:
    @pandera_validate(allow_pandas_dataframe=True, dump_output=True)
    def jkl(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1]})
""",
    )
    module = _load(source_file)

    module.Abc().jkl()

    assert _names(project)[0].startswith("src.abc.def.ghi.Abc=jkl.")


@pytest.mark.unit
def test_dataframe_index_is_kept_as_a_column(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def jkl() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="label"))
""",
    )
    module = _load(source_file)

    module.jkl()

    assert pd.read_parquet(_dump_files(project)[0]).columns.tolist() == ["label", "a"]


# ---------------------------------------------------------------------------
# Composite returns
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tuple_with_dataframe_writes_json_and_parquet(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return (1, 2.5, pd.DataFrame({"a": [1, 2]}))
""",
    )
    module = _load(source_file)

    module.stu()

    names = _names(project)
    parquet = [name for name in names if name.endswith(".parquet")]
    descriptor = [name for name in names if name.endswith(".json")]
    assert len(parquet) == 1 and len(descriptor) == 1
    assert parquet[0].startswith("src.abc.def.ghi.stu.3.")
    assert descriptor[0].startswith("src.abc.def.ghi.stu.")
    payload = _json_of(project, descriptor[0])
    assert payload[:2] == [1, 2.5]
    assert payload[2]["output_dump_reference"]["parquet"] == parquet[0]


@pytest.mark.unit
def test_nested_dict_tuple_list_paths(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/pqr.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def mno():
    return {
        "a": 1,
        "b": "akmjahkjshakjsh",
        "d": {"e": "text", "f": (0, [7, np.arange(3)], pd.DataFrame({"c": [1]}))},
    }
""",
    )
    module = _load(source_file)

    module.mno()

    names = _names(project)
    parquet = sorted(name for name in names if name.endswith(".parquet"))
    assert [name[: -len(".parquet")].rsplit(".", 1)[0].split(".")[-1] for name in parquet] == ["d=f=2=2", "d=f=3"]
    assert all(name.startswith("src.abc.def.pqr.mno.") for name in parquet)
    payload = _json_of(project, next(name for name in names if name.endswith(".json")))
    assert payload["a"] == 1
    assert payload["b"] == "akmjahkjshakjsh"
    assert payload["d"]["e"] == "text"
    assert payload["d"]["f"][0] == 0
    assert payload["d"]["f"][1][0] == 7


@pytest.mark.unit
def test_multiple_frames_each_get_their_own_file(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return {
        "one": pd.DataFrame({"a": [1]}),
        "two": np.arange(3),
        "three": pd.DataFrame({"b": [1, 2]}),
    }
""",
    )
    module = _load(source_file)

    module.stu()

    names = _names(project)
    parquet = sorted(name for name in names if name.endswith(".parquet"))
    assert len(parquet) == 3
    assert [name[: -len(".parquet")].split(".")[-2] for name in parquet] == ["one", "three", "two"]


@pytest.mark.unit
def test_json_never_contains_frame_data(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return (pd.DataFrame({"secret_column": list(range(50))}),)
""",
    )
    module = _load(source_file)

    module.stu()

    descriptor = next(path for path in _dump_files(project) if path.suffix == ".json")
    assert "secret_column" not in descriptor.read_text(encoding="utf-8")


@pytest.mark.unit
def test_non_frame_leaf_values_are_preserved(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return {
        "int": 3,
        "float": 1.5,
        "none": None,
        "flag": True,
        "numpy": np.int64(7),
        "when": pd.Timestamp("2024-01-01"),
    }
""",
    )
    module = _load(source_file)

    module.stu()

    assert _json_of(project, _names(project)[0]) == {
        "int": 3,
        "float": 1.5,
        "none": None,
        "flag": True,
        "numpy": 7,
        "when": "2024-01-01T00:00:00",
    }


@pytest.mark.unit
def test_unencodable_frame_is_skipped_but_the_json_is_written(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

class Opaque:
    def __repr__(self):
        return "opaque"


@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return (pd.DataFrame({"a": [Opaque()]}), {"b": 1})
""",
    )
    module = _load(source_file)

    module.stu()

    names = _names(project)
    assert len(names) == 1
    assert names[0].endswith(".json")
    assert _json_of(project, names[0])[0]["output_dump_reference"]["parquet"] is None


@pytest.mark.unit
def test_scalar_return_writes_only_json(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(dump_output=True)
def stu():
    return {"a": 1}
""",
    )
    module = _load(source_file)

    module.stu()

    names = _names(project)
    assert len(names) == 1
    assert names[0].startswith("src.abc.def.ghi.stu.")
    assert names[0].endswith(".json")


# ---------------------------------------------------------------------------
# Content hashing / versions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hash_is_seven_hex_characters(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return (pd.DataFrame({"a": [1]}),)
""",
    )
    module = _load(source_file)

    module.stu()

    assert len(_names(project)) == 2
    for digest in _hashes(_names(project)):
        assert len(digest) == HASH_LENGTH == 7
        assert is_content_hash(digest)
    assert content_hash(b"payload") == content_hash(b"payload")
    assert content_hash(b"payload") != content_hash(b"payloae")


@pytest.mark.unit
def test_identical_output_reuses_the_same_file(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return (pd.DataFrame({"a": [1, 2]}), {"b": 3})
""",
    )
    module = _load(source_file)

    module.stu()
    first_names = _names(project)
    module.stu()

    assert len(first_names) == 2
    assert _names(project) == first_names
    pd.testing.assert_frame_equal(pd.read_parquet(_dump_files(project)[0]), pd.DataFrame({"a": [1, 2]}))


@pytest.mark.unit
def test_changed_output_keeps_both_versions(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu(rows):
    return (pd.DataFrame({"a": list(range(rows))}), {"rows": rows})
""",
    )
    module = _load(source_file)

    module.stu(3)
    module.stu(4)

    parquet = sorted(name for name in _names(project) if name.endswith(".parquet"))
    assert len(parquet) == 2
    assert len([name for name in _names(project) if name.endswith(".json")]) == 2
    assert pd.read_parquet(project / "logs" / "output_dump" / parquet[0]).shape == (3, 1)
    assert pd.read_parquet(project / "logs" / "output_dump" / parquet[1]).shape == (4, 1)


@pytest.mark.unit
def test_each_extracted_frame_is_hashed_on_its_own_content(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu(rows):
    return (pd.DataFrame({"a": list(range(rows))}), pd.DataFrame({"b": [1, 2]}))
""",
    )
    module = _load(source_file)

    module.stu(2)
    module.stu(3)

    parquet = sorted(name for name in _names(project) if name.endswith(".parquet"))
    bases = [name[: -len(".parquet")].rsplit(".", 1)[0] for name in parquet]
    assert len(parquet) == 3
    assert bases.count("src.abc.def.ghi.stu.1") == 2
    assert bases.count("src.abc.def.ghi.stu.2") == 1
    assert len({name[: -len(".parquet")] for name in parquet}) == 3


@pytest.mark.unit
def test_json_hash_is_deterministic_across_key_order(project: Path):
    source_file = _write_module(
        project,
        "src/abc/def/ghi.py",
        """

@pandera_validate(allow_pandas_dataframe=True, dump_output=True)
def stu():
    return {"b": 1, "a": 2, "d": {"f": 1, "e": 2}}
""",
    )
    module = _load(source_file)
    module.stu()
    first = _names(project)
    module.stu()

    assert _names(project) == first

    source_file.write_text(
        source_file.read_text(encoding="utf-8").replace(
            '{"b": 1, "a": 2, "d": {"f": 1, "e": 2}}', '{"d": {"e": 2, "f": 1}, "a": 2, "b": 1}'
        ),
        encoding="utf-8",
    )
    _load(source_file).stu()

    assert _names(project) == first
