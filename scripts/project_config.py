from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 compatibility for the ACTS LCG view.
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "default.toml"
LOCAL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "local.toml"


def activate_cli_config(argv: list[str] | None = None) -> Path | None:
    arguments = list(sys.argv if argv is None else argv)
    selected: str | None = None
    for index, argument in enumerate(arguments):
        if argument == "--config":
            if index + 1 >= len(arguments):
                raise ValueError("--config requires a path")
            selected = arguments[index + 1]
        elif argument.startswith("--config="):
            selected = argument.split("=", maxsplit=1)[1]
    if selected is None:
        return None
    path = Path(selected).expanduser().resolve()
    os.environ["ML_ALIGNMENT_CONFIG"] = str(path)
    return path


def add_config_argument(parser) -> None:
    parser.add_argument(
        "--config",
        help=(
            "Path to a TOML project configuration override. The value is loaded "
            "before artifact defaults are constructed."
        ),
    )


@dataclass(frozen=True)
class ProjectConfig:
    repository_root: Path
    data_root: Path
    acts_root: Path
    target_layers: tuple[tuple[int, int], ...]
    active_dof: tuple[int, ...]
    source_path: Path

    @property
    def target_layer_suffix(self) -> str:
        return "_".join(f"v{volume}l{layer}" for volume, layer in self.target_layers)

    @property
    def target_layer_dir(self) -> str:
        return "__".join(
            f"Vol{volume}_Lay{layer}" for volume, layer in self.target_layers
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository_root": str(self.repository_root),
            "data_root": str(self.data_root),
            "acts_root": str(self.acts_root),
            "target_layers": [
                f"{volume}:{layer}" for volume, layer in self.target_layers
            ],
            "active_dof": list(self.active_dof),
            "target_layer_suffix": self.target_layer_suffix,
            "target_layer_dir": self.target_layer_dir,
            "source_path": str(self.source_path),
        }


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_project_path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _parse_target_layers(values: list[str]) -> tuple[tuple[int, int], ...]:
    layers: list[tuple[int, int]] = []
    for value in values:
        try:
            volume_text, layer_text = value.split(":", maxsplit=1)
            volume, layer = int(volume_text), int(layer_text)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"target layer {value!r} must use positive volume:layer syntax"
            ) from error
        if volume <= 0 or layer <= 0:
            raise ValueError("target layers require positive volume:layer values")
        layers.append((volume, layer))
    if not layers:
        raise ValueError("at least one target layer is required")
    if len(set(layers)) != len(layers):
        raise ValueError("duplicate target layers are not allowed")
    return tuple(layers)


def _parse_active_dof(values: list[int]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate active degrees of freedom are not allowed")
    if any(value < 0 or value > 5 for value in parsed):
        raise ValueError("active degrees of freedom must be 0 through 5")
    return parsed


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    data = _read_toml(DEFAULT_CONFIG_PATH)
    source_path = DEFAULT_CONFIG_PATH

    if LOCAL_CONFIG_PATH.exists():
        data = _merge(data, _read_toml(LOCAL_CONFIG_PATH))
        source_path = LOCAL_CONFIG_PATH

    selected_path: Path | None
    if path is not None:
        selected_path = Path(path).expanduser().resolve()
    else:
        environment_path = os.environ.get("ML_ALIGNMENT_CONFIG")
        selected_path = (
            Path(environment_path).expanduser().resolve()
            if environment_path
            else None
        )
    if selected_path is not None:
        if not selected_path.is_file():
            raise FileNotFoundError(f"configuration file not found: {selected_path}")
        data = _merge(data, _read_toml(selected_path))
        source_path = selected_path

    project = data.get("project", {})
    acts = data.get("acts", {})
    alignment = data.get("alignment", {})
    return ProjectConfig(
        repository_root=REPOSITORY_ROOT,
        data_root=_resolve_project_path(project.get("data_root", "data")),
        acts_root=_resolve_project_path(acts.get("root", "../alignTest")),
        target_layers=_parse_target_layers(alignment.get("target_layers", [])),
        active_dof=_parse_active_dof(alignment.get("active_dof", [])),
        source_path=source_path,
    )
