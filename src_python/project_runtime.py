from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Iterable, Sequence

__all__ = ["load_shared_c_ext", "resolve_repo_root"]

_DEFAULT_REPO_MARKERS = (
    Path("src_python/project_runtime.py"),
    Path("shared_c/pybind.cpp"),
)
_COMMON_SEARCH_ROOTS = (
    Path("/content"),
    Path("/content/drive"),
    Path("/content/drive/MyDrive"),
)
_DLL_DIR_HANDLES: list[object] = []


def _iter_search_roots(extra_roots: Iterable[Path] = ()) -> Iterable[Path]:
    search_roots: list[Path] = []
    if "__file__" in globals():
        search_roots.append(Path(__file__).resolve().parent)
    search_roots.append(Path.cwd().resolve())
    search_roots.extend(_COMMON_SEARCH_ROOTS)
    search_roots.append(Path.home())
    search_roots.extend(extra_roots)

    seen: set[Path] = set()
    for root in search_roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen or not root.exists():
            continue
        seen.add(root)
        yield root
def _find_repo_root(repo_markers: Sequence[Path]) -> Path | None:
    for root in _iter_search_roots():
        for candidate in (root, *root.parents):
            if all((candidate / marker).exists() for marker in repo_markers):
                return candidate

        marker = repo_markers[0]
        try:
            match = next(root.rglob(marker.as_posix()))
        except (StopIteration, OSError):
            continue

        candidate = match
        for _ in marker.parts:
            candidate = candidate.parent
        if all((candidate / repo_marker).exists() for repo_marker in repo_markers):
            return candidate

    return None
def resolve_repo_root(markers: Sequence[str | Path] | None = None) -> Path:
    repo_markers = tuple(Path(marker) for marker in (markers or _DEFAULT_REPO_MARKERS))

    repo_root = _find_repo_root(repo_markers)
    if repo_root is not None:
        return repo_root

    markers_text = ", ".join(str(marker) for marker in repo_markers)
    raise FileNotFoundError(f"Could not locate the repo root using markers: {markers_text}")


def load_shared_c_ext(root: Path | None = None):
    repo_root = Path(root).resolve() if root is not None else resolve_repo_root()
    build_dir = repo_root / "build"
    build_dir.mkdir(exist_ok=True)

    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))

    compiler = shutil.which("g++")
    if os.name == "nt" and compiler:
        compiler_bin = str(Path(compiler).resolve().parent)
        os.environ["PATH"] = compiler_bin + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            _DLL_DIR_HANDLES.append(add_dll_directory(compiler_bin))

    try:
        return importlib.import_module("shared_c_ext")
    except ImportError:
        if compiler is None:
            raise ModuleNotFoundError(
                "shared_c_ext is not available and g++ was not found. "
                "Build the extension manually or run this in an environment with g++."
            )

        try:
            import pybind11
        except ModuleNotFoundError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "pybind11"],
                check=True,
            )
            import pybind11

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        output_path = build_dir / f"shared_c_ext{ext_suffix}"
        compile_cmd = [compiler, "-O3", "-Wall", "-shared", "-std=c++17"]
        if os.name != "nt":
            compile_cmd.append("-fPIC")
        compile_cmd.extend(
            [
                f"-I{sysconfig.get_paths()['include']}",
                f"-I{pybind11.get_include()}",
                str(repo_root / "shared_c" / "pybind.cpp"),
            ]
        )

        if os.name == "nt":
            python_libdir = sysconfig.get_config_var("LIBDIR")
            python_library = sysconfig.get_config_var("LIBRARY") or sysconfig.get_config_var("LDLIBRARY")
            if python_libdir and python_library:
                lib_name = Path(python_library).stem.removeprefix("lib")
                compile_cmd.extend([f"-L{python_libdir}", f"-l{lib_name}"])

        compile_cmd.extend(["-o", str(output_path)])

        subprocess.run(compile_cmd, cwd=repo_root, check=True)
        importlib.invalidate_caches()
        sys.modules.pop("shared_c_ext", None)
        return importlib.import_module("shared_c_ext")
