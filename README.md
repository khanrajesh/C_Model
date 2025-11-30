# C++ ML Models With Pybind11 (Windows Quickstart)

Build core machine learning models in C++ (math helpers in C), expose them to Python via pybind11, and compare against popular Python libraries. Everything below is expressed as run-once commands so a fresh machine can build and run the code.

## What We Are Doing Next
- Keep models in C++ for performance; keep low-level math helpers in C.
- Export the C++ implementations to Python with pybind11.
- Add Python notebooks and scripts to benchmark against common libraries (NumPy/Pandas/Scikit-learn equivalents later).

## Project Layout
- `build/` – compiled binaries and Python extension artifacts.
- `data/` – CSV datasets (`common_linear_regression.csv`, `Student_Performance.csv`).
- `shared_c/` – shared C/C++ sources and the pybind entry (`pybind.cpp`).
- `src_c/` – C++ entrypoint (`main.cpp`).
- `src_python/` – Python entrypoint (`main.py` placeholder).
- `src_notebook/` – Jupyter notebook (`main.ipynb`).

## Prerequisites
- Windows 10/11 64-bit.
- One C++ toolchain:
  - **MSVC**: Visual Studio 2019/2022 with “Desktop development with C++” (Build Tools is fine), includes `cl` and the “x64 Native Tools” prompt.
  - **or MinGW-w64 g++** 11+ in `PATH`.
- **Python 3.10+** with `pip`.
- **CMake** 3.20+ (optional, only if you prefer CMake builds).
- Python packages: `pybind11`, `numpy`, `pandas`, `jupyter`.

Quick checks:
```powershell
cl 2>$null          # should print Microsoft (R) C/C++ Optimizing Compiler
g++ --version       # optional alternative toolchain
python --version
cmake --version     # optional
```

## Setup (run once)
```powershell
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"

# Optional: isolate Python packages
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install pybind11 numpy pandas jupyter

mkdir build
```

## Build and Run Commands (PowerShell)
Pick MSVC (recommended on Windows) or MinGW for compilation. Run each block independently.

### A) Build and run the native C++ app (MSVC)
```powershell
# Open "x64 Native Tools Command Prompt for VS 2022" (or run vcvars64.bat), then:
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"
cl /std:c++17 /EHsc src_c\main.cpp /Fe:build\shared_c.exe
.\build\shared_c.exe
```

### A2) Build and run the native C++ app (MinGW g++)
```powershell
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"
g++ -std=c++17 -O2 src_c/main.cpp -o build/shared_c.exe
.\build\shared_c.exe
```

### B) Build the pybind11 Python extension (C++ -> Python, MSVC)
```powershell
# In the MSVC dev prompt (cl available)
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"

# Gather Python and pybind11 paths
$PYTHON_INCLUDE = python -c "import sysconfig; print(sysconfig.get_paths()['include'])"
$PYTHON_LIBDIR  = python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
$PYTHON_LIB     = python -c "import sysconfig; print(sysconfig.get_config_var('LIBRARY'))"
$PYBIND11_INC   = python -c "import pybind11; print(pybind11.get_include())"

cl /std:c++17 /EHsc /LD shared_c\pybind.cpp `
  /I"$PYTHON_INCLUDE" `
  /I"$PYBIND11_INC" `
  /Ishared_c `
  /link /OUT:build\shared_c_ext.pyd `
        /IMPLIB:build\shared_c_ext.lib `
        /LIBPATH:"$PYTHON_LIBDIR" "$PYTHON_LIB"
```
Outputs: `build/shared_c_ext.pyd`, `build/shared_c_ext.lib`, `build/shared_c_ext.exp`.

### B2) Build the pybind11 Python extension (C++ -> Python, MinGW g++)
```powershell
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"
$PYTHON_INCLUDE = python -c "import sysconfig; print(sysconfig.get_paths()['include'])"
$PYTHON_LIBDIR  = python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
$PYTHON_LIB_BASENAME = python -c "import os, sysconfig; lib = sysconfig.get_config_var('LIBRARY'); print(os.path.splitext(lib)[0].lstrip('lib'))"
$PYBIND11_INC   = python -c "import pybind11; print(pybind11.get_include())"

g++ -O3 -Wall -shared -std=c++17 `
  -I"$PYTHON_INCLUDE" `
  -I"$PYBIND11_INC" `
  -Ishared_c `
  shared_c/pybind.cpp `
  -L"$PYTHON_LIBDIR" `
  -l$PYTHON_LIB_BASENAME `
  -o build/shared_c_ext.pyd
```

### C) Use the Python module or script
```powershell
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"
$env:PYTHONPATH = "$(Resolve-Path build);$env:PYTHONPATH"
python -c "import shared_c_ext; print('pybind module loaded'); print(shared_c_ext.__doc__)"

# Placeholder script hook (edit src_python\main.py as needed)
python src_python/main.py
```

### D) Run the notebook
```powershell
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"
jupyter notebook src_notebook/main.ipynb
# or
jupyter lab src_notebook/main.ipynb
```

### E) Combined build (native exe + pybind module, MSVC)
```powershell
cd "C:\Users\Rajesh Khan\Desktop\C++ Model"
cl /std:c++17 /EHsc src_c\main.cpp /Fe:build\shared_c.exe

$PYTHON_INCLUDE = python -c "import sysconfig; print(sysconfig.get_paths()['include'])"
$PYTHON_LIBDIR  = python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
$PYTHON_LIB     = python -c "import sysconfig; print(sysconfig.get_config_var('LIBRARY'))"
$PYBIND11_INC   = python -c "import pybind11; print(pybind11.get_include())"

cl /std:c++17 /EHsc /LD shared_c\pybind.cpp `
  /I"$PYTHON_INCLUDE" /I"$PYBIND11_INC" /Ishared_c `
  /link /OUT:build\shared_c_ext.pyd /IMPLIB:build\shared_c_ext.lib `
        /LIBPATH:"$PYTHON_LIBDIR" "$PYTHON_LIB"
```

## Troubleshooting Quick Checks
- `cl` or `g++` not found: open the correct developer prompt or add MinGW to `PATH`.
- Python headers/libs missing: reinstall Python with “Add to PATH” and “Download debugging symbols” options; ensure `python -c "import sysconfig; ..."` returns valid paths.
- `pybind11` missing: `pip install --upgrade pybind11`.
- Import fails: ensure `build/` is on `PYTHONPATH` before importing `shared_c_ext`.
- Rebuild cleanly: delete old artifacts in `build/` and rerun the compile commands.

## Roadmap
- Add more C++ models and supporting C math kernels.
- Expand Python bindings to cover new models and parameters.
- Create benchmarking notebooks against common Python libraries to validate parity and performance.
