# SGLang (MiniCPM-Sala fork)

## Repo structure

Three sub-projects with different build systems:

| Path | What | Build |
|------|------|-------|
| `python/sglang/` | Main Python package (FastAPI server, 140+ model definitions, SRT runtime) | `setuptools` + `setuptools-scm` |
| `sgl-kernel/` | Pre-compiled CUDA kernels (C++17, CUDA >= 11.8) | `scikit-build-core` + CMake |
| `sgl-model-gateway/` | Rust API gateway (axum, gRPC, WASM) | Cargo |

CLI entrypoint: `sglang` → `python/sglang/cli/main.py:main` (subcommands: `serve`, `generate`).

## Custom installer

`install_minicpm_sala.sh` — one-click setup using `uv`, Python 3.12, creates a local venv (`sglang_minicpm_sala_env/`), installs 3rdparty submodules with CUDA kernels.

## First-time setup

```bash
git submodule update --init --recursive
```

Two 3rdparty submodules at `3rdparty/`: `sparse_kernel` and `infllmv2_cuda_impl` (branch `minicpm_sala`).

## Install commands

```bash
# Main package (editable, all deps)
pip install -e python[all]
uv pip install -e python[all]

# NPU only
pip install -e python[srt_npu]

# sgl-kernel
cd sgl-kernel && make install   # pip install -e . --no-build-isolation

# sgl-model-gateway Python bindings
cd sgl-model-gateway && make python-dev   # debug build (maturin develop)
```

## Lint & format

```bash
SKIP=no-commit-to-branch pre-commit run --all-files
```

Lint checks only: `ruff` (F401/F821), `isort` (profile=black), `black`, `codespell`, `clang-format` (C++/CUDA), `nbstripout`. **No typechecker (mypy/pyright) configured.**

Format per sub-project: `sgl-kernel` → `make format`, `sgl-model-gateway` → `make fmt` (nightly rustfmt).

gRPC generated files (`*_pb2*`) are excluded from lint.

## Test framework

Hybrid: `unittest` (primary for CI) + `pytest` (kernel/gateway tests).

```bash
# CI test orchestrator (dispatches registered tests)
python3 test/run_suite.py --hw cuda --suite stage-a-test-1

# sgl-kernel tests
cd sgl-kernel && make test

# sgl-model-gateway tests
cd sgl-model-gateway && make test       # cargo test
cd sgl-model-gateway && make python-test  # pytest e2e_test/

# JIT kernel tests
pytest python/sglang/jit_kernel/tests/
```

### CI test registration

Tests in `test/registered/` declare metadata inline:

```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, suite="stage-a-test-1")
```

Suites: `stage-a-*`, `stage-b-*`, `stage-c-*` (per-commit), `nightly-*` (nightly).

## Key env

- `SGLANG_IS_IN_CI=true` — set in CI pipelines
- `test/pytest.ini` sets `asyncio_mode = auto`

## Generated / ignored artifacts

- `python/sglang/_version.py` (setuptools-scm)
- `build/`, `dist/`, `*.egg-info/`
- `*.so`, `*.nsys-rep`, `*.ncu-rep`, `tl_out/`
- `sgl-model-gateway/Cargo.lock`
