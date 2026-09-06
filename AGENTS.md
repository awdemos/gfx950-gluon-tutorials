# Agent Notes: gfx950-gluon-tutorials

Hands-on tutorial repository for writing high-performance Gluon kernels on AMD MI350 / MI355 GPUs (gfx950). Each kernel version demonstrates one optimization concept; the READMEs are part of the artifact.

## Repository Layout

- `kernels/gemm/intra_wave/a16w16/` — FP16 GEMM tutorial, v0 → v9 (start here).
- `kernels/gemm/intra_wave/a8w8/` — BF8 GEMM with the same design adapted for 8-bit.
- `kernels/gemm/intra_wave/a4w4/` — MXFP4 GEMM with per-group microscaling.
- `kernels/attention/` — Flash Attention forward kernels (fmha_v3, fmha_v4).
- `docs/` — Performance philosophy, LDS throughput, memory-bandwidth, MFMA-efficiency notes.
- `scripts/` — Benchmarking, rocprof/ATT automation, counter collection, perf-table generation.
- `experiments/` — Standalone validations referenced by kernel READMEs.
- `layout_plot/` — LaTeX-based layout visualizations.

## Environment Setup

A ROCm/Gfx9-capable environment is required. Install Python dependencies and a recent Triton/Gluon toolchain:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # if present; otherwise install triton and dependencies manually
```

Style tools are configured in `pyproject.toml`:

```bash
pip install black ruff
black kernels scripts experiments
ruff check kernels scripts experiments
```

## Running a Kernel

Most kernels are invoked from the repository root as Python scripts that auto-tune and benchmark on the local GPU:

```bash
# Example: run the FP16 GEMM tutorial variant
python kernels/gemm/intra_wave/a16w16/v0_naive.py

# Or run the prepared benchmark wrapper
python scripts/benchmark_prepared.py --kernel kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop.py
```

## Profiling & Counter Collection

```bash
# Install the rocprof ATT decoder once
bash scripts/install_att_decoder.sh

# Collect hardware counters
python scripts/collect_counters.py --kernel <path> --output counters.csv

# Generate a perf summary table
python scripts/run_perf_table.py --input counters.csv --output perf.md
```

## Style & Lint

- `black` line length 100, target Python 3.10.
- `ruff` with `E`, `F`, `I`, `W` rules; `E501` and `E741` ignored.
- IR dumps (`ir_dump*`, `*.ttgir`, `*.llir`, `*.amdgcn`) and upstream-ported FA kernels (`fmha_v*.py`) are excluded from formatting/lint.
- Kernel READMEs are part of the tutorial; treat them with the same care as the code.

## Common Issues

- **No ROCm device found**: kernels require an AMD GPU; most scripts detect the device automatically and will fail gracefully on non-ROCm hosts.
- **Counter collection missing**: `rocprof` must be on `PATH`; install the ATT decoder for instruction-level traces.
- **Format CI failure**: run `bash scripts/format_fix.sh` before pushing.

## License

MIT License. See `LICENSE`.
