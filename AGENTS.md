# AGENTS.md

**Canonical developer guidance lives in [`CLAUDE.md`](./CLAUDE.md). Read it first.** This file is a pointer so Codex and any `AGENTS.md`-seeking agent share the same map as Claude Code; `README.md` remains the user/product reference.

One-screen orientation (full detail in `CLAUDE.md`):

- **What this is** — `runtheta` (package `theta`), a GPU thermal-power forensics agent built on one signal: `R_θ = (T_junction − T_ref) / P_GPU`.
- **Commands** — `pip install -e ".[dev]"`, `pytest tests/ -v --tb=short`, `ruff check theta/ --select E,F,W --ignore E501`, `python -m build`, CLI entrypoint `theta`.
- **No GPU in CI or on a laptop** — tests run on simulated/recorded telemetry; never assume a live NVML device.
- **Architecture** — one-tick pipeline (`collector → window → R_θ → classifier → detector/peer → governor → alerter/exporter`) wired by `theta/agent/daemon.py`; vendor collectors plug in behind `theta/agent/hal.py`.
- **Invariants** — classify only on steady-state windows; non-T4 hardware must be calibrated before `monitor` runs; temporal vs. peer detection are kept deliberately separate; `schedulable ≠ healthy`; optional deps stay inert when absent; tests pin real incidents (E009).

Keep this file thin. If guidance changes, update `CLAUDE.md` only.
