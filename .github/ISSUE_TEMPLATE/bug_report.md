---
name: Bug report
about: Create a report to help us improve ROCmHub
title: '[BUG] '
labels: bug
assignees: ''
---

## Description
A clear and concise description of what the bug is.

## Environment & Hardware Diagnostic
Please run `rocmhub doctor` or `rocmhub env --json` and paste the output:

- **OS / Distro**: (e.g. Ubuntu 22.04 LTS, macOS Darwin 24.x)
- **Architecture**: (e.g. x86_64, arm64)
- **Python Version**: (e.g. 3.10.12)
- **ROCm Version**: (e.g. ROCm 6.2.0, or "Not installed")
- **PyTorch Version**: (e.g. 2.4.0+rocm6.0, 2.4.0, or None)
- **GPU Model**: (e.g. Radeon RX 7900 XTX, Radeon Pro W7900, Instinct MI300X, or "None / CPU")
- **Target Architecture (gfx)**: (e.g. gfx1100, gfx90a, gfx942)
- **KFD Driver Access**: (e.g. /dev/kfd rw, or N/A)

## Command Executed
Exact command that produced the error:
```bash
rocmhub <command> [args]
```

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
What actually happened (including exit code, error messages, and tracebacks):
```text
<paste output / error here>
```

## Relevant Logs or Artifacts
Attach or paste `manifest.json`, `build_manifest.json`, or terminal logs if applicable.
