# Contributing to ROCmHub

Thank you for your interest in contributing to ROCmHub!

ROCmHub is built with an uncompromising commitment to engineering discipline, reproducibility, and **hardware truthfulness**. Before submitting code, please review the guidelines below.

---

## The Prime Rule: Hardware Truthfulness

ROCmHub enforces a strict distinction between software readiness and physical hardware execution:

1. **Mock Test $\ne$ Hardware Validation**: Passing unit tests with mocked GPU detection or simulated outputs confirms code execution, **not** physical AMD hardware compatibility.
2. **Static gfx List $\ne$ Hardware Support Claim**: Mentioning an AMD architecture (e.g. `gfx1100` or `gfx90a`) in planning or static catalogs is advisory, not proof of hardware certification.
3. **Zero Synthetic Metrics**: Never generate synthetic, estimated, or interpolated latency, throughput, or memory metrics. When a GPU measurement cannot be taken, the status must strictly be `NOT_MEASURED` with values set to `None`.
4. **Strict `EXECUTED` Gate**: The status `EXECUTED` and the attribute `amd_validated=True` are granted **only** when 5 independent facts are confirmed: process success, inference execution confirmed via JSON payload, PyTorch HIP runtime used, physical AMD GPU hardware used, and output validation passed without corruption.

---

## Local Development Setup

### 1. Python Environment

ROCmHub requires Python 3.9+.

```bash
# Clone the repository
git clone <repository-url>
cd ROCmHub

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development and test dependencies
pip install -e ".[dev,test]"
```

### 2. Frontend Environment

ROCmHub's web application requires Node.js 18+ and npm.

```bash
cd frontend
npm install
cd ..
```

---

## Verification & Quality Gates

Every pull request must pass the full local quality suite before submission:

### 1. Python Linting & Formatting
```bash
ruff check src tests scripts
```

### 2. Static Type Checking
```bash
mypy src tests/test_api.py tests/test_doctor.py tests/test_execution_safety.py
```

### 3. Automated Test Suite
```bash
# Full unit & safety invariant suite (368+ tests)
pytest
```

### 4. Package Build & Smoke Install
```bash
# Build wheel and sdist
python -m pip install build
python -m build

# Run automated fresh environment installation smoke test
python scripts/fresh_install_smoke.py
```

### 5. Frontend Tests & Production Build
```bash
cd frontend
npm test -- --run
npm run build
cd ..
```

### 6. Real Browser E2E Acceptance (Optional / Pre-Release)
```bash
# Requires Google Chrome installed
node scripts/browser_e2e.js
```

---

## Commit Guidelines

We use conventional commit messages:

- `feat(...)`: New capability or command
- `fix(...)`: Bug fix or security hardening
- `docs(...)`: Documentation updates
- `test(...)`: Adding or updating test suites
- `refactor(...)`: Code refactoring without behavioral change
- `release(...)`: Version release packaging

Keep commit messages concise and descriptive.
