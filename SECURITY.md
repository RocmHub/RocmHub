# Security Policy

## Reporting Security Issues

We take the security of ROCmHub seriously. If you discover a vulnerability or security boundary issue, please report it responsibly.

### Recommended Reporting Channel

Please report security vulnerabilities privately via **GitHub Security Advisories**:
- Navigate to the **Security** tab of the repository.
- Click **Report a vulnerability** to open a confidential report.

Please **do not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

---

## What to Include in Your Report

To help us triage and resolve the issue quickly, please provide:
1. **Description**: A clear description of the vulnerability, including affected versions and components.
2. **Steps to Reproduce**: A minimal, reproducible proof-of-concept (PoC) script, command sequence, or HTTP payload.
3. **Potential Impact**: An explanation of what an attacker could achieve (e.g. arbitrary file overwrite, secret leakage, denial of service).
4. **Environment**: Host OS, architecture, Python version, ROCm version (if applicable).

### Important: Do Not Include Live Secrets
- Never include active Hugging Face tokens, private keys, passwords, or personal credentials in your report or PoC.
- Replace any real tokens with placeholders such as `hf_***REDACTED***`.

---

## Project Security Maturity

ROCmHub is currently at version `0.1.0` (MVP / Pre-Alpha). The software is designed primarily for local development workstations and dedicated lab environments. It features built-in security controls including:
- Input path traversal validation (`validate_job_path`) preventing directory escape;
- Fail-closed secret scanner (`scan_for_secrets`) scrubbing tokens and credentials from generated artifacts;
- Strict regex validation on model and job identifiers;
- Global exception sanitization preventing traceback leakage.

Authentication and multi-tenant isolation are not included in the 0.1.0 release.
