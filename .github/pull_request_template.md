## Summary of Changes
Provide a brief summary of the proposed changes, context, and motivation.

## Verification Checklist

Please verify each item before submitting:

- [ ] **Tests**: New and existing automated tests pass locally (`pytest`, `ruff check`, `mypy src tests/test_api.py`).
- [ ] **Frontend**: Frontend tests pass and production build succeeds (`npm test`, `npm run build` in `frontend/`).
- [ ] **Hardware Truthfulness**: **NO synthetic or fake GPU metrics** are introduced. Diagnostic runs cleanly output `NOT_MEASURED` or `CONFIG_ONLY`.
- [ ] **Hardware Claims**: **NO unverified hardware support claims** are made. Mock tests are NOT presented as hardware validation.
- [ ] **Documentation**: Corresponding documentation and schemas have been updated where applicable.
- [ ] **Backwards Compatibility**: Preserves API schemas, manifest formats, and semantic exit codes.
