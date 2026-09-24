# UX acceptance

The acceptance journey uses the real FastAPI application, Vite frontend, SQLite job store, SSE updates, and browser rendering. Network requests are delayed only long enough to make transient loading states deterministic in screenshots.

## State coverage

| Product | Covered states | Screenshots |
| --- | --- | --- |
| Model Explorer | idle/empty, input, loading skeleton, failure + toast, retry, inspected result | `01`–`05` |
| Forge Studio | all six wizard steps, plan loading, review, running, artifact output, failure, retry recovery | `06`–`14` |
| AI Engineer | idle/input, running activity, domain failure, retry, executive recommendation | `15`–`19` |
| Optimization | idle comparison, running baseline/candidate lanes, domain failure, retry, completed comparison | `20`–`24` |
| Mobile | completed Optimization, Forge, Engineer, and Explorer result states | `25`–`28` |

Screenshots are stored in `screenshots/acceptance/`.

## Run locally

```bash
node scripts/ux_acceptance.js
```

Run the command from the repository root. The runner locates the project relative to its own script, uses the current Node executable, and locates Python from `ROCMHUB_PYTHON`, the active virtual environment, or the repository's `.venv`. Install `puppeteer-core` in `frontend/` and set `CHROME_PATH` (or `PUPPETEER_EXECUTABLE_PATH`) to a Chrome/Chromium executable.

The runner uses isolated ports `8780` and `5180` and a unique temporary directory for its database and workspace. It shuts down only those isolated processes when finished, leaving the normal development servers untouched.
