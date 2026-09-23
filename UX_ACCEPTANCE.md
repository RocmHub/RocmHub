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
/Users/netcars/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node scripts/ux_acceptance.js
```

The runner uses isolated ports `8780` and `5180` and removes its temporary database before every run. It shuts down only those isolated processes when finished, leaving the normal development servers untouched.
