# DIVY site

The public-facing research and engineering site for [DIVY](https://github.com/nihas14m-star/divy), a provider-neutral AI orchestration and evaluation system.

This project is intentionally separate from the engineering repository. The engineering repository remains the source of truth for implementation details, test results, benchmark outputs, security boundaries, and reproducibility commands.

## Local development

```bash
npm install
npm run dev
```

Vite prints the local URL after startup.

## Production build

```bash
npm run build
npm run preview
```

The production build is emitted to `dist/` and is ignored by Git.

## QA completed locally

- `npm run build` succeeds.
- The page was opened in a local Chromium session and visually checked across the hero, architecture, routing, evidence, evaluation, results, security, future-work, and footer sections.
- Architecture tabs, raw/derived evidence switching, in-page navigation, and the reproduction-command copy control were exercised.
- The DIVY engineering repository test suite passes with `179 passed`.

## Content boundaries

The site deliberately does not claim that dynamic routing is superior. Its conclusion matches the current validated evidence:

> Dynamic routing has not demonstrated a measurable advantage over simpler strategies in the current experiments.

Security messaging also remains explicit: local `test.run` uses host-process execution for deterministic fixtures, while hosted adapters fail closed before network access until real process isolation is available.
