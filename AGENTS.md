# AGENTS.md — Guidance for AI coding agents

Purpose
-------
This file helps AI coding agents quickly understand the repository, run the code locally, and make safe, minimal changes. It is intentionally short and links to detailed docs instead of copying them.

Quick start (commands)
----------------------
Install dependencies and run the data discovery sanity check:

```
pip install -r requirements.txt
python src/data_loader.py
```

Runtime notes
-------------
- The single authoritative entry for sources is `src/data_loader.py` — all other modules expect pandas DataFrames returned by this class.
- To point the loader to a custom CSV folder set the env var `AVAILABILITY_DATA_DIR` (PowerShell example):

```
$env:AVAILABILITY_DATA_DIR = "D:\path\to\csvs"
```

Project structure & phases
--------------------------
- Phase 1: data discovery, `src/data_loader.py`, docs. (Implemented)
- Phase 2: analytics / recommendation logic (stubs in `src/` — `*_analysis.py`, `recommendation_engine.py`).
- Phase 3: dashboard (placeholder at `dashboard/app.py`).

Key files to inspect
--------------------
- `src/data_loader.py` — source-aware loader and the primary runtime API.
- `requirements.txt` — minimal runtime dependencies (pandas, numpy).
- `README.md`, `docs/DATA_DICTIONARY.md`, `docs/RELATIONSHIPS.md` — canonical documentation and data contracts.

Agent guidelines (concise)
-------------------------
- Link, don't embed: reference docs in `docs/` or `README.md` instead of copying.
- Avoid implementing Phase 2/3 features unless the ticket explicitly requests them. Phase boundaries are intentional.
- When running code, prefer `python src/data_loader.py` as the Phase-1 smoke test.
- Do not modify raw CSVs; write outputs to `outputs/` for artifacts.
- Preserve existing APIs: code outside `data_loader.py` assumes DataFrame schemas; change schemas only after updating `README.md` and `docs/`.

Focused area: "ter"
--------------------
User argument: `ter`. Interpreted as terminal/run tasks: this file highlights runtime commands, env vars, and the minimal steps an agent should take when executing or validating the project locally.

Next suggested customizations
---------------------------
- Add a small execution prompt (`prompts/run-data-loader.md`) that runs the loader, captures `loader.profile()`, and returns a short JSON summary for automated checks.
- Create a tiny test harness `tests/test_data_loader.py` that asserts discovered tables are present (optional Phase-1 improvement).

Links
-----
- [README.md](README.md)
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)
- [docs/RELATIONSHIPS.md](docs/RELATIONSHIPS.md)
