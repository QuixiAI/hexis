# Development Guidelines

This file provides guidance for contributors working on the AGI Memory System.

## Project Structure & Module Organization

- `db/schema.sql`: single source of truth for the database schema (extensions, tables, functions, triggers, views). Schema is applied on fresh DB init.
- `docker-compose.yml`, `Dockerfile`, `ops/Dockerfile.worker`: local stack (Postgres + embeddings + optional workers).
- `apps/workers/worker.py`: stateless workers: heartbeat (conscious) + maintenance (subconscious).
- `core/cognitive_memory_api.py`: thin Python client for the “DB is the brain” API surface.
- `apps/cli/hexis_cli.py`: CLI entrypoint (`hexis …`) for local workflows; `apps/cli/hexis_init.py` bootstraps DB config; `apps/mcp/hexis_mcp_server.py` exposes MCP tools.
- `tests/db/test_db.py`: DB integration test suite (pytest + asyncpg) covering schema, workers, and database functions.
- `tests/core/test_core_api.py`: core API integration tests (core/cognitive_memory_api.py).
- `tests/cli/test_cli.py`: CLI smoke tests.
- Docs: `README.md` (user-facing), `docs/architecture.md` (design/architecture consolidation).

## Build, Test, and Development Commands

- Start services (passive): `docker compose up -d` (db + embeddings).
- Start services (active): `docker compose --profile active up -d` (adds `heartbeat_worker` + `maintenance_worker`).
- Reset DB volume (schema changes): `docker compose down -v && docker compose up -d`.
- Configure agent (gates heartbeats): `./hexis init` (or `hexis init` if installed).
- Run tests (all): `pytest tests -q` (expects Docker services up).
- Run DB tests: `pytest tests/db -q`
- Run core tests: `pytest tests/core -q`
- Run CLI tests: `pytest tests/cli -q`

## Coding Style & Naming Conventions

- Python: follow Black formatting conventions; prefer type hints and explicit names over abbreviations.
- Keep the DB as the authority: add/modify SQL functions in `db/schema.sql` rather than duplicating logic in Python.
- Prefer additive, backwards-compatible schema changes; avoid renames unless necessary.

## Testing Guidelines

- Framework: `pytest` + `pytest-asyncio` (session loop scope).
- Tests are integration-style; use transactions and rollbacks where practical to avoid cross-test coupling.
- Naming: `test_*` functions; use `get_test_identifier()` patterns in `tests/utils.py` for unique data.

## Commit & Pull Request Guidelines

- Commits: short, imperative summaries (e.g., “Add MCP server tools”, “Gate heartbeat on config”).
- PRs: include rationale, how to run/verify, and any DB reset requirements; call out changes to `db/schema.sql`, `docker-compose.yml`, and `README.md`.

## Configuration & Safety Notes

- Secrets: store API keys in environment variables (`.env`), not in Postgres; DB config stores env var *names* only.
- Heartbeat is gated until `agent.is_configured=true` (set via `hexis init`).

## Skills

- Use the `reflex` skill for any work involving Reflex UI code (`ui/*.py`), rx components, or state/event/rendering issues.
