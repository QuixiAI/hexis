from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv


def _print_err(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _find_compose_file(start: Path | None = None) -> Path | None:
    """
    Find docker-compose.yml (preferred) or ops/docker-compose.yml by walking up from CWD.
    """
    cur = (start or Path.cwd()).resolve()
    for parent in (cur,) + tuple(cur.parents):
        legacy_compose = parent / "docker-compose.yml"
        if legacy_compose.exists():
            return legacy_compose
        ops_compose = parent / "ops" / "docker-compose.yml"
        if ops_compose.exists():
            return ops_compose
    return None


def _stack_root_from_compose(compose_file: Path) -> Path:
    if compose_file.parent.name == "ops":
        return compose_file.parent.parent
    return compose_file.parent


def ensure_docker() -> str:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        _print_err("Docker is not installed or not on PATH. Install Docker Desktop: https://docs.docker.com/get-docker/")
        raise SystemExit(1)
    try:
        subprocess.run([docker_bin, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        _print_err("Docker is installed but not running. Start Docker Desktop and retry.")
        raise SystemExit(1)
    return docker_bin


def ensure_compose(docker_bin: str) -> list[str]:
    try:
        subprocess.run([docker_bin, "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return [docker_bin, "compose"]
    except Exception:
        pass
    compose_bin = shutil.which("docker-compose")
    if compose_bin:
        return [compose_bin]
    _print_err("Docker Compose not available. Install Compose: https://docs.docker.com/compose/install/")
    raise SystemExit(1)


def resolve_env_file(stack_root: Path) -> Path | None:
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / ".env.local",
        stack_root / ".env",
        stack_root / ".env.local",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_compose(
    compose_cmd: list[str],
    compose_file: Path,
    stack_root: Path,
    args: list[str],
    env_file: Path | None,
) -> int:
    cmd = compose_cmd + ["-f", str(compose_file)]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    cmd += args

    try:
        result = subprocess.run(cmd, cwd=stack_root, env=os.environ.copy())
        return result.returncode
    except FileNotFoundError:
        _print_err("Failed to run docker compose. Ensure Docker is installed.")
        return 1


def _run_compose_capture(
    compose_cmd: list[str], compose_file: Path, stack_root: Path, args: list[str], env_file: Path | None
) -> tuple[int, str]:
    cmd = compose_cmd + ["-f", str(compose_file)]
    if env_file:
        cmd += ["--env-file", str(env_file)]
    cmd += args
    try:
        p = subprocess.run(cmd, cwd=stack_root, env=os.environ.copy(), capture_output=True, text=True)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip()
    except FileNotFoundError:
        return 1, "Failed to run docker compose. Ensure Docker is installed."


def _env_dsn() -> str:
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "43815")
    db_name = os.getenv("POSTGRES_DB", "hexis_memory")
    db_user = os.getenv("POSTGRES_USER", "hexis_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "hexis_password")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


async def _connect_with_retry(dsn: str, wait_seconds: int) -> asyncpg.Connection:
    deadline = time.monotonic() + wait_seconds
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return await asyncpg.connect(dsn, ssl=False, command_timeout=60.0)
        except Exception as e:  # pragma: no cover (timing-dependent)
            last_err = e
            await asyncio.sleep(1)
    raise TimeoutError(f"Failed to connect to Postgres after {wait_seconds}s: {last_err!r}")


def _redact_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))  # deep copy via json
    contact = out.get("user.contact")
    if isinstance(contact, dict):
        destinations = contact.get("destinations")
        if isinstance(destinations, dict):
            contact["destinations"] = {k: "***" for k in destinations.keys()}
    return out


def _coerce_json_value(val: Any) -> Any:
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return val
        try:
            return json.loads(s)
        except Exception:
            return val
    return val


async def _status_payload(
    dsn: str, *, wait_seconds: int, include_embedding_health: bool = True
) -> dict[str, Any]:
    conn = await _connect_with_retry(dsn, wait_seconds)
    try:
        payload: dict[str, Any] = {"dsn": dsn}
        payload["db_time"] = str(await conn.fetchval("SELECT now()"))

        payload["agent_configured"] = bool(await conn.fetchval("SELECT is_agent_configured()"))
        payload["heartbeat_paused"] = bool(await conn.fetchval("SELECT is_paused FROM heartbeat_state WHERE id = 1"))
        payload["should_run_heartbeat"] = bool(await conn.fetchval("SELECT should_run_heartbeat()"))
        try:
            payload["maintenance_paused"] = bool(await conn.fetchval("SELECT is_paused FROM maintenance_state WHERE id = 1"))
            payload["should_run_maintenance"] = bool(await conn.fetchval("SELECT should_run_maintenance()"))
        except Exception:
            payload["maintenance_paused"] = None
            payload["should_run_maintenance"] = None

        payload["pending_external_calls"] = int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM external_calls WHERE status = 'pending'::external_call_status"
            )
        )
        payload["pending_outbox_messages"] = int(
            await conn.fetchval("SELECT COUNT(*) FROM outbox_messages WHERE status = 'pending'")
        )

        payload["embedding_service_url"] = await conn.fetchval(
            "SELECT value FROM embedding_config WHERE key = 'service_url'"
        )
        payload["embedding_dimension"] = int(await conn.fetchval("SELECT embedding_dimension()"))

        if include_embedding_health:
            try:
                payload["embedding_service_healthy"] = bool(
                    await conn.fetchval("SELECT check_embedding_service_health()")
                )
            except Exception as e:
                payload["embedding_service_healthy"] = False
                payload["embedding_service_error"] = repr(e)

        return payload
    finally:
        await conn.close()


async def _config_rows(dsn: str, *, wait_seconds: int) -> dict[str, Any]:
    conn = await _connect_with_retry(dsn, wait_seconds)
    try:
        rows = await conn.fetch("SELECT key, value FROM config ORDER BY key")
        out: dict[str, Any] = {}
        for r in rows:
            out[str(r["key"])] = _coerce_json_value(r["value"])
        return out
    finally:
        await conn.close()


async def _config_validate(dsn: str, *, wait_seconds: int) -> tuple[list[str], list[str]]:
    """
    Returns (errors, warnings).
    """
    conn = await _connect_with_retry(dsn, wait_seconds)
    try:
        errors: list[str] = []
        warnings: list[str] = []

        rows = await conn.fetch("SELECT key, value FROM config ORDER BY key")
        cfg: dict[str, Any] = {str(r["key"]): _coerce_json_value(r["value"]) for r in rows}
        required_keys = [
            "agent.is_configured",
            "agent.objectives",
            "llm.heartbeat",
            "llm.chat",
        ]
        for k in required_keys:
            if k not in cfg:
                errors.append(f"Missing config key: {k}")

        is_conf = cfg.get("agent.is_configured")
        if is_conf is not True:
            # Some drivers return jsonb scalars as strings.
            if is_conf == "true":
                is_conf = True
        if is_conf is not True:
            errors.append("agent.is_configured is not true (run `hexis init`).")

        objectives = cfg.get("agent.objectives")
        if not isinstance(objectives, list) or not objectives:
            errors.append("agent.objectives must be a non-empty array (run `hexis init`).")

        def _validate_llm(name: str) -> None:
            val = cfg.get(name)
            if not isinstance(val, dict):
                errors.append(f"{name} must be an object (run `hexis init`).")
                return
            provider = str(val.get("provider") or "").strip().lower()
            model = str(val.get("model") or "").strip()
            endpoint = str(val.get("endpoint") or "").strip()
            api_key_env = str(val.get("api_key_env") or "").strip()

            if not provider:
                errors.append(f"{name}.provider is required")
            if not model and provider not in {"ollama"}:
                warnings.append(f"{name}.model is empty (will rely on worker defaults)")

            # Keys are provided via environment variables; DB stores the env var name.
            if provider in {"openai", "anthropic", "openai_compatible"}:
                if api_key_env:
                    if os.getenv(api_key_env) is None:
                        errors.append(f"{name}.api_key_env={api_key_env} is not set in environment")
                else:
                    # Local endpoints often don't require a key; warn rather than fail.
                    if not endpoint or ("localhost" not in endpoint and "127.0.0.1" not in endpoint):
                        warnings.append(f"{name}.api_key_env not set (LLM calls may fail)")

        _validate_llm("llm.heartbeat")
        _validate_llm("llm.chat")

        # Basic heartbeat config sanity.
        interval = await conn.fetchval(
            "SELECT value FROM heartbeat_config WHERE key = 'heartbeat_interval_minutes'"
        )
        if interval is None or float(interval) <= 0:
            errors.append("heartbeat_config.heartbeat_interval_minutes must be > 0")

        return errors, warnings
    finally:
        await conn.close()


async def _demo(dsn: str, *, wait_seconds: int) -> dict[str, Any]:
    from core.cognitive_memory_api import CognitiveMemory, MemoryType

    # Wait for DB.
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        # Wait for embeddings so the demo doesn't flake during container startup.
        deadline = time.monotonic() + wait_seconds
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                ok = await conn.fetchval("SELECT check_embedding_service_health()")
                if ok is True:
                    break
            except Exception as e:  # pragma: no cover (timing-dependent)
                last = e
            await asyncio.sleep(1)
        else:
            raise TimeoutError(f"Embedding service not healthy after {wait_seconds}s: {last!r}")
    finally:
        await conn.close()

    async with CognitiveMemory.connect(dsn) as mem:
        # Minimal end-to-end: remember -> recall -> hydrate -> working memory.
        m1 = await mem.remember("Demo: the user prefers short, direct answers", type=MemoryType.SEMANTIC, importance=0.7)
        m2 = await mem.remember(
            "Demo: the user is working on the Hexis memory system",
            type=MemoryType.EPISODIC,
            importance=0.6,
        )
        held = await mem.hold("Demo: temporary context in working memory", ttl_seconds=600)

        recall = await mem.recall("What do I know about the user's preferences?", limit=5)
        hydrate = await mem.hydrate("Summarize what we know about the user", include_goals=False)
        working_hits = await mem.search_working("temporary context", limit=5)

        return {
            "remembered_ids": [str(m1), str(m2)],
            "working_memory_id": str(held),
            "recall_count": len(recall.memories),
            "hydrate_memory_count": len(hydrate.memories),
            "working_search_count": len(working_hits),
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hexis", description="Manage Hexis Memory Docker stack")
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="Start the stack")
    up.add_argument("--build", action="store_true", help="Build images before starting")
    up.set_defaults(func="up")

    down = sub.add_parser("down", help="Stop the stack")
    down.set_defaults(func="down")

    logs = sub.add_parser("logs", help="Show logs")
    logs.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    logs.set_defaults(func="logs")

    ps = sub.add_parser("ps", help="List services")
    ps.set_defaults(func="ps")

    chat = sub.add_parser("chat", help="Run the conversation loop (forwards args to core.conversation)")
    chat.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to core.conversation")
    chat.set_defaults(func="chat")

    ingest = sub.add_parser("ingest", help="Run the ingestion pipeline (forwards args to core.ingest)")
    ingest.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to core.ingest")
    ingest.set_defaults(func="ingest")

    worker = sub.add_parser("worker", help="Run background workers (forwards args to apps.workers.worker)")
    worker.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to apps.workers.worker")
    worker.set_defaults(func="worker")

    init = sub.add_parser("init", help="Interactive Hexis setup wizard (stores config in Postgres)")
    init.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to apps.cli.hexis_init")
    init.set_defaults(func="init")

    mcp = sub.add_parser("mcp", help="Run MCP server exposing CognitiveMemory tools (stdio)")
    mcp.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to apps.mcp.hexis_mcp_server")
    mcp.set_defaults(func="mcp")

    start = sub.add_parser("start", help="Start workers (active profile)")
    start.set_defaults(func="start")

    stop = sub.add_parser("stop", help="Stop workers (containers remain)")
    stop.set_defaults(func="stop")

    status = sub.add_parser("status", help="Show system status (db/config/queue)")
    status.add_argument("--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars")
    status.add_argument("--wait-seconds", type=int, default=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    status.add_argument("--json", action="store_true", help="Output JSON")
    status.add_argument("--no-docker", action="store_true", help="Skip docker compose checks")
    status.set_defaults(func="status")

    config = sub.add_parser("config", help="Show/validate agent configuration stored in Postgres")
    cfg_sub = config.add_subparsers(dest="config_command", required=True)

    cfg_show = cfg_sub.add_parser("show", help="Print config table")
    cfg_show.add_argument("--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars")
    cfg_show.add_argument("--wait-seconds", type=int, default=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    cfg_show.add_argument("--json", action="store_true", help="Output JSON")
    cfg_show.add_argument("--no-redact", action="store_true", help="Do not redact contact destinations")
    cfg_show.set_defaults(func="config_show")

    cfg_validate = cfg_sub.add_parser("validate", help="Validate required config keys and environment references")
    cfg_validate.add_argument("--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars")
    cfg_validate.add_argument("--wait-seconds", type=int, default=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    cfg_validate.set_defaults(func="config_validate")

    demo = sub.add_parser("demo", help="Run a quick end-to-end sanity check against the DB")
    demo.add_argument("--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars")
    demo.add_argument("--wait-seconds", type=int, default=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    demo.add_argument("--json", action="store_true", help="Output JSON")
    demo.set_defaults(func="demo")

    return p


def _run_module(module: str, argv: list[str]) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    cmd = [sys.executable, "-m", module, *argv]
    try:
        result = subprocess.run(cmd, env=os.environ.copy())
        return result.returncode
    except FileNotFoundError:
        _print_err(f"Failed to run {cmd[0]!r}")
        return 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    compose_file = _find_compose_file()
    stack_root = _stack_root_from_compose(compose_file) if compose_file else Path.cwd()
    env_file = resolve_env_file(stack_root)

    docker_cmds = {"up", "down", "ps", "logs", "start", "stop"}
    docker_bin: str | None = None
    compose_cmd: list[str] | None = None
    if args.func in docker_cmds:
        if compose_file is None:
            _print_err("docker-compose.yml not found.")
            return 1
        docker_bin = ensure_docker()
        compose_cmd = ensure_compose(docker_bin)

    if args.func == "up":
        up_args = ["up", "-d"]
        if args.build:
            up_args.append("--build")
        return run_compose(compose_cmd or [], compose_file, stack_root, up_args, env_file)
    if args.func == "down":
        return run_compose(compose_cmd or [], compose_file, stack_root, ["down"], env_file)
    if args.func == "ps":
        return run_compose(compose_cmd or [], compose_file, stack_root, ["ps"], env_file)
    if args.func == "logs":
        log_args = ["logs"] + (["-f"] if args.follow else [])
        return run_compose(compose_cmd or [], compose_file, stack_root, log_args, env_file)
    if args.func == "chat":
        return _run_module("core.conversation", args.args)
    if args.func == "ingest":
        return _run_module("core.ingest", args.args)
    if args.func == "worker":
        return _run_module("apps.workers.worker", args.args)
    if args.func == "init":
        return _run_module("apps.cli.hexis_init", args.args)
    if args.func == "mcp":
        return _run_module("apps.mcp.hexis_mcp_server", args.args)
    if args.func == "start":
        return run_compose(
            compose_cmd or [],
            compose_file,
            stack_root,
            ["--profile", "active", "up", "-d", "heartbeat_worker", "maintenance_worker"],
            env_file,
        )
    if args.func == "stop":
        return run_compose(
            compose_cmd or [],
            compose_file,
            stack_root,
            ["stop", "heartbeat_worker", "maintenance_worker"],
            env_file,
        )
    if args.func == "status":
        dsn = args.dsn or _env_dsn()
        payload = asyncio.run(_status_payload(dsn, wait_seconds=args.wait_seconds))
        if not args.no_docker:
            try:
                docker_bin = ensure_docker()
                compose_cmd = ensure_compose(docker_bin)
                if compose_file is None:
                    raise SystemExit
                rc, out = _run_compose_capture(compose_cmd, compose_file, stack_root, ["ps"], env_file)
                payload["docker_ps_rc"] = rc
                payload["docker_ps"] = out
            except SystemExit:
                payload["docker_ps_rc"] = 1
                payload["docker_ps"] = "Docker not available"
        if args.json:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            lines = [
                f"DB time: {payload.get('db_time')}",
                f"Agent configured: {payload.get('agent_configured')}",
                f"Heartbeat paused: {payload.get('heartbeat_paused')}",
                f"Should run heartbeat: {payload.get('should_run_heartbeat')}",
                f"Maintenance paused: {payload.get('maintenance_paused')}",
                f"Should run maintenance: {payload.get('should_run_maintenance')}",
                f"Embedding URL: {payload.get('embedding_service_url')}",
                f"Embedding healthy: {payload.get('embedding_service_healthy')}",
                f"Pending external_calls: {payload.get('pending_external_calls')}",
                f"Pending outbox_messages: {payload.get('pending_outbox_messages')}",
            ]
            sys.stdout.write("\n".join(lines) + "\n")
        return 0
    if args.func == "config_show":
        dsn = args.dsn or _env_dsn()
        cfg = asyncio.run(_config_rows(dsn, wait_seconds=args.wait_seconds))
        if not args.no_redact:
            cfg = _redact_config(cfg)
        sys.stdout.write(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        return 0
    if args.func == "config_validate":
        dsn = args.dsn or _env_dsn()
        errors, warnings = asyncio.run(_config_validate(dsn, wait_seconds=args.wait_seconds))
        for w in warnings:
            _print_err(f"warning: {w}")
        if errors:
            for e in errors:
                _print_err(f"error: {e}")
            return 1
        sys.stdout.write("ok\n")
        return 0
    if args.func == "demo":
        dsn = args.dsn or _env_dsn()
        result = asyncio.run(_demo(dsn, wait_seconds=args.wait_seconds))
        if args.json:
            sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(
                "Demo ok\n"
                f"- remembered_ids: {', '.join(result['remembered_ids'])}\n"
                f"- recall_count: {result['recall_count']}\n"
                f"- hydrate_memory_count: {result['hydrate_memory_count']}\n"
                f"- working_search_count: {result['working_search_count']}\n"
            )
        return 0

    _print_err("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
