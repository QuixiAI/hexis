from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from getpass import getpass
from typing import Any

import asyncpg
from dotenv import load_dotenv


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


def _print_err(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _prompt(
    label: str,
    *,
    default: str | None = None,
    required: bool = False,
    secret: bool = False,
) -> str:
    while True:
        suffix = f" [{default}]" if default is not None and default != "" else ""
        prompt = f"{label}{suffix}: "
        raw = getpass(prompt) if secret else input(prompt)
        value = raw.strip()
        if not value and default is not None:
            value = str(default)
        if required and not value:
            _print_err("Value required.")
            continue
        return value


def _prompt_int(label: str, *, default: int, min_value: int | None = None) -> int:
    while True:
        raw = _prompt(label, default=str(default), required=True)
        try:
            value = int(raw)
        except ValueError:
            _print_err("Enter an integer.")
            continue
        if min_value is not None and value < min_value:
            _print_err(f"Must be >= {min_value}.")
            continue
        return value


def _prompt_float(label: str, *, default: float, min_value: float | None = None) -> float:
    while True:
        raw = _prompt(label, default=str(default), required=True)
        try:
            value = float(raw)
        except ValueError:
            _print_err("Enter a number.")
            continue
        if min_value is not None and value < min_value:
            _print_err(f"Must be >= {min_value}.")
            continue
        return value


def _prompt_yes_no(label: str, *, default: bool) -> bool:
    default_str = "y" if default else "n"
    while True:
        raw = _prompt(label, default=default_str).lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        _print_err("Enter y/n.")


def _prompt_list(label: str, *, required: bool = False) -> list[str]:
    print(f"{label} (one per line; blank to finish):")
    items: list[str] = []
    while True:
        raw = input("> ").strip()
        if not raw:
            if required and not items:
                _print_err("At least one item required.")
                continue
            return items
        items.append(raw)


def _env_db_config() -> DbConfig:
    return DbConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "agi_db"),
        user=os.getenv("POSTGRES_USER", "agi_user"),
        password=os.getenv("POSTGRES_PASSWORD", "agi_password"),
    )


async def _connect_with_retry(dsn: str, *, wait_seconds: int) -> asyncpg.Connection:
    deadline = time.monotonic() + wait_seconds
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = await asyncpg.connect(dsn, ssl=False, command_timeout=60.0)
            return conn
        except Exception as e:
            last_err = e
            await asyncio.sleep(1)
    raise TimeoutError(f"Failed to connect to Postgres after {wait_seconds}s: {last_err!r}")


async def _ensure_schema_has_config(conn: asyncpg.Connection) -> None:
    ok = await conn.fetchval("SELECT to_regclass('public.config') IS NOT NULL")
    if not ok:
        raise RuntimeError(
            "Database schema is missing `config` table. "
            "If you just updated `schema.sql`, reset the DB volume and retry: `docker compose down -v && docker compose up -d`."
        )


async def _get_heartbeat_config(conn: asyncpg.Connection) -> dict[str, float]:
    rows = await conn.fetch("SELECT key, value FROM heartbeat_config")
    return {r["key"]: float(r["value"]) for r in rows}

async def _get_maintenance_config(conn: asyncpg.Connection) -> dict[str, float]:
    rows = await conn.fetch("SELECT key, value FROM maintenance_config")
    return {r["key"]: float(r["value"]) for r in rows}


async def _set_config(conn: asyncpg.Connection, key: str, value: Any) -> None:
    await conn.execute("SELECT set_config($1, $2::jsonb)", key, json.dumps(value))


async def _run_init(dsn: str, *, wait_seconds: int) -> int:
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        await _ensure_schema_has_config(conn)

        hb = await _get_heartbeat_config(conn)
        maint = {}
        try:
            maint = await _get_maintenance_config(conn)
        except Exception:
            maint = {}
        default_interval = int(hb.get("heartbeat_interval_minutes", 60))
        default_max_energy = float(hb.get("max_energy", 20))
        default_regen = float(hb.get("base_regeneration", 10))
        default_max_active_goals = int(hb.get("max_active_goals", 3))
        default_maint_interval = int(maint.get("maintenance_interval_seconds", 60)) if maint else 60

        print("AGI init: configure heartbeat + objectives + guardrails.\n")

        heartbeat_interval = _prompt_int(
            "Heartbeat interval (minutes)", default=default_interval, min_value=1
        )
        maintenance_interval = _prompt_int(
            "Subconscious maintenance interval (seconds)",
            default=default_maint_interval,
            min_value=1,
        )
        max_energy = _prompt_float("Max energy budget", default=default_max_energy, min_value=0.0)
        base_regeneration = _prompt_float(
            "Energy regenerated per heartbeat", default=default_regen, min_value=0.0
        )
        max_active_goals = _prompt_int(
            "Max active goals", default=default_max_active_goals, min_value=0
        )

        objectives = _prompt_list("Major objectives", required=True)
        guardrails = _prompt_list("Guardrails / boundaries (plain language)", required=False)
        initial_message = _prompt(
            "Initial message to the AGI (stored + provided to the heartbeat)",
            default="",
            required=False,
        )

        print("\nModel configuration (stored in DB; worker will also use env vars for keys).")
        hb_provider = _prompt(
            "Heartbeat model provider (openai|anthropic|openai_compatible|ollama|openrouter)",
            default=os.getenv("LLM_PROVIDER", "openai"),
            required=True,
        )
        hb_model = _prompt("Heartbeat model", default=os.getenv("LLM_MODEL", "gpt-4o"), required=True)
        hb_endpoint = _prompt(
            "Heartbeat endpoint (blank for provider default)",
            default=os.getenv("OPENAI_BASE_URL", ""),
            required=False,
        )
        hb_key_env = _prompt(
            "Heartbeat API key env var name (e.g. OPENAI_API_KEY; blank for none)",
            default="OPENROUTER_API_KEY" if hb_provider == "openrouter" else ("OPENAI_API_KEY" if hb_provider.startswith("openai") else ""),
            required=False,
        )

        chat_provider = _prompt(
            "Chat model provider (openai|anthropic|openai_compatible|ollama|openrouter)",
            default=hb_provider,
            required=True,
        )
        chat_model = _prompt("Chat model", default=hb_model, required=True)
        chat_endpoint = _prompt("Chat endpoint (blank for provider default)", default=hb_endpoint, required=False)
        chat_key_env = _prompt(
            "Chat API key env var name (blank for none)",
            default=hb_key_env,
            required=False,
        )

        contact_channels = _prompt_list(
            "How should the AGI reach you? (e.g. email, sms, telegram, signal) [names only]",
            required=False,
        )
        contact_details: dict[str, str] = {}
        for ch in contact_channels:
            contact_details[ch] = _prompt(f"  {ch} destination (address/handle)", default="", required=False, secret=False)

        tools = _prompt_list(
            "Tools the AGI can use (e.g. email, sms, tweet, web_research) [names only]",
            required=False,
        )

        enable_autonomy = _prompt_yes_no("Enable autonomous heartbeats now?", default=True)
        enable_maintenance = _prompt_yes_no("Enable subconscious maintenance now?", default=True)

        async with conn.transaction():
            await conn.execute(
                "UPDATE heartbeat_config SET value = $1 WHERE key = 'heartbeat_interval_minutes'",
                float(heartbeat_interval),
            )
            await conn.execute(
                "UPDATE heartbeat_config SET value = $1 WHERE key = 'max_energy'", float(max_energy)
            )
            await conn.execute(
                "UPDATE heartbeat_config SET value = $1 WHERE key = 'base_regeneration'",
                float(base_regeneration),
            )
            await conn.execute(
                "UPDATE heartbeat_config SET value = $1 WHERE key = 'max_active_goals'",
                float(max_active_goals),
            )
            try:
                await conn.execute(
                    "UPDATE maintenance_config SET value = $1 WHERE key = 'maintenance_interval_seconds'",
                    float(maintenance_interval),
                )
            except Exception:
                pass

            await _set_config(conn, "agent.objectives", objectives)
            await _set_config(
                conn,
                "agent.budget",
                {
                    "max_energy": max_energy,
                    "base_regeneration": base_regeneration,
                    "heartbeat_interval_minutes": heartbeat_interval,
                    "max_active_goals": max_active_goals,
                },
            )
            await _set_config(conn, "agent.guardrails", guardrails)
            await _set_config(conn, "agent.initial_message", initial_message)
            await _set_config(conn, "agent.tools", [{"name": t, "enabled": True} for t in tools])

            await _set_config(
                conn,
                "llm.heartbeat",
                {
                    "provider": hb_provider,
                    "model": hb_model,
                    "endpoint": hb_endpoint,
                    "api_key_env": hb_key_env,
                },
            )
            await _set_config(
                conn,
                "llm.chat",
                {
                    "provider": chat_provider,
                    "model": chat_model,
                    "endpoint": chat_endpoint,
                    "api_key_env": chat_key_env,
                },
            )
            await _set_config(
                conn,
                "user.contact",
                {
                    "channels": contact_channels,
                    "destinations": contact_details,
                },
            )

            await _set_config(conn, "agent.is_configured", True)

            if enable_autonomy:
                await conn.execute("UPDATE heartbeat_state SET is_paused = FALSE WHERE id = 1")
            else:
                await conn.execute("UPDATE heartbeat_state SET is_paused = TRUE WHERE id = 1")

            try:
                if enable_maintenance:
                    await conn.execute("UPDATE maintenance_state SET is_paused = FALSE WHERE id = 1")
                else:
                    await conn.execute("UPDATE maintenance_state SET is_paused = TRUE WHERE id = 1")
            except Exception:
                pass

        print("\nSaved configuration to Postgres `config` table.")
        print("Next steps:")
        print("- Start services: `docker compose up -d`")
        print("- Start workers: `docker compose --profile active up -d` (or `--profile heartbeat` / `--profile maintenance`)")
        print("- Verify: `SELECT is_agent_configured();`, `SELECT should_run_heartbeat();`, `SELECT should_run_maintenance();`")
        return 0
    finally:
        await conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agi init", description="Interactive bootstrap for AGI configuration (stored in Postgres).")
    p.add_argument("--dsn", default=None, help="Postgres DSN; defaults to POSTGRES_* env vars")
    p.add_argument("--wait-seconds", type=int, default=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.dsn:
        dsn = args.dsn
    else:
        dsn = _env_db_config().dsn()

    try:
        return asyncio.run(_run_init(dsn, wait_seconds=args.wait_seconds))
    except KeyboardInterrupt:
        _print_err("\nCancelled.")
        return 130
    except Exception as e:
        _print_err(f"init failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
