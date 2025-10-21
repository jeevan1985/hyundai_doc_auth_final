"""Typer-based user management CLI for the Result GUI module.

This script provides a production-ready command-line interface for managing
application users stored in PostgreSQL. It complements the in-GUI admin tools
and allows one-stop management for:

- Creating users
- Setting/resetting passwords
- Changing roles
- Deleting users
- Listing users (with pagination and search)
- Showing a single user's details

The CLI reads configuration from the local config.yaml with environment-aware
DB settings (see config_loader.py), uses the configured ``auth_db_schema`` and
``app_user_table``, and hashes passwords with bcrypt.

Usage (Windows CMD):
    # Create a virtual environment and install dependencies (ensure typer is installed)
    #   python -m venv .venv
    #   .venv\\Scripts\\activate
    #   pip install -r requirements.txt

    # Create an admin user
    #   python manage_users.py create --username admin --password "StrongP@ss" --role admin

    # Reset a user's password
    #   python manage_users.py set-password --username alice --password "NewP@ss"

    # Change a user's role
    #   python manage_users.py change-role --username alice --role viewer

    # Delete a user
    #   python manage_users.py delete --username alice --yes

    # List users (page 1, 25 per page)
    #   python manage_users.py list --page 1 --per-page 25 --q adm

    # Show details for a single user
    #   python manage_users.py show --username admin
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import typer

from auth import hash_password
from config_loader import AppConfig, load_config
from db import Database, UserRecord

app = typer.Typer(add_completion=False, help="User management CLI for Result GUI")


def _load_db() -> Tuple[AppConfig, Database]:
    """Load configuration and initialize the database layer.

    Returns:
        A tuple of (config, db) where config is the loaded ``AppConfig`` and db
        is an initialized ``Database``.

    Raises:
        SystemExit: If configuration loading fails.
    """
    base_dir = Path(__file__).resolve().parent
    cfg: AppConfig = load_config(base_dir / "config.yaml")
    db = Database(cfg)
    # Ensure auth schema/table exist; tolerant if DB is currently unavailable.
    db.ensure_auth_schema_and_table()
    return cfg, db


@app.command("create")
def cmd_create(
    username: str = typer.Option(..., "--username", help="Username to create"),
    password: str = typer.Option(..., "--password", help="Plaintext password"),
    role: str = typer.Option("viewer", "--role", help="Role: admin|viewer"),
) -> None:
    """Create a new user account.

    Args:
        username: Desired unique username.
        password: Plaintext password (will be bcrypt-hashed).
        role: Role for the user: "admin" or "viewer".

    Raises:
        SystemExit: If the user already exists or DB operation fails.
    """
    role_norm = role.strip().lower()
    if role_norm not in {"admin", "viewer"}:
        typer.secho("Error: role must be 'admin' or 'viewer'", fg=typer.colors.RED)
        raise SystemExit(1)

    _, db = _load_db()
    try:
        pwd_hash = hash_password(password)
        new_id = db.create_user(username=username.strip(), password_hash=pwd_hash, role=role_norm)
        typer.secho(f"Created user '{username}' (id={new_id})", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise SystemExit(1)


@app.command("set-password")
def cmd_set_password(
    username: str = typer.Option(..., "--username", help="Username to update"),
    password: str = typer.Option(..., "--password", help="New plaintext password"),
) -> None:
    """Set or reset a user's password.

    Args:
        username: Username whose password will be updated.
        password: New plaintext password.

    Raises:
        SystemExit: If the user does not exist or DB operation fails.
    """
    _, db = _load_db()
    rec: Optional[UserRecord] = db.get_user_by_username(username.strip())
    if not rec:
        typer.secho(f"Error: user not found: {username}", fg=typer.colors.RED)
        raise SystemExit(1)
    try:
        pwd_hash = hash_password(password)
        db.update_user_password_by_id(user_id=rec.id, new_password_hash=pwd_hash)
        typer.secho(f"Password updated for '{username}'", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise SystemExit(1)


@app.command("change-role")
def cmd_change_role(
    username: str = typer.Option(..., "--username", help="Username to update"),
    role: str = typer.Option(..., "--role", help="New role: admin|viewer"),
) -> None:
    """Change a user's role.

    Args:
        username: Username whose role to change.
        role: New role value: "admin" or "viewer".

    Raises:
        SystemExit: If the user does not exist or DB operation fails.
    """
    role_norm = role.strip().lower()
    if role_norm not in {"admin", "viewer"}:
        typer.secho("Error: role must be 'admin' or 'viewer'", fg=typer.colors.RED)
        raise SystemExit(1)

    _, db = _load_db()
    rec: Optional[UserRecord] = db.get_user_by_username(username.strip())
    if not rec:
        typer.secho(f"Error: user not found: {username}", fg=typer.colors.RED)
        raise SystemExit(1)
    try:
        db.update_user_role(user_id=rec.id, role=role_norm)
        typer.secho(f"Role updated for '{username}' -> {role_norm}", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise SystemExit(1)


@app.command("delete")
def cmd_delete(
    username: str = typer.Option(..., "--username", help="Username to delete"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation"),
) -> None:
    """Delete a user by username.

    Args:
        username: Username to delete.
        yes: If True, skip the confirmation prompt.

    Raises:
        SystemExit: If the user does not exist or DB operation fails.
    """
    _, db = _load_db()
    rec: Optional[UserRecord] = db.get_user_by_username(username.strip())
    if not rec:
        typer.secho(f"Error: user not found: {username}", fg=typer.colors.RED)
        raise SystemExit(1)

    if not yes:
        confirm = typer.confirm(f"Delete user '{username}' (id={rec.id})?")
        if not confirm:
            typer.secho("Aborted.", fg=typer.colors.YELLOW)
            raise SystemExit(0)

    try:
        db.delete_user(user_id=rec.id)
        typer.secho(f"User '{username}' deleted", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise SystemExit(1)


@app.command("list")
def cmd_list(
    q: Optional[str] = typer.Option(None, "--q", help="Case-insensitive username search"),
    page: int = typer.Option(1, "--page", min=1, help="Page number (1-based)"),
    per_page: int = typer.Option(25, "--per-page", min=1, max=200, help="Rows per page"),
) -> None:
    """List users with pagination and optional search.

    Args:
        q: Optional substring to match in usernames (case-insensitive).
        page: 1-based page index.
        per_page: Number of rows per page (1..200).
    """
    _, db = _load_db()
    users, total = db.list_users(page=page, per_page=per_page, search=q)

    if not users:
        typer.secho("No users.", fg=typer.colors.YELLOW)
        return

    header = f"Showing {len(users)} of {total} users (page {page})"
    typer.secho(header, fg=typer.colors.CYAN)
    typer.echo("ID\tUSERNAME\tROLE\tCREATED_AT")
    for u in users:
        typer.echo(f"{u.id}\t{u.username}\t{u.role}\t{u.created_at}")


@app.command("show")
def cmd_show(username: str = typer.Option(..., "--username", help="Username to show")) -> None:
    """Show details for a single user.

    Args:
        username: Username to display.

    Raises:
        SystemExit: If the user does not exist.
    """
    _, db = _load_db()
    rec: Optional[UserRecord] = db.get_user_by_username(username.strip())
    if not rec:
        typer.secho(f"Error: user not found: {username}", fg=typer.colors.RED)
        raise SystemExit(1)

    typer.secho("User details:", fg=typer.colors.CYAN)
    typer.echo(f"  id         : {rec.id}")
    typer.echo(f"  username   : {rec.username}")
    typer.echo(f"  role       : {rec.role}")
    typer.echo(f"  created_at : {rec.created_at}")


if __name__ == "__main__":
    app()  # Typer CLI entry point
