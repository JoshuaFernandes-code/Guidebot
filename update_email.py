#!/usr/bin/env python3
"""
update_email.py — Manage dashboard login emails and per-user passwords.

Usage:
    python update_email.py add user@example.com
    python update_email.py list
    python update_email.py delete user@example.com
    python update_email.py delete-by-id 5
    python update_email.py set-password user@example.com    # prompts (hidden input)
    echo 'my-secret' | python update_email.py set-password user@example.com

Each allowed email has its OWN password. Passwords can ONLY be set/changed
through this script. They are stored as salted PBKDF2-HMAC-SHA256 hashes —
never in plaintext. Changing a user's password logs out only that user.
"""

import getpass
import sys
from pipeline import emails_db


def cmd_add(email: str, added_by: str | None = None):
    try:
        result = emails_db.add_email(email, added_by=added_by)
        print(f"Added: {result['email']} (id={result['id']}, added_at={result['added_at']})")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list():
    emails = emails_db.list_emails()
    if emails:
        print(f"{'ID':<6} {'Email':<40} {'Password':<10} {'Added At':<30} {'Added By'}")
        print("-" * 100)
        for e in emails:
            has_pw = "set" if emails_db.has_password(e["email"]) else "NOT SET"
            print(f"{e['id']:<6} {e['email']:<40} {has_pw:<10} {e['added_at']:<30} {e['added_by'] or '-'}")
    else:
        print("No allowed emails registered.")

    missing = emails_db.users_without_password()
    if missing:
        print("\nUsers WITHOUT a password (login rejected until set):")
        for u in missing:
            print(f"  - {u['email']}  ->  python update_email.py set-password {u['email']}")


def cmd_delete(email: str):
    if emails_db.delete_email(email):
        print(f"Deleted: {email}")
    else:
        print(f"Not found: {email}", file=sys.stderr)
        sys.exit(1)


def cmd_delete_by_id(email_id: int):
    if emails_db.delete_email_by_id(email_id):
        print(f"Deleted email id={email_id}")
    else:
        print(f"Not found: id={email_id}", file=sys.stderr)
        sys.exit(1)


def cmd_set_password(email: str):
    if not emails_db.is_allowed(email):
        print(f"Error: email not registered: {email}", file=sys.stderr)
        print("Register it first: python update_email.py add " + email, file=sys.stderr)
        sys.exit(1)

    if not sys.stdin.isatty():
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass(f"New password for {email}: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Error: passwords do not match", file=sys.stderr)
            sys.exit(1)

    try:
        emails_db.set_password(email, password)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Password set for {email}. That user's sessions were invalidated.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "add":
        if len(sys.argv) < 3:
            print("Usage: python update_email.py add <email>", file=sys.stderr)
            sys.exit(1)
        cmd_add(sys.argv[2])

    elif command == "list":
        cmd_list()

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python update_email.py delete <email>", file=sys.stderr)
            sys.exit(1)
        cmd_delete(sys.argv[2])

    elif command == "delete-by-id":
        if len(sys.argv) < 3:
            print("Usage: python update_email.py delete-by-id <id>", file=sys.stderr)
            sys.exit(1)
        try:
            cmd_delete_by_id(int(sys.argv[2]))
        except ValueError:
            print("Error: ID must be an integer", file=sys.stderr)
            sys.exit(1)

    elif command == "set-password":
        if len(sys.argv) < 3:
            print("Usage: python update_email.py set-password <email>", file=sys.stderr)
            sys.exit(1)
        cmd_set_password(sys.argv[2])

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()