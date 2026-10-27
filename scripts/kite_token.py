"""
Daily Zerodha access-token refresh.

The token expires every morning around 07:30 IST and every data call fails
after that, so this runs once before the session starts.

    python -m scripts.kite_token                 # step 1: print the login URL
    python -m scripts.kite_token <request_token> # step 2: exchange and save

The request_token is the value Zerodha puts in the redirect URL after you
log in. It is single-use and expires in MINUTES, which is why it cannot be
committed to a file and re-run the next day.

Credentials come from the environment (ZERODHA_API_KEY / ZERODHA_API_SECRET),
never from source. `--write` updates the .env in place so the running bot's
next restart picks it up; without it the token is only printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._io import write_text  # noqa: E402

from src.kite_env import load_dotenv as _load_dotenv  # noqa: E402  (also loads the .env)

ENV_KEY = "ZERODHA_ACCESS_TOKEN"


def _env_path(explicit: Optional[str]) -> Path:
    """Where to write. Mirrors the bot's own search order so the file it
    writes is the file the bot will read."""
    if explicit:
        return Path(explicit)
    loaded = _load_dotenv()
    if loaded is not None:
        return loaded
    return Path.cwd() / ".env"


def _upsert(path: Path, key: str, value: str) -> None:
    """Replace KEY= in place, or append it. Never rewrites anything else —
    this file holds the Telegram token and the API secret too."""
    lines = path.read_text().splitlines() if path.is_file() else []
    out, replaced = [], False
    for line in lines:
        if line.strip().startswith(f"{key}=") and not line.strip().startswith("#"):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    write_text(path, "\n".join(out) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request_token", nargs="?",
                    help="the request_token from the redirect URL after login")
    ap.add_argument("--write", action="store_true",
                    help=f"write {ENV_KEY} into the .env the bot reads")
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args(argv)

    _load_dotenv()
    api_key = os.getenv("ZERODHA_API_KEY", "")
    api_secret = os.getenv("ZERODHA_API_SECRET", "")
    if not api_key or not api_secret:
        print("ZERODHA_API_KEY / ZERODHA_API_SECRET are not set.\n"
              "Put them in your .env (they do not change day to day):\n"
              "    ZERODHA_API_KEY=...\n"
              "    ZERODHA_API_SECRET=...")
        return 2

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("kiteconnect is not installed:  pip install kiteconnect")
        return 2

    kite = KiteConnect(api_key=api_key)

    if not args.request_token:
        print("1. Open this URL and log in:\n")
        print(f"   {kite.login_url()}\n")
        print("2. After login you land on a redirect URL containing")
        print("   `request_token=XXXX`. Copy that value and run:\n")
        print("   python -m scripts.kite_token XXXX --write\n")
        print("   It is single-use and expires in minutes — do step 2 straight "
              "away.")
        return 0

    try:
        data = kite.generate_session(args.request_token, api_secret=api_secret)
        token = data["access_token"]
    except Exception as e:
        print(f"could not exchange the request_token: {e}\n"
              f"The most common cause is that it was already used or has "
              f"expired — start again from step 1.")
        return 1

    if args.write:
        path = _env_path(args.env_file)
        try:
            _upsert(path, ENV_KEY, token)
            print(f"{ENV_KEY} written to {path}")
            print("Restart the bot (or send it /token <value> on Telegram) to "
                  "pick it up.")
        except Exception as e:
            print(f"could not write {path}: {e}")
            print(f"export {ENV_KEY}={token}")
            return 1
    else:
        print(f"export {ENV_KEY}={token}")
        print("(re-run with --write to save it into your .env)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
