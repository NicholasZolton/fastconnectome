"""Dataset management commands."""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run connectomes as agents")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download and compile a model")
    prepare.add_argument("model", choices=["malecns-v1"])
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser


def _prepare_malecns(data_dir: Path) -> None:
    if importlib.util.find_spec("stonkfly") is None:
        raise RuntimeError(
            "MaleCNS support is optional; install `fastconnectome[malecns]`"
        )
    environment = os.environ.copy()
    environment["STONKFLY_DATA"] = str(data_dir.resolve())
    subprocess.run(
        [sys.executable, "-m", "stonkfly", "prepare"],
        check=True,
        env=environment,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare" and args.model == "malecns-v1":
        _prepare_malecns(args.data_dir)


if __name__ == "__main__":
    main()
