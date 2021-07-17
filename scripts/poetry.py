__all__ = ["reformat"]

import subprocess
import sys
from pathlib import Path
from typing import AnyStr, Union


BASE_PATH = Path(__file__).resolve().parents[1]


def _write_output(out: Union[str, bytes], /, err: bool = False):
    file = sys.stderr if err else sys.stdout
    print(out.decode() if isinstance(out, bytes) else out, file=file)


def _poetry_run(cmd: str, *args: str):
    _write_output(f"{BASE_PATH}$ poetry run {cmd}")
    try:
        res = subprocess.check_output(
            ["poetry", "run", cmd, *args], cwd=BASE_PATH
        )
    except subprocess.CalledProcessError as e:
        _write_output(e.output, err=True)
        sys.exit(e.returncode)
    else:
        _write_output(res)


def reformat():
    _poetry_run("black", ".")
    _poetry_run("isort", ".")


def check():
    _poetry_run("black", "--check", ".")
    _poetry_run("isort", "--check", ".")
    _poetry_run("mypy")


if __name__ == "__main__":
    check()