"""``python -m ghostlm.agent`` entry point. See ``runner.cli_main``."""

import sys

from .runner import cli_main


if __name__ == "__main__":
    sys.exit(cli_main())
