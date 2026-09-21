"""ROCmHub package entrypoint for python -m rocmhub."""

import sys

from rocmhub.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
