"""Checkout wrapper for the installed woof hex score command."""
from woof.hex.cli import main
import sys
if __name__ == "__main__":
    raise SystemExit(main(["score", *sys.argv[1:]]))
