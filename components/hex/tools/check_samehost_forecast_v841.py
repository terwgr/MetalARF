#!/usr/bin/env python3
"""Compatibility entry point for the shipped forecast receipt door."""
from woof.hex.drivers.forecast_receipt import main

if __name__ == "__main__":
    raise SystemExit(main())
