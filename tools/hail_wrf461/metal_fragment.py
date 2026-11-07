#!/usr/bin/env python3
"""Emit the Metal HAILCAST fragment from the CUDA translation.

The statements are untouched; address spaces, the binary64 type and the
four forward gotos (Metal has none; they become breaks out of the
integration and out of the embryo body) change.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("hailcast_wrf", "hailcast_wrf39")


def generate(stem):
    text = (ROOT / f"woof/core/kernels/{stem}.cuh").read_text()
    text = text.replace("__device__ ", "")
    text = text.replace("const HailArray<float>&",
                        "thread const HailStorage<float,NZ>&")
    text = text.replace("if((zfzl<(zbas-1000))) { goto hail_label_400; }",
                        "do { if((zfzl<(zbas-1000))) { break; }")
    text = text.replace("hail_label_400:;", "} while(0);")
    for label in (100, 200, 300):
        text = text.replace(f"{{ goto hail_label_{label}; }}", "{ break; }")
        text = text.replace(f"hail_label_{label}:;", "")
    assert "goto" not in text
    text = re.sub(r"\b(float|int)&", r"thread \1&", text)
    text = re.sub(r"\bdouble&", "thread HailR8&", text)
    text = re.sub(r"\bdouble\b", "HailR8", text)
    output = ROOT / f"woof/metal/shaders/{stem}_fragment.metalh"
    output.write_text(f"// Generated from woof/core/kernels/{stem}.cuh by "
                      "tools/hail_wrf461/metal_fragment.py.\n" + text)


def main():
    for stem in SOURCES:
        generate(stem)


if __name__ == "__main__":
    main()
