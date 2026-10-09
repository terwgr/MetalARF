#!/usr/bin/env python3
"""Generate typed MSL bindings and Python launch layouts from one registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "woof" / "metal" / "kernel_signatures.json"
OUTPUT = ROOT / "woof" / "metal" / "generated"
MSL_TYPES = {"float": "float", "float32": "float", "int32": "int"}
RESOURCE_MSL_TYPES = {**MSL_TYPES, "uint32": "uint", "int64": "long",
                      "uint64": "ulong", "float64_bits": "ulong"}
CTYPES_TYPES = {"float32": "ctypes.c_float", "int32": "ctypes.c_int32"}


def generate(output_dir: Path = OUTPUT) -> list[Path]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("schema") != "woof-metal-kernel-signatures-v1":
        raise ValueError("unsupported Metal kernel signature schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    python_rows = []
    manifest_rows = []
    source_bindings: dict[str, list[str]] = {}
    for kernel in registry["kernels"]:
        name = kernel["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError(f"invalid kernel name {name!r}")
        resources = kernel["resources"]
        shapes = kernel["shapes"]
        params = kernel["parameters"]
        if len(resources) > 64:
            raise ValueError(f"{name}: maximum of 64 resource arguments")
        if set(shapes) != {row[0] for row in resources}:
            raise ValueError(f"{name}: every resource needs a declared shape rule")
        if any(expr == "any_1d" for alternatives in shapes.values()
               for shape in alternatives for expr in shape):
            if any(shape != ["any_1d"] for alternatives in shapes.values()
                   for shape in alternatives if "any_1d" in shape):
                raise ValueError(f"{name}: any_1d must be the only shape extent")
        layout = {k: kernel[k] for k in (
            "name", "module", "resources", "shapes", "parameters",
            "threads_per_group", "layout_version", "arithmetic_policies")}
        layout_json = json.dumps(layout, sort_keys=True, separators=(",", ":"))
        layout_hash = hashlib.sha256(layout_json.encode()).hexdigest()
        shader_path = ROOT / kernel["source"]
        shader_hash = hashlib.sha256(shader_path.read_bytes()).hexdigest()
        title = "".join(part.title() for part in name.split("_"))
        msl = [f"// Generated from kernel_signatures.json; layout {layout_hash}.",
               f"struct {title}Resources {{"]
        for index, (field, storage, access) in enumerate(resources):
            if storage not in RESOURCE_MSL_TYPES or access not in ("read", "read_write"):
                raise ValueError(f"{name}: invalid resource descriptor {field}")
            const = "const " if access == "read" else ""
            msl.append(f"    device {const}{RESOURCE_MSL_TYPES[storage]} *{field} [[id({index})]];")
        msl.append("};")
        msl.append(f"struct {title}Params {{")
        for field, dtype in params:
            if dtype not in MSL_TYPES:
                raise ValueError(f"{name}: invalid parameter type {dtype}")
            msl.append(f"    {MSL_TYPES[dtype]} {field};")
        msl.append("};")
        msl.append(f"static_assert(sizeof({title}Params) == {4 * len(params)}, "
                   f'"{name} parameter ABI size changed");')
        binding_path = output_dir / f"{name}_bindings.metalh"
        binding_path.write_text("\n".join(msl) + "\n", encoding="utf-8")
        generated.append(binding_path)
        source_bindings.setdefault(Path(kernel["source"]).stem, []).append(
            binding_path.name)
        fields = [(field, CTYPES_TYPES[dtype]) for field, dtype in params]
        python_rows.append(
            f"{name.upper()}_RESOURCES = {tuple(tuple(row) for row in resources)!r}\n"
            f"{name.upper()}_ACCESS = {tuple(row[2] for row in resources)!r}\n"
            f"{name.upper()}_SHAPES = {tuple((field, tuple(tuple(shape) for shape in alternatives)) for field, alternatives in shapes.items())!r}\n"
            f"{name.upper()}_PARAM_FIELDS = {fields!r}\n"
            f"{name.upper()}_THREADS_PER_GROUP = {kernel['threads_per_group']}\n"
            f"{name.upper()}_LAYOUT_HASH = {layout_hash!r}\n"
            f"{name.upper()}_SHADER_HASH = {shader_hash!r}\n")
        manifest_rows.append({"name": name, "module": kernel["module"],
                              "cuda_source": kernel["cuda_source"],
                              "cuda_symbol": kernel["cuda_symbol"],
                              "metal_source": kernel["source"],
                              "resource_count": len(resources),
                              "parameter_size": 4 * len(params),
                              "threads_per_group": kernel["threads_per_group"],
                              "layout_version": kernel["layout_version"],
                              "layout_hash": layout_hash,
                              "shader_hash": shader_hash,
                              "arithmetic_policies": kernel["arithmetic_policies"]})
    # Shaders include one source-family header. Generate that header as a
    # stable aggregator over the per-entrypoint ABI structs above.
    for source_stem, headers in sorted(source_bindings.items()):
        aggregate_path = output_dir / f"{source_stem}_bindings.metalh"
        if aggregate_path.name in headers:
            continue
        aggregate_path.write_text(
            "// Generated from kernel_signatures.json; do not edit.\n"
            + "\n".join(f'#include "{header}"' for header in headers)
            + "\n", encoding="utf-8")
        generated.append(aggregate_path)
    py_path = output_dir / "kernel_specs.py"
    py_path.write_text(
        '"""Generated Metal entry-point signatures; do not edit by hand."""\n'
        "import ctypes\n\n" + "\n".join(python_rows), encoding="utf-8")
    generated.append(py_path)
    init_path = output_dir / "__init__.py"
    init_path.write_text('"""Generated Metal kernel ABI metadata."""\n',
                         encoding="utf-8")
    generated.append(init_path)
    manifest_path = output_dir / "registry.json"
    manifest_path.write_text(json.dumps({"schema": "woof-metal-registry-v1",
                                         "kernels": manifest_rows},
                                        indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    generated.append(manifest_path)
    return generated


def add_fragments(paths: list[Path]) -> list[str]:
    """Append a fragment's new kernels to the registry (append-only).

    A kernel already in the registry must match the fragment exactly; a
    differing entry is an ABI change and is refused, not overwritten.
    """
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_name = {kernel["name"]: kernel for kernel in registry["kernels"]}
    added = []
    for path in paths:
        fragment = json.loads(Path(path).read_text(encoding="utf-8"))
        kernels = fragment["kernels"] if "kernels" in fragment else [fragment]
        for kernel in kernels:
            current = by_name.get(kernel["name"])
            if current is None:
                registry["kernels"].append(kernel)
                by_name[kernel["name"]] = kernel
                added.append(kernel["name"])
            elif current != kernel:
                raise ValueError(f"{path}: {kernel['name']} differs from the "
                                 "registered ABI; registry entries are append-only")
    if added:
        REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return added


def replace_fragments(paths: list[Path]) -> list[str]:
    """Replace registered kernels with a fragment's revised ABI.

    The deliberate counterpart of ``add_fragments``: every kernel in the
    fragment must already be registered, so a typo cannot add a kernel, and
    the registry keeps its order. The caller rebuilds the shader and bundle
    in the same commit as the revised signature.
    """
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    index = {kernel["name"]: position
             for position, kernel in enumerate(registry["kernels"])}
    replaced = []
    for path in paths:
        fragment = json.loads(Path(path).read_text(encoding="utf-8"))
        kernels = fragment["kernels"] if "kernels" in fragment else [fragment]
        for kernel in kernels:
            position = index.get(kernel["name"])
            if position is None:
                raise ValueError(f"{path}: {kernel['name']} is not registered; "
                                 "use --add-fragment for new kernels")
            if registry["kernels"][position] != kernel:
                registry["kernels"][position] = kernel
                replaced.append(kernel["name"])
    if replaced:
        REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare generated files without writing them")
    parser.add_argument("--add-fragment", action="append", default=[], metavar="JSON",
                        help="append this signature fragment's new kernels to "
                             "kernel_signatures.json before generating (repeatable)")
    parser.add_argument("--replace-fragment", action="append", default=[], metavar="JSON",
                        help="replace already-registered kernels with this fragment's "
                             "revised ABI before generating (repeatable)")
    args = parser.parse_args()
    if args.replace_fragment:
        print(json.dumps({"replaced": replace_fragments(args.replace_fragment)}))
    if args.add_fragment:
        print(json.dumps({"appended": add_fragments(args.add_fragment)}))
    if args.check:
        with tempfile.TemporaryDirectory(prefix="woof-metal-signatures-") as tmp:
            generated = generate(Path(tmp))
            mismatches = []
            for path in generated:
                relative = path.relative_to(tmp)
                checked_in = OUTPUT / relative
                if not checked_in.is_file() or path.read_bytes() != checked_in.read_bytes():
                    mismatches.append(str(relative))
            if mismatches:
                print(json.dumps({"stale_or_missing": mismatches}))
                return 1
        print(json.dumps({"checked": len(generated), "status": "current"}))
        return 0
    paths = generate()
    print(json.dumps([str(path.relative_to(ROOT)) for path in paths]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
