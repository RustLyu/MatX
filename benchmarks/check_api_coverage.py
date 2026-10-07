#!/usr/bin/env python3
"""Check that every public MatX symbol has a benchmark call site."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


INTERNAL_FACTORIES = {
    "matx_blas_make_reference",
    "matx_dense_linsolve_make_cblas",
    "matx_linsolve_make_cxsparse",
    "matx_linsolve_make_mumps",
    "matx_linsolve_make_suitesparse_klu",
    "matx_linsolve_make_superlu",
    "matx_linsolve_make_umfpack",
    "matx_sparse_make_reference_aocl",
    "matx_sparse_make_reference_grb",
    "matx_vec_blas_make_reference",
}


def exported_matx_symbols(build_dir: Path) -> set[str]:
    archives = sorted(build_dir.rglob("libmatx_*.a"))
    if not archives:
        raise RuntimeError(f"no libmatx_*.a archives found under {build_dir}")

    symbols: set[str] = set()
    for archive in archives:
        output = subprocess.run(
            ["nm", "-g", "--defined-only", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 2 and re.fullmatch(r"matx_[A-Za-z0-9_]+", fields[-1]):
                symbols.add(fields[-1])
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build-verify"))
    parser.add_argument(
        "--benchmark-source",
        type=Path,
        default=Path(__file__).with_name("matx_benchmarks.cpp"),
    )
    args = parser.parse_args()

    try:
        symbols = exported_matx_symbols(args.build_dir)
        source = args.benchmark_source.read_text(encoding="utf-8")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"API coverage check failed: {error}", file=sys.stderr)
        return 2

    public_symbols = symbols - INTERNAL_FACTORIES
    called_symbols = {
        symbol
        for symbol in public_symbols
        if re.search(rf"\b{re.escape(symbol)}\s*\(", source)
    }
    missing = sorted(public_symbols - called_symbols)
    if missing:
        print("Public symbols missing from benchmark source:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        return 1

    print(
        f"API call-site coverage: {len(called_symbols)}/{len(public_symbols)} "
        f"public exported symbols called from benchmark source; "
        f"{len(INTERNAL_FACTORIES & symbols)} internal factories excluded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
