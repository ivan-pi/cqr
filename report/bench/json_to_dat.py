#!/usr/bin/env python3
"""Convert Google Benchmark JSON (from bench_geqrf_collect) into a report .dat.

Reads the JSON emitted by `bench_geqrf_collect --benchmark_format=json` and
writes the nine-column table the gnuplot scripts expect (see
report/data/README.md). One row per matrix order, joining the three
implementations measured at that order.

Usage:
    json_to_dat.py results.json -o ../data/geqrf_x86.dat            # auto vendor
    json_to_dat.py results.json -o ../data/geqrf_arm.dat --vendor armpl
    json_to_dat.py results.json --stat mean                        # default: median

The vendor is the non-cqr, non-lapack implementation present in the JSON
(`mkl` on x86, `armpl` on Arm); pass --vendor to force it. A provenance header
(host, CPU count, clock, and the source JSON) is written as #-comments.
"""
import argparse
import json
import os
import statistics
import sys


def impl_and_n(entry):
    """(impl, n) for a benchmark entry. impl from the run name's first segment;
    n from the `n` counter if present, else the name's second segment."""
    run = entry.get("run_name", entry["name"])
    parts = run.split("/")
    impl = parts[0]
    if "n" in entry:
        n = int(round(float(entry["n"])))
    else:
        n = int(parts[1])
    return impl, n


def collect(benchmarks, stat):
    """{(impl, n): {mat_per_s, gflops, relerr}} using the chosen aggregate.

    Prefer Google Benchmark's own `<stat>` aggregate rows; if the JSON has only
    raw iterations, aggregate them here with the same statistic."""
    aggregates = [b for b in benchmarks if b.get("run_type") == "aggregate"]
    out = {}
    if any(b.get("aggregate_name") == stat for b in aggregates):
        for b in aggregates:
            if b.get("aggregate_name") != stat:
                continue
            impl, n = impl_and_n(b)
            out[(impl, n)] = {
                "mat_per_s": float(b.get("mat_per_s", 0.0)),
                "gflops": float(b.get("gflops", 0.0)),
                "relerr": float(b.get("relerr", 0.0)),
            }
        return out
    # Fallback: no aggregates -> group raw iterations and reduce ourselves.
    reducer = {"median": statistics.median, "mean": statistics.fmean}[stat]
    groups = {}
    for b in benchmarks:
        if b.get("run_type") == "aggregate":
            continue
        impl, n = impl_and_n(b)
        g = groups.setdefault((impl, n), {"mat_per_s": [], "gflops": [], "relerr": []})
        g["mat_per_s"].append(float(b.get("mat_per_s", 0.0)))
        g["gflops"].append(float(b.get("gflops", 0.0)))
        if "relerr" in b:
            g["relerr"].append(float(b["relerr"]))
    for key, g in groups.items():
        out[key] = {
            "mat_per_s": reducer(g["mat_per_s"]),
            "gflops": reducer(g["gflops"]),
            "relerr": max(g["relerr"]) if g["relerr"] else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="Google Benchmark JSON output")
    ap.add_argument("-o", "--out", help="output .dat (default: stdout)")
    ap.add_argument("--vendor", choices=["auto", "mkl", "armpl"], default="auto")
    ap.add_argument("--stat", choices=["median", "mean"], default="median")
    args = ap.parse_args()

    with open(args.json) as fh:
        doc = json.load(fh)
    data = collect(doc["benchmarks"], args.stat)

    impls = {impl for (impl, _) in data}
    if "cqr" not in impls or "lapack" not in impls:
        sys.exit("error: JSON must contain both 'cqr' and 'lapack' benchmarks")
    vendor = args.vendor
    if vendor == "auto":
        cand = sorted(impls - {"cqr", "lapack"})
        if len(cand) != 1:
            sys.exit(f"error: cannot infer vendor from {sorted(impls)}; pass --vendor")
        vendor = cand[0]

    ns = sorted({n for (impl, n) in data if impl == "cqr"})

    ctx = doc.get("context", {})
    lines = [
        f"# measured data from {os.path.basename(args.json)}  (stat: {args.stat})",
        "# host={host} cpus={cpus} clock_mhz={mhz} caches={caches}".format(
            host=ctx.get("host_name", "?"),
            cpus=ctx.get("num_cpus", "?"),
            mhz=ctx.get("mhz_per_cpu", "?"),
            caches=len(ctx.get("caches", []) or []),
        ),
        "# n  cqr_gflops  cqr_mats_s  {0}_mats_s  lapack_mats_s  "
        "sp_cqr_lap  sp_{0}_lap  sp_cqr_{0}  relerr".format(vendor),
    ]
    for n in ns:
        cqr = data.get(("cqr", n))
        ven = data.get((vendor, n))
        lap = data.get(("lapack", n))
        if not (cqr and ven and lap):
            print(f"warning: missing an implementation at n={n}; skipping",
                  file=sys.stderr)
            continue
        cqr_m, ven_m, lap_m = cqr["mat_per_s"], ven["mat_per_s"], lap["mat_per_s"]
        lines.append(
            f"{n:4d}  {cqr['gflops']:10.3f}  {cqr_m:12.4e}  {ven_m:12.4e}  "
            f"{lap_m:12.4e}  {cqr_m/lap_m:7.3f}  {ven_m/lap_m:7.3f}  "
            f"{cqr_m/ven_m:7.3f}  {cqr['relerr']:.2e}"
        )

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}  ({len(ns)} rows, vendor={vendor})", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
