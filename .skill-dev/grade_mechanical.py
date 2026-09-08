#!/usr/bin/env python3
"""Mechanical half of the eval grading.

Only checks things that are objectively decidable from the artifacts: does a
string appear, did a binary get produced, does the source use a given API.
Emits the matching line as EVIDENCE rather than a bare boolean, because a
keyword can appear in a negated context ("don't use LD_LIBRARY_PATH") and a
bare grep would score that as a pass. Judgment-dependent assertions are left
to a human/grader read and are marked MANUAL.
"""
import json
import pathlib
import re
import sys

WS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def find(text, *patterns, flags=re.I):
    """Return the first matching line for any pattern, else None."""
    for line in text.splitlines():
        for p in patterns:
            if re.search(p, line, flags):
                return line.strip()[:160]
    return None


def grade(eval_name, run_dir):
    out = run_dir / "outputs"
    if not out.is_dir():
        return None
    blob = "\n".join(
        p.read_text(errors="replace") for p in out.rglob("*")
        if p.is_file() and p.suffix in {".md", ".cpp", ".txt", ".cmake", ".sh", ".json"}
    )
    src = "\n".join(
        p.read_text(errors="replace") for p in out.rglob("*.cpp") if p.is_file()
    )
    checks = []

    def add(text, hit, manual=False):
        checks.append({
            "text": text,
            "passed": None if manual else bool(hit),
            "evidence": (hit if isinstance(hit, str) else
                         ("found" if hit else "no match")) if not manual
                        else "MANUAL: needs judgment read",
        })

    if eval_name == "rpath-ctest-failure":
        add("Identifies libs not on default loader path",
            find(blob, r"loader path", r"not on the default", r"rpath", r"LD_LIBRARY_PATH"))
        add("Explains ctest/CI does not inherit the activated shell",
            find(blob, r"ctest.*(inherit|environment)", r"(inherit|environment).*ctest",
                 r"does not inherit"))
        add("Recommends baking in an rpath as the durable fix",
            find(blob, r"-Wl,-rpath", r"\brpath\b"))
        add("Gives a concrete CMake-level fix",
            find(blob, r"target_link_options", r"BUILD_RPATH", r"INSTALL_RPATH",
                 r"set_tests_properties"))
        add("Notes LD_LIBRARY_PATH is a fragile stopgap", None, manual=True)
        add("Verified the fix by actually running something", None, manual=True)

    elif eval_name == "esimd-cpu-reality":
        add("Valid ESIMD program (esimd namespace + explicit_simd attribute)",
            find(src, r"intel::esimd") and find(src, r"sycl_explicit_simd"))
        add("Produced a compiled binary (compile succeeded)",
            "found" if any(p.is_file() and p.stat().st_mode & 0o111 and p.suffix == ""
                           for p in out.rglob("*")) else None)
        add("Actually attempted to run the binary",
            find(blob, r"ext_intel_esimd is not supported", r"\./esimd", r"ran (it|the)",
                 r"executed"))
        add("Explains ESIMD needs an Intel GPU (aspect), not a fixable bug",
            find(blob, r"ext_intel_esimd", r"only .*(Intel )?GPU", r"GPU[- ]only"))
        add("Does NOT falsely claim ESIMD verified end-to-end", None, manual=True)
        add("Checked the real device situation (sycl-ls / aspect query)",
            find(blob, r"sycl-ls", r"aspect::ext_intel_esimd", r"get_devices"))

    elif eval_name == "route-choice-writeup":
        has_conda = find(blob, r"conda-forge", r"dpcpp_linux-64")
        has_apt = find(blob, r"apt\.repos\.intel\.com", r"intel-oneapi-compiler")
        add("Presents both install routes",
            (has_conda + " || " + has_apt) if (has_conda and has_apt) else None)
        add("Makes an explicit recommendation", None, manual=True)
        add("Justifies via the ctest/CI relocatability constraint",
            find(blob, r"ctest", r"\bCI\b", r"relocatab", r"without a sourced",
                 r"sourced (shell|environment)"))
        add("Includes -Wl,-rpath in build instructions",
            find(blob, r"-Wl,-rpath"))
        add("Includes a verification step (sycl-ls)",
            find(blob, r"sycl-ls"))
        add("Gives concrete runnable commands", find(blob, r"```"))

    return checks


rows = []
for eval_dir in sorted(p for p in WS.iterdir() if p.is_dir()):
    for run in ("with_skill", "without_skill"):
        rd = eval_dir / run
        checks = grade(eval_dir.name, rd) if rd.is_dir() else None
        if checks is None:
            continue
        (rd / "grading.json").write_text(json.dumps(
            {"eval_name": eval_dir.name, "run": run, "expectations": checks}, indent=2))
        auto = [c for c in checks if c["passed"] is not None]
        rows.append((eval_dir.name, run,
                     sum(c["passed"] for c in auto), len(auto),
                     sum(1 for c in checks if c["passed"] is None)))

print(f"{'eval':<24} {'run':<15} {'auto-passed':>12}  {'manual':>7}")
for name, run, p, n, m in rows:
    print(f"{name:<24} {run:<15} {p:>6}/{n:<5}  {m:>7}")
if not rows:
    print("(no completed runs yet)")
