---
name: intel-sycl-toolchain
description: >
  Install, configure and troubleshoot an Intel oneAPI DPC++/SYCL toolchain (icpx,
  sycl-ls) and Intel MKL on Linux, including ephemeral containers and sandboxes with
  restricted network egress. Use this whenever the user wants to set up, install or
  repair a SYCL / DPC++ / oneAPI / icpx build environment, wants to compile or
  benchmark SYCL or ESIMD code, or hits errors like "libsycl.so.N: cannot open shared
  object file", "No device of requested type available", "Required aspect
  ext_intel_esimd is not supported on the device", "Unrecognized build options:
  -vc-codegen", or "icpx: command not found" on a machine that had it earlier. Also
  consult it before writing or building any SYCL/ESIMD program in a fresh container,
  since the toolchain is usually not preinstalled and disappears when the container is
  recycled.
---

# Intel oneAPI DPC++/SYCL toolchain

Two routes install `icpx`, and they are not interchangeable — they differ in what
else you get and, crucially, in whether the binaries they produce can run without a
sourced environment. Pick deliberately; the wrong choice surfaces later as a test
harness that cannot find a SYCL device.

## Step 1: find out which routes are open

Network egress is set per environment and it changes. Do not assume — a policy that
blocked Intel last month may allow it today, and vice versa. Probe first; it costs
seconds and decides everything downstream.

```bash
for h in apt.repos.intel.com repo.anaconda.com registrationcenter-download.intel.com; do
  printf '%-42s %s\n' "$h" \
    "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 https://$h/ 2>&1 || echo unreachable)"
done
```

`200` means reachable. `000`, `400`, a timeout or a curl error means blocked. GitHub
is blocked in many sandboxes, so building DPC++ from source or fetching `intel/llvm`
release tarballs is usually not an option regardless.

## Step 2: choose a route

| | **A. Intel apt repo** | **B. conda-forge** |
|---|---|---|
| Needs | `apt.repos.intel.com` | `repo.anaconda.com` |
| Gives | full oneAPI: compiler, MKL, VTune, Advisor | compiler + OpenCL CPU runtime |
| Binaries run without activation | **no** — must `source setvars.sh` | **yes** — with one `-rpath` flag |
| Install size | several GB | ~5 GB env |

**Both provide a working `icpx` and an OpenCL CPU device that actually runs kernels.**
The deciding question is usually the third row.

Choose **A** when you need the wider oneAPI stack (MKL, profilers) and you control the
shell that runs your binaries.

Choose **B** when binaries must run *without* a sourced environment — CMake/ctest,
CI jobs, scripts, anything spawned by a harness that does not inherit your shell.
This is a real and verified difference, not a preference: a conda-built binary with
`-Wl,-rpath,$CONDA_PREFIX/lib` runs in a completely clean environment, while an
apt/oneAPI binary fails device discovery even with `-rpath` *and* `LD_LIBRARY_PATH`
set, because `setvars.sh` also wires up OpenCL ICD registration and half a dozen
component library directories (tcm, umf, tbb under a versioned `intel64/gcc4.8`
subdirectory, debugger, …). Reproducing that by hand is a rabbit hole; don't try.

The bundled script probes and installs whichever route is available:

```bash
bash scripts/install-sycl-toolchain.sh            # auto-detect
bash scripts/install-sycl-toolchain.sh --route conda   # force
```

It is idempotent — re-running when `icpx` already exists just prints the version, so
it is safe at the start of any session.

## Step 3: build and run

**Route A (apt/oneAPI).** Source the environment in *every* shell that compiles or
runs:

```bash
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O2 -std=c++17 prog.cpp -o prog && ./prog
```

**Route B (conda-forge).** Activate to compile, and bake in the rpath so the binary
stands alone afterwards:

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate sycl
icpx -fsycl -O2 -std=c++17 -Wl,-rpath,"$CONDA_PREFIX/lib" prog.cpp -o prog
./prog        # works from anywhere, no activation needed
```

The `-Wl,-rpath` is what makes route B's binaries relocatable. Omit it and you get
`libsycl.so.9: cannot open shared object file` the moment the binary runs outside an
activated shell — which is exactly what happens under ctest.

## Step 4: verify before trusting any result

```bash
sycl-ls
```

In a typical container this prints exactly one device:

```
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) ... OpenCL 3.0
```

**That line governs what you may conclude from anything you run.** An OpenCL *CPU*
device executes SYCL kernels correctly, so it validates numerics, APIs and logic. It
is not a GPU: it says nothing about Xe throughput, and some features refuse to run on
it entirely (see `references/esimd.md`). When reporting benchmark results from such a
device, say so plainly rather than letting CPU timings stand in as GPU evidence —
that distinction is usually the entire point of the exercise.

To ask the device what it supports instead of guessing, query aspects directly;
`references/troubleshooting.md` has a ready-made probe.

## Intel MKL

MKL has its own routes, and the distro one is easy to overlook:

- `apt install libmkl-dev` — Ubuntu's **own** archive, unrelated to Intel's servers,
  so it works even when Intel's repo is blocked. Header check:
  `/usr/include/mkl/mkl_compact.h`.
- `apt install intel-oneapi-mkl-devel` — Intel's repo, newer, part of route A.

For most builds the distro package is sufficient and far smaller.

## Expect the toolchain to vanish

Containers get recycled, and `/opt/conda` or `/opt/intel` goes with them. The symptom
is `icpx: command not found` on a machine where you were compiling minutes earlier.
This is normal, not a bug to investigate — just re-run the installer.

Two things follow:

- **Anything worth keeping must be committed to the repo**, not left in the container.
  That includes this skill, which is why it lives in the repo's `.claude/skills/`
  rather than `~/.claude/skills/`.
- **A `SessionStart` hook can front-run the problem** by reinstalling automatically;
  `references/troubleshooting.md` has the pattern (async so it does not block startup,
  idempotent so it is cheap when already satisfied).

## Going further

- `references/troubleshooting.md` — each failure mode with its exact error text and
  fix: the two distinct rpath/device-discovery errors, missing `omp.h`, a
  half-installed conda prefix, the `sycl::info` namespace ambiguity, the device-aspect
  probe, and the SessionStart hook. Read it when a build or run fails.
- `references/esimd.md` — Intel ESIMD (`sycl::ext::intel::esimd`): why it compiles
  everywhere but runs only on Intel GPUs, its two runtime failure modes, cross-TU
  linking with `SYCL_EXTERNAL`, and the extra flags `invoke_simd` needs. Read it
  before writing or debugging ESIMD code.
