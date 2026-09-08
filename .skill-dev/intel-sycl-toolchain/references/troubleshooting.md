# SYCL toolchain troubleshooting

Each entry is a real observed failure: the exact message, why it happens, and the
fix. Match on the error text.

## Contents

- [Runtime: libsycl.so.N cannot open shared object file](#runtime-libsyclson-cannot-open-shared-object-file)
- [Runtime: No device of requested type available](#runtime-no-device-of-requested-type-available)
- [Build: omp.h file not found](#build-omph-file-not-found)
- [Build: reference to 'info' is ambiguous](#build-reference-to-info-is-ambiguous)
- [Install: conda exists but has no conda binary](#install-conda-exists-but-has-no-conda-binary)
- [Install: apt install intel-oneapi-toolkit has no candidate](#install-apt-install-intel-oneapi-toolkit-has-no-candidate)
- [Gone: icpx command not found on a machine that had it](#gone-icpx-command-not-found-on-a-machine-that-had-it)
- [Device aspect probe](#device-aspect-probe)
- [SessionStart hook pattern](#sessionstart-hook-pattern)

---

## Runtime: libsycl.so.N cannot open shared object file

```
./prog: error while loading shared libraries: libsycl.so.9: cannot open shared object file
```

The binary linked fine but the loader cannot find the SYCL runtime, which lives in
the toolchain prefix rather than a system directory. This appears the moment a binary
runs outside the shell where the toolchain was activated — most often under
`ctest`, a build script, or CI, which do not inherit your interactive environment.

**conda route — bake in an rpath:**

```bash
icpx -fsycl ... -Wl,-rpath,"$CONDA_PREFIX/lib" prog.cpp -o prog
```

In CMake, add it to the target so tests inherit it:

```cmake
target_link_options(mytarget PRIVATE "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
```

**apt/oneAPI route — source the environment in the shell that runs the binary:**

```bash
source /opt/intel/oneapi/setvars.sh
```

`LD_LIBRARY_PATH="$CONDA_PREFIX/lib"` also works as a stopgap, but it is fragile
precisely because harnesses drop it. Prefer the rpath.

## Runtime: No device of requested type available

```
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  No device of requested type available.
```

This is a *different* problem from the previous one and the fix is different — the
libraries loaded fine, but no SYCL device was discovered. Device discovery goes
through OpenCL ICD registration, which is environment state, not a link-time
property.

The distinction matters most on the apt/oneAPI route, where the dependency is wider
than it looks. Measured on a 2026.1 install, adding directories one at a time:

| `LD_LIBRARY_PATH` contains | result |
|---|---|
| compiler `lib` only | fails |
| + `tbb` | still fails |
| + `tcm`, `umf` (the full setvars set, 7 entries) | **works** |

So it is possible without sourcing — but each of those paths carries a component
version number that changes on every toolkit update, so anything hard-coding them
breaks silently at the next upgrade. `OCL_ICD_FILENAMES` alone does not help, since
`libintelocl.so` itself needs `libtbb.so.12` resolvable. Source the environment
instead of reconstructing it:

```bash
source /opt/intel/oneapi/setvars.sh      # then run the binary in that shell
```

If binaries must run without a sourced environment, use the conda route instead —
its binaries are relocatable with a single rpath flag. That difference is the main
practical reason to pick one route over the other.

Also check the obvious: `sycl-ls` with no output means no device at all is
registered, which is a broken install rather than an environment problem.

## Build: omp.h file not found

```
fatal error: 'omp.h' file not found
```

The DPC++ compiler does not bundle the OpenMP headers on the conda route.

```bash
conda install -n sycl -c conda-forge llvm-openmp
icpx -fsycl -qopenmp ...        # note -qopenmp, not -fopenmp
```

## Build: reference to 'info' is ambiguous

```
error: reference to 'info' is ambiguous
note: candidate found by name lookup is 'sycl::info'
note: candidate found by name lookup is 'sycl::ext::intel::esimd::info'
```

Caused by combining `using namespace sycl;` with `using namespace
sycl::ext::intel::esimd;` — both namespaces define `info`. Qualify the use:

```cpp
q.get_device().get_info<sycl::info::device::name>()   // not info::device::name
```

Prefer a namespace alias over a `using namespace` for esimd:
`namespace esimd = sycl::ext::intel::esimd;`

## Install: conda exists but has no conda binary

An interrupted Miniconda install leaves a prefix directory that later runs treat as
"already installed", so they skip the bootstrap and then fail on a missing `conda`.
Detect by testing for the binary, not the directory, and start clean:

```bash
[ -x /opt/conda/bin/conda ] || { rm -rf /opt/conda; bash miniconda.sh -b -p /opt/conda; }
```

## Install: apt install intel-oneapi-toolkit has no candidate

There is no package by that name. The real ones are component packages:

| Want | Package |
|---|---|
| DPC++/SYCL compiler | `intel-oneapi-compiler-dpcpp-cpp` |
| MKL (Intel repo) | `intel-oneapi-mkl-devel` |
| MKL (Ubuntu archive, smaller) | `libmkl-dev` |

The Intel packages need Intel's apt repo added first (key + sources list; the
bundled installer does this). `libmkl-dev` comes from Ubuntu's own archive and needs
no extra repo — a useful fallback when Intel's servers are blocked.

## Gone: icpx command not found on a machine that had it

The container was recycled and `/opt/conda` or `/opt/intel` went with it. This is
expected behaviour in ephemeral environments, not corruption. Re-run the installer;
it takes a few minutes and restores the same environment. Do not spend time
diagnosing it.

## Device aspect probe

Rather than guessing whether a device supports a feature, ask it. This also prints
whether you are on a CPU or GPU device, which determines what your results mean.

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
    for (auto &d : sycl::device::get_devices()) {
        std::printf("%-45s cpu=%d gpu=%d  esimd=%s  fp64=%s\n",
            d.get_info<sycl::info::device::name>().c_str(),
            d.is_cpu(), d.is_gpu(),
            d.has(sycl::aspect::ext_intel_esimd) ? "YES" : "no",
            d.has(sycl::aspect::fp64) ? "YES" : "no");
        auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
        std::printf("   sub-group sizes:"); for (auto s : sg) std::printf(" %zu", s);
        std::printf("\n   local_mem_size: %zu B\n",
            (size_t)d.get_info<sycl::info::device::local_mem_size>());
    }
}
```

Sub-group sizes matter because `[[sycl::reqd_sub_group_size(N)]]` with an
unsupported `N` is a hard launch failure, not a slow fallback — check before
selecting a width.

## SessionStart hook pattern

To reprovision automatically when a container starts, use a `SessionStart` hook.
Two properties make it cheap: `async` so it does not block session startup, and an
early-exit idempotency check so an already-provisioned container costs nothing.

`.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh" } ] }
    ]
  }
}
```

`.claude/hooks/session-start.sh`:

```bash
#!/bin/bash
set -euo pipefail
echo '{"async": true, "asyncTimeout": 600000}'   # don't block startup

# Only manage dependencies in the remote container; leave local machines alone.
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

# Idempotency check: test for a specific artifact, not a directory.
if [ -f /usr/include/mkl/mkl_compact.h ]; then
  echo "Intel MKL already present; nothing to do."; exit 0
fi

export DEBIAN_FRONTEND=noninteractive
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends libmkl-dev
```
