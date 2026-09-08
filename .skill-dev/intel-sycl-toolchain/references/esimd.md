# Intel ESIMD (Explicit SIMD)

`sycl::ext::intel::esimd` is an Intel-only SYCL extension for explicitly vectorized
Intel GPU code. Instead of the SIMT model — a scalar kernel per work-item that the
compiler vectorizes across a sub-group — you write vector code directly:
`simd<double, 8>` puts the width in the *type*, and one work-item owns the whole
vector. ESIMD kernels always run at sub-group size 1; the compiler never vectorizes
across work-items.

Useful when you want GNU-vector-style ergonomics (`vector_size` attribute) on Xe, or
predictable block load/store codegen rather than trusting the vectorizer.

## The thing to know first: it does not run on CPU

ESIMD **compiles anywhere** — the host compile never checks whether a suitable
device exists — but it **runs only on Intel GPUs**. In a CPU-only container every
ESIMD kernel fails at submission. Budget for this: you can develop, compile-check and
review ESIMD code in a sandbox, but you cannot validate numerics or measure
performance without Xe hardware. Say so when reporting results.

Two distinct failure modes, both meaning the same thing:

```
SYCL exception: Required aspect ext_intel_esimd is not supported on the device
```

```
Build program log ... Compilation failed
Unrecognized build options: -vc-codegen -disable-finalizer-msg
```

The first is the runtime's aspect check. The second appears when the check is
bypassed and the OpenCL CPU compiler is handed the Vector-Compute backend flags that
the ESIMD path injects. Neither is fixable in software.

Check before running rather than catching an exception:

```cpp
if (!q.get_device().has(sycl::aspect::ext_intel_esimd)) { /* skip / report */ }
```

## Minimal kernel

```cpp
#include <sycl/ext/intel/esimd.hpp>          // the ONLY required include
namespace esimd = sycl::ext::intel::esimd;

q.parallel_for(sycl::range<1>(n / VL), [=](sycl::id<1> i)
                                           [[intel::sycl_explicit_simd]] {
    auto off = i * VL;
    esimd::simd<double, VL> a(A + off), b(B + off);
    (a + b).copy_to(C + off);
});
```

`<sycl/ext/intel/esimd.hpp>` alone is enough for device functions —
`<sycl/sycl.hpp>` is needed only in the translation unit that creates the queue and
launches. Note where the attributes go: the **kernel** carries
`[[intel::sycl_explicit_simd]]`, a **device function** carries `SYCL_ESIMD_FUNCTION`.

## Memory handles: one body, several address spaces

`block_load` / `block_store` are overloaded on the memory handle with a uniform
`(handle, byte_offset, props)` shape:

```cpp
block_load<T,N>(const T *usm_ptr,    size_t   byte_offset, props)
block_load<T,N>(accessor device_acc, uint64_t byte_offset, props)
block_load<T,N>(local_accessor lacc, uint32_t byte_offset, props)   // SLM
```

Because the call spelling is identical, templating a function on the handle type
lets **one body serve global memory and SLM** — no policy struct and no duplicated
algorithm:

```cpp
template <typename T, int V, typename H>
SYCL_ESIMD_FUNCTION void kernel_body(H h, std::size_t off, ...) {
    auto v = esimd::block_load<T, V>(h, off);   // resolves by handle type
}
```

A raw `T*` reaches global/USM only. SLM is *not* pointer-addressable in ESIMD: use
the `local_accessor` overload above, or the lower-level `slm_init<Size>()` +
`slm_block_load<T,N>(uint32_t byte_offset)` API. A pointer obtained from SLM will not
work — the pointer overload lowers to a stateless/A64 global message, which cannot
reach SLM.

**Alignment hazard.** From the header: passing a `byte_offset` not aligned to 16
bytes without declaring the actual alignment in `props` produces *incorrect* store
results on Gen12 — silently wrong values, not an error. A `simd<double,8>` block is
64 bytes so naturally-strided offsets are fine; narrower types or smaller widths need
an explicit `esimd::alignment<…>` property.

## Wrapping the verbosity

Raw `block_load`/`block_store` calls bury the algorithm. A small view struct fusing
`{handle, byte offset, byte strides}` restores readable indexing and costs nothing:

```cpp
template <typename T, int V, typename H> struct block_view {
    using vec = esimd::simd<T, V>;
    H h; std::size_t off, istep, jstep;          // byte offset and byte strides
    std::size_t at(int i, int j) const { return off + i * istep + j * jstep; }
    vec operator()(int i, int j) const { return esimd::block_load<T, V>(h, at(i, j)); }
    void store(int i, int j, const vec &v) const { esimd::block_store<T,V>(h, at(i,j), v); }
};
```

Have `operator()` return the **loaded value**, not a proxy reference. This is not a
style preference: esimd's arithmetic operators are SFINAE-constrained templates, and
template argument deduction never applies user-defined conversions, so a proxy type
cannot participate in `a * b` — every read would need an explicit cast. Returning the
value keeps the hot loop clean:

```cpp
s -= A(i, l) * B(l, j);          // works
s -= A.ref(i, l) * B(l, j);      // does NOT compile: proxy in arithmetic
```

If you also want assignment syntax on stores, add a separate `ref(i,j)` returning a
proxy with `operator=`. Its copy-assignment must be declared explicitly — otherwise
the implicit one copies the handle and offset, **rebinding the proxy instead of
storing the value**, so `A.ref(1,0) = A.ref(0,0)` silently writes nothing.

## Cross-translation-unit ESIMD

Direct calls across translation units work with no special flags. The header carries
declarations only:

```cpp
// header
SYCL_EXTERNAL void f(double a, const double *x) SYCL_ESIMD_FUNCTION;

// .cpp: definition, plus one explicit instantiation per <T,V> if templated
template void f_t<double, 8>(double, const double *);
```

`SYCL_EXTERNAL` gives external *device* linkage; `SYCL_ESIMD_FUNCTION` marks it
ESIMD. Compile each TU with `icpx -fsycl -c` and link normally.

**Templates must be explicitly instantiated** in the defining TU. Without it the
device linker emits only a warning and still produces a binary:

```
warning: Undefined function _Z3f_tIdLi8EE... found in ....bc.
         This may result in runtime errors.
```

which becomes a runtime failure later. Treat that warning as an error.

## invoke_simd

`invoke_simd` bridges a SIMT SYCL kernel to an ESIMD function so the sub-group is
passed as a vector — the way to keep an ordinary SPMD launch while writing the body
vector-style. Unlike direct calls it needs extra flags at both compile and run time:

```bash
icpx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr prog.cpp
IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 ./prog
```

The callee is `[[intel::device_indirectly_callable]] SYCL_EXTERNAL ... __regcall ...
SYCL_ESIMD_FUNCTION`, and the sub-group size must equal the `simd` width. It remains
experimental — prefer a direct ESIMD kernel unless you specifically need the SPMD
outer launch.

## SLM notes

SLM is a **work-group** resource, so an SLM kernel needs `nd_range`; a plain `range`
launch has no work-groups and no SLM. Size it with a `local_accessor` in the command
group, or `slm_init<Bytes>()` inside an ESIMD kernel.

Check capacity on the host and fail with a clear message rather than at launch:

```cpp
auto cap = q.get_device().get_info<sycl::info::device::local_mem_size>();
```

Typical Xe budgets are 64–128 KB per work-group. If each work-item takes a large
slice, few work-groups stay resident per Xe-core and occupancy collapses — an
SLM-resident design can be *slower* than leaving data in global memory and relying on
cache. The regime where it wins is small per-work-item footprints, where several
work-items still fit. Measure; do not assume.

When work-items own disjoint SLM slices, no `barrier()` is needed — a barrier is only
required when they share data.
