# Download

`cqr` is distributed as **source**. Released versions are published as
[GitHub Releases](https://github.com/ivan-pi/cqr/releases), each with an
automatically generated source tarball.

[All releases](https://github.com/ivan-pi/cqr/releases){ .md-button .md-button--primary }
[Latest release](https://github.com/ivan-pi/cqr/releases/latest){ .md-button }

## Release tarballs

Every tagged release has a stable source-archive URL. For a tag `vX.Y.Z`:

```text
https://github.com/ivan-pi/cqr/archive/refs/tags/vX.Y.Z.tar.gz   # .tar.gz
https://github.com/ivan-pi/cqr/archive/refs/tags/vX.Y.Z.zip      # .zip
```

The [latest-release page](https://github.com/ivan-pi/cqr/releases/latest) always
redirects to the newest version and links its **Source code (tar.gz / zip)**
assets, so you can point a script or a package recipe at it without hard-coding
a version.

!!! note "No tagged release yet"
    `cqr` is at `0.1.0` and has not been tagged yet, so the Releases page is
    currently empty. Until the first release, fetch the development sources
    directly (below). Once `v0.1.0` is tagged, its tarball appears at the URL
    pattern above.

## Build from a tarball

```sh
curl -fsSL https://github.com/ivan-pi/cqr/archive/refs/tags/v0.1.0.tar.gz | tar xz
cd cqr-0.1.0
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

See the [User guide](guide.md) for prerequisites, performance flags, and
MKL-free builds.

## Development sources

To track `main` instead of a release:

```sh
git clone https://github.com/ivan-pi/cqr.git
cd cqr
```

## Verifying the source

GitHub's source archives are generated on demand; if you need to pin an exact
tree, record the **release tag** (or the commit it points to) rather than a
checksum of the tarball, since the archive bytes are not guaranteed to be
byte-stable over time. Cloning the tag and checking out its commit gives a
reproducible tree:

```sh
git clone --branch v0.1.0 --depth 1 https://github.com/ivan-pi/cqr.git
```
