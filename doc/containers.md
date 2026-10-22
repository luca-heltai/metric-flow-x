# Run MetricFlowX in a container

The container workflow builds an OCI image containing the Release `blood_flow` executable, the compiled
MetricFlowX shared library, its headers and CMake package, and the matching
deal.II SDK. It targets Linux AMD64 and ARM64. Docker, Podman, and Apptainer use
the same image. A container runtime must already be available; MetricFlowX,
deal.II, a compiler, and their dependencies need no host installation.

The image intentionally retains the complete deal.II SDK and its dependencies.
Expect a large initial download and roughly 12 GB of local image storage per
architecture. Dataset downloads and simulation output require additional space.

## Standalone launcher

Download the launcher from a versioned GitHub release when one is available.
The release launcher has the immutable image digest embedded. After this infrastructure is merged, the development
launcher can be downloaded without cloning the repository:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/luca-heltai/metric-flow-x/main/scripts/run_container.sh \
  --output run_container.sh
bash run_container.sh --help
```

Put inputs in a working directory, including the materialized parameter file,
mesh, and optional inflow CSV. The launcher mounts the working directory at `/work` and at its original
absolute host path, runs in `/work`, and keeps outputs in that
same host directory with the invoking user's ownership:

```bash
bash run_container.sh --workdir /path/to/case parameters.prm
bash run_container.sh --runtime podman --workdir /path/to/case parameters.prm
bash run_container.sh --runtime apptainer --workdir /path/to/case parameters.prm
```

Runtime detection checks Docker, then Podman, then Apptainer. Use `--runtime`
when several are installed. It checks executable availability, so select a
working alternative explicitly if a Docker daemon is unavailable.

Every file referenced by a parameter file must be accessible below the mounted
working directory. Absolute paths below that directory work through the second mount; paths
outside it are unavailable. Choose a common parent directory when datasets
and outputs live in different subdirectories. Unexpanded `.prm.in` templates
must be materialized before execution. Outputs are not copied
out of the image: they are written directly into the mounted host directory.
Dataset acquisition and parameter materialization are separate steps.

For reproducible runs, select the paper's recorded image digest and parameter
and dataset snapshots:

```bash
bash run_container.sh --image ghcr.io/luca-heltai/metric-flow-x@sha256:DIGEST \
  --workdir /path/to/case parameters.prm
```

`DIGEST` must be replaced with a published image digest. `main` becomes available after a successful publication from `main`;
`:main` is the moving development image; `:sha-COMMIT` identifies a tested source revision. Release
tags such as `:vX.Y.Z` are published when the corresponding Git tag is pushed.
Record the digest for archival reproduction because registry tags can move.
`METRIC_FLOW_X_IMAGE` provides an alternative default image reference.

## MPI and SDK access

MPI runs inside the container, on one node, with its bundled MPI installation:

```bash
bash run_container.sh --ranks 2 --workdir /path/to/case parameters.prm
```

This launcher does not configure multinode Slurm or an external host MPI.
Those require a compatible MPI/PMIx setup and separate validation. The chosen
parameter file and linear solver must support the requested number of ranks.

The library and SDK are installed under `/opt/metric-flow-x`. To open a shell,
configure a downstream CMake application, or inspect dependencies:

```bash
bash run_container.sh --exec bash
bash run_container.sh --workdir /path/to/consumer --exec cmake \
  -S . -B build -DCMAKE_BUILD_TYPE=Release
```

The package is found through `CMAKE_PREFIX_PATH`; consumers link
`MetricFlowX::metric_flow_x`. This image installs the Release variant only.
The historical `scripts/run_docker.sh` delegates to this launcher with Docker
selected and no longer parses a removed GitLab configuration.

## Publication and verification

The container workflow builds natively on GitHub's AMD64 and ARM64 runners.
For each architecture it runs the installed executable with one and two MPI
ranks using the inflow integration smoke test, compiles a downstream library
consumer, and checks launcher output ownership. AMD64 additionally checks
Podman and conversion to Apptainer SIF. These checks establish infrastructure
operation; they do not validate the paper's scientific results.

Pull requests build and test without publishing images. Pushes to `main`,
version tags, `container-*` preview tags, and manual workflow runs publish tested architecture images and
then their combined OCI index. Manual runs on a feature branch publish only its
commit tag; `container-*` tags publish an explicitly named preview image. A workflow artifact records the digest and a launcher pinned to it;
version tags also attach these files to an existing release or create a draft
release. The workflow does not merge a pull request or publish a draft release.

GHCR creates a new package as private even when its repository is public. Set
this package's visibility to public in its GitHub package settings. Publication
checks anonymous index access and fails clearly until that setting is applied.
See [GitHub's container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

The Dockerfile pins the multi-architecture deal.II 9.7.1 base by digest. Rebuilds
reuse that base; they do not copy binaries from a developer's build directory.

```bash
docker build --build-arg BUILD_JOBS=2 -t metric-flow-x:local .
bash scripts/run_container.sh --image metric-flow-x:local --workdir /path/to/case parameters.prm
```
