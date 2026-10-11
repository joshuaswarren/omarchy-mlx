# Install

Back to the [README](../README.md). The one-line script is in the README.

## The Omarchy package

The packages are published in the `[omarchy]` pacman repository at
`https://pkgs.omarchy.org/edge/$arch`. A fresh install from the Omarchy
Mac installer writes that repository into `/etc/pacman.conf`. A build
that does not list it cannot find the packages, and they are not in the
AUR. On such a build, use the script below. As of 2026-10-11 the
repository serves 0.7.28-2, and 0.7.32 waits on
[omarchy-pkgs#839](https://github.com/omacom/omarchy-pkgs/pull/839).

Where the repository is configured, `omarchy-mac-ml` is the meta
package. It pulls in `omarchy-mlx` (the runtime, as a system venv under
`/usr/lib/omarchy-mlx` with launchers in `/usr/bin`),
`omarchy-mlx-vulkan` (the private Honeykrisp ICD under
`/usr/lib/omarchy-mlx/vulkan/`), and the ANE packages. The Vulkan
package leaves the system Mesa driver installed for the desktop. The
system install builds offline from the vendored, hash-locked wheel set
attached to each release.

The Omarchy menu row for this package, Install > AI > MLX + Core ML
(Apple Silicon), is proposed in
[omacom/omarchy#14805](https://github.com/omacom/omarchy/pull/14805) and
may not be in your build. The Mac platform repository's menu has the row
Install > AI > MLX (Apple GPU), added in
[omacom/omarchy-mac#385](https://github.com/omacom/omarchy-mac/pull/385)
(merged 2026-09-12). It runs the script below and does not install the
private driver. A build from another fork may carry neither row.

## Install with the script

On an Arch install for an M1-class Mac or an M2 Max, the script sets
up a private venv under `~/.local/share/mlx-omarchy` and puts the
launchers in `~/.local/bin`. It registers MLX Chat in the launcher
menu. It installs `lapack`, `blas`, and `openblas`. It leaves the
system Mesa driver installed and does not install `omarchy-mlx-vulkan`.
The runtime then uses the system Honeykrisp driver. Use Mesa 26.2.4 or
newer. Older Mesa can return wrong results from int8 and other
shift-heavy kernels, and the runtime prints a warning. The known
differences from the packaged driver are in
[docs/known-defects.md](known-defects.md). To use the private ICD,
install `omarchy-mlx-vulkan` from the `[omarchy]` repository.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/main/install.sh | bash
```

The flags: `--ane` sets up ANE device ownership. Use `--voice` for the
optional speech packages. Use `--uninstall` to remove it. The script
picks the latest release, checks the wheel against the release
`SHA256SUMS`, and asks for Python 3.14 on aarch64.

## From a release wheel

Releases attach wheels named
`mlx_omarchy-<version>-cp314-cp314-linux_aarch64.whl` for Apple
Silicon and `mlx_omarchy-<version>-cp311-cp311-linux_x86_64.whl` for
x86_64 dev boxes, plus a `SHA256SUMS` covering every asset. Example for
v0.7.10:

```bash
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install "https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.10/mlx_omarchy-0.32.4.dev202610012048+6cff5ea-cp314-cp314-linux_aarch64.whl"
~/.venvs/mlx/bin/pip install --no-deps mlx-lm==0.31.3
```

The `--no-deps` on `mlx-lm` matters. It depends on upstream `mlx`,
which would clobber this wheel. On x86_64 there is no ANE. You need a
software Vulkan driver plus `MLX_OMARCHY_ALLOW_NON_APPLE=1` to import
the module.

The wheels ship GPU kernels, not the driver. The Omarchy package path
brings the Honeykrisp driver with it (`omarchy-mlx-vulkan`, a private
ICD under `/usr/lib/omarchy-mlx/vulkan/`). A script or wheel install
needs that package. It leaves the system Mesa driver installed. See
[docs/install-omarchy.md](install-omarchy.md).

The demo video is 2:47, unedited. It runs from the one-line install to
a streamed answer on an M1.

[demo](https://github.com/user-attachments/assets/7b2326f0-4679-4784-9622-e403b99be853)
