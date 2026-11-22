# Vendored GCC runtime licence texts

These texts are copied into every Linux wheel that vendors a GCC runtime
library (`libgomp`, `libgfortran`, `libgcc_s`). Redistributing those
libraries without their licence text is not allowed, so the texts travel
with the wheel instead of being looked up in the build host's system
licence directory.

| File | Text | Source |
|---|---|---|
| `COPYING3` | GNU General Public License, version 3 | GCC source distribution (`COPYING3`) |
| `COPYING.RUNTIME` | GCC Runtime Library Exception, version 3.1 | GCC source distribution (`COPYING.RUNTIME`) |

`COPYING.LIB` (needed when `libquadmath` is in the runtime closure) is not
vendored here: its text depends on the GCC version that produced the
library. A build that needs it must point `MLX_OMARCHY_GCC_LICENSE_DIR` at
a directory holding the exact text for its toolchain; the build then fails
with the file name it could not find rather than shipping a wheel without
the licence.
