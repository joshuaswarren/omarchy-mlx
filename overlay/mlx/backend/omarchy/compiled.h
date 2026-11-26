// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// GPU tape interpreter for the Compiled primitive. Walks the tape in order
// and runs each node through its own primitive GPU path into a fresh
// temporary; intermediate buffers stay alive through encoder temporaries
// until the submission completes. Every class upstream fuses has a GPU
// path, and each refuses its own unsupported dtypes by name. There is no
// CPU fallback in Omarchy builds, and no fusion speedup is claimed: each
// node dispatches separately.
void eval_compiled_tape(
    const std::vector<array>& tape,
    const std::vector<array>& tape_inputs,
    const std::vector<array>& tape_outputs,
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const Stream& stream);

} // namespace mlx::core::omarchy
