// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Fence on top of the event implementation. The events are real timeline
// semaphores on GPU streams, so fences carry real device ordering.

#include "mlx/fence.h"

#include "mlx/event.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  Event event;

  FenceImpl(uint32_t count, Stream s) : count(count), event(s) {}
};

Fence::Fence(Stream s) {
  fence_ = std::make_shared<FenceImpl>(0, s);
}

void Fence::wait(Stream s, const array&, uint32_t) {
  // The events are timeline semaphores; waiting at the event's current
  // value covers every update() this fence published before the wait was
  // enqueued, which is a superset of the requested `value`.
  cast<FenceImpl>().event.wait(s);
}

uint32_t Fence::update(Stream s, const array&, bool) {
  auto& f = cast<FenceImpl>();
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr, "[rtmod] FENCE-UPDATE count=%u st=%d\n", f.count, s.index);
  }
  f.count++;
  f.event.set_value(f.count);
  f.event.signal(s);
  return f.count;
}

} // namespace mlx::core
