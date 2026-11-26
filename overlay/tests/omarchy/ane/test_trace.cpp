// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/backend/omarchy/trace.h"

using mlx::core::omarchy::trace::ane_trace_snapshot;
using mlx::core::omarchy::trace::counters;
using mlx::core::omarchy::trace::record_ane_completion;
using mlx::core::omarchy::trace::record_ane_model_loaded;
using mlx::core::omarchy::trace::record_ane_package_cache_hit;
using mlx::core::omarchy::trace::record_ane_package_compiled;
using mlx::core::omarchy::trace::record_ane_submission;
using mlx::core::omarchy::trace::record_ane_timeout;
using mlx::core::omarchy::trace::record_ane_worker_start;

namespace {

void reset_ane_trace() {
  auto& values = counters();
  values.ane_models_loaded.store(0, std::memory_order_relaxed);
  values.ane_packages_compiled.store(0, std::memory_order_relaxed);
  values.ane_package_cache_hits.store(0, std::memory_order_relaxed);
  values.ane_worker_starts.store(0, std::memory_order_relaxed);
  values.ane_submissions.store(0, std::memory_order_relaxed);
  values.ane_timeouts.store(0, std::memory_order_relaxed);
  values.ane_input_bytes.store(0, std::memory_order_relaxed);
  values.ane_output_bytes.store(0, std::memory_order_relaxed);
  values.ane_exec_ns.store(0, std::memory_order_relaxed);
}

void record_fixture_events() {
  record_ane_model_loaded();
  record_ane_package_compiled();
  record_ane_package_cache_hit();
  record_ane_worker_start();
  record_ane_submission(128);
  record_ane_submission(256);
  record_ane_completion(512, 1'000);
  record_ane_timeout(250);
}

} // namespace

TEST_CASE("ANE trace events obey the compile-time gate") {
  reset_ane_trace();
  record_fixture_events();
  auto snapshot = ane_trace_snapshot();
#ifdef MLX_OMARCHY_ANE_TRACING
  CHECK(snapshot.ane_models_loaded == 1);
  CHECK(snapshot.ane_packages_compiled == 1);
  CHECK(snapshot.ane_package_cache_hits == 1);
  CHECK(snapshot.ane_worker_starts == 1);
  CHECK(snapshot.ane_submissions == 2);
  CHECK(snapshot.ane_timeouts == 1);
  CHECK(snapshot.ane_input_bytes == 384);
  CHECK(snapshot.ane_output_bytes == 512);
  CHECK(snapshot.ane_exec_ns == 1'250);
#else
  CHECK(snapshot.ane_models_loaded == 0);
  CHECK(snapshot.ane_packages_compiled == 0);
  CHECK(snapshot.ane_package_cache_hits == 0);
  CHECK(snapshot.ane_worker_starts == 0);
  CHECK(snapshot.ane_submissions == 0);
  CHECK(snapshot.ane_timeouts == 0);
  CHECK(snapshot.ane_input_bytes == 0);
  CHECK(snapshot.ane_output_bytes == 0);
  CHECK(snapshot.ane_exec_ns == 0);
#endif
}
