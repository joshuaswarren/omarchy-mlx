// Host test harness for overlay/mlx/backend/omarchy/gpu_profiler.h: drives the profiler through the same hook sequence encoder.cpp uses (begin, per-dispatch
// before/after, end of command buffer, submit end, join) against a FAKE Vulkan device table, and prints what it asked the "GPU" to do as one JSON line on stdout.
//   harness <submits> <dispatches_per_submit> [rejoin]
// Environment (read by the profiler): MLX_OMARCHY_GPU_PROFILE=<file>, MLX_OMARCHY_GPU_PROFILE_MODE=cb|dispatch.
// The fake GetQueryPoolResults returns 1000 + 100 * query_index for each query, so a record's t0/t1 can be checked exactly.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mlx/backend/omarchy/gpu_profiler.h"

namespace mlx::core::omarchy::vk {

namespace {
struct Counts {
  int timestamps{0};
  int barriers{0};
  int resets{0};
  std::vector<uint32_t> ts_queries;
} g;

VKAPI_ATTR VkResult VKAPI_CALL fake_create_pool(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*, VkQueryPool* pool) {
  static uintptr_t next = 0x100;
  *pool = reinterpret_cast<VkQueryPool>(next++);
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_reset(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {
  g.resets++;
}
VKAPI_ATTR void VKAPI_CALL fake_write_ts(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t query) {
  g.timestamps++;
  g.ts_queries.push_back(query);
}
VKAPI_ATTR void VKAPI_CALL fake_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t,
    const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
  g.barriers++;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_results(VkDevice, VkQueryPool, uint32_t first, uint32_t count, size_t, void* data, VkDeviceSize, VkQueryResultFlags) {
  auto* out = static_cast<uint64_t*>(data);
  for (uint32_t i = 0; i < count; ++i) {
    out[i] = 1000 + 100ull * (first + i);
  }
  return VK_SUCCESS;
}
} // namespace

DeviceTable& device_table() {
  static DeviceTable t = [] {
    DeviceTable x;
    x.CreateQueryPool = fake_create_pool;
    x.CmdResetQueryPool = fake_reset;
    x.CmdWriteTimestamp = fake_write_ts;
    x.CmdPipelineBarrier = fake_barrier;
    x.GetQueryPoolResults = fake_results;
    return x;
  }();
  return t;
}

const char* result_string(VkResult) {
  return "fake";
}

int timestamps() { return g.timestamps; }
int barriers() { return g.barriers; }
int resets() { return g.resets; }
const std::vector<uint32_t>& ts_queries() { return g.ts_queries; }

} // namespace mlx::core::omarchy::vk

int main(int argc, char** argv) {
  using namespace mlx::core::omarchy;
  const int submits = argc > 1 ? std::atoi(argv[1]) : 1;
  const int dispatches = argc > 2 ? std::atoi(argv[2]) : 3;
  int owner = 0;
  Device device;
  auto& p = prof::get();
  p.attach(&owner, device);
  ComputeParams params;
  VkBuffer buf = reinterpret_cast<VkBuffer>(0x10);
  ComputeBinding binding{buf, 0, 64};
  VkCommandBuffer cmd = reinterpret_cast<VkCommandBuffer>(0x20);
  for (int s = 1; s <= submits; ++s) {
    p.on_begin(&owner, 0, cmd, 5);
    for (int d = 0; d < dispatches; ++d) {
      p.before_dispatch(&owner, 0, cmd, true);
      p.after_dispatch(&owner, 0, cmd, static_cast<ComputeKernel>(0), params, std::span<const ComputeBinding>(&binding, 1), 1, 1, 1, 7, 0);
    }
    p.before_end(&owner, 0, cmd);
    p.on_submit_end(&owner, static_cast<uint64_t>(s), 10, 0);
  }
  p.on_join(&owner, static_cast<uint64_t>(submits), 100, 200, 300, "test");
  if (argc > 3 && std::string(argv[3]) == "rejoin") {
    p.on_join(&owner, static_cast<uint64_t>(submits), 400, 500, 600, "test");   // a second join must not emit anything again
  }
  std::printf("{\"timestamps\":%d,\"barriers\":%d,\"resets\":%d,\"ts_queries\":[", vk::timestamps(), vk::barriers(), vk::resets());
  for (size_t i = 0; i < vk::ts_queries().size(); ++i) {
    std::printf("%s%u", i ? "," : "", vk::ts_queries()[i]);
  }
  std::printf("]}\n");
  return 0;
}
