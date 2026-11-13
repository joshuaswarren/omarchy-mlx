// Frame-pacing proxy for issue #19: a 60 Hz "compositor stand-in".
//
// Submits one tiny command buffer (a 256-byte fill) every ~16.6 ms and
// measures, per submission, the host latency from just before
// vkQueueSubmit to the fence signaling. A desktop compositor experiences
// exactly this latency for its own small per-frame submissions when an
// MLX workload shares the GPU: while a long MLX submission holds the
// queue, this latency spikes to that submission's remaining length.
//
// Output: one line per run with n, p50, p99, max, mean (ms), plus the
// count of submissions whose latency exceeded 2x the frame budget
// (33.3 ms at 60 Hz; the ">2x-refresh" metric from the issue).
//
// Build: cc -O2 -o submit-latency submit-latency.c $(pkg-config --libs vulkan)
// Run:   ./submit-latency [seconds] [--rate 60]

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char** argv) {
  double seconds = 10.0;
  double rate_hz = 60.0;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--rate") == 0 && i + 1 < argc) {
      rate_hz = atof(argv[++i]);
    } else {
      seconds = atof(argv[i]);
    }
  }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "submit-latency";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance instance;
  if (vkCreateInstance(&ici, NULL, &instance) != VK_SUCCESS) {
    fprintf(stderr, "vkCreateInstance failed\n");
    return 2;
  }
  uint32_t ndev = 0;
  VkResult enum_res = vkEnumeratePhysicalDevices(instance, &ndev, NULL);
  if (enum_res != VK_SUCCESS || ndev == 0) {
    fprintf(stderr, "no physical devices res=%d ndev=%u\n",
            (int)enum_res, ndev);
    return 2;
  }
  VkPhysicalDevice* devs = (VkPhysicalDevice*)calloc(ndev, sizeof(*devs));
  enum_res = vkEnumeratePhysicalDevices(instance, &ndev, devs);
  if (enum_res != VK_SUCCESS && enum_res != VK_INCOMPLETE) {
    fprintf(stderr, "enumerate failed res=%d\n", (int)enum_res);
    return 2;
  }
  // Pick the Apple (Honeykrisp) device: the same GPU the MLX decode
  // runs on; llvmpipe would measure a CPU queue, not the shared GPU.
  VkPhysicalDevice pd = VK_NULL_HANDLE;
  for (uint32_t i = 0; i < ndev; ++i) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(devs[i], &props);
    fprintf(stderr, "device[%u]: %s\n", i, props.deviceName);
    if (strstr(props.deviceName, "Apple") != NULL) {
      pd = devs[i];
    }
  }
  if (pd == VK_NULL_HANDLE) {
    pd = devs[0];
  }
  uint32_t qfam = 0, nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
  VkQueueFamilyProperties* qp =
      (VkQueueFamilyProperties*)calloc(nq, sizeof(*qp));
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
  for (uint32_t i = 0; i < nq; ++i) {
    if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfam = i;
      break;
    }
  }
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) {
    fprintf(stderr, "vkCreateDevice failed\n");
    return 2;
  }
  VkQueue queue;
  vkGetDeviceQueue(dev, qfam, 0, &queue);

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = 256;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer buffer;
  vkCreateBuffer(dev, &bci, NULL, &buffer);
  VkDeviceMemory mem;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buffer, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if ((mr.memoryTypeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      mai.memoryTypeIndex = i;
      break;
    }
  }
  vkAllocateMemory(dev, &mai, NULL, &mem);
  vkBindBufferMemory(dev, buffer, mem, 0);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = qfam;
  VkCommandPool pool;
  vkCreateCommandPool(dev, &pci, NULL, &pool);
  VkCommandBufferAllocateInfo cai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(dev, &cai, &cmd);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  vkCmdFillBuffer(cmd, buffer, 0, 256, 0x01010101u);
  vkEndCommandBuffer(cmd);

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(dev, &fci, NULL, &fence);

  size_t cap = (size_t)(seconds * rate_hz) + 16;
  double* lat = (double*)calloc(cap, sizeof(double));
  size_t n = 0;
  double frame_ms = 1000.0 / rate_hz;
  double next = now_ms();
  double end = next + seconds * 1000.0;
  while (n < cap && (next = now_ms()) < end) {
    vkResetFences(dev, 1, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    double t0 = now_ms();
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    lat[n++] = now_ms() - t0;
    // pace to the frame budget
    double sleep_until = t0 + frame_ms;
    double now;
    while ((now = now_ms()) < sleep_until) {
      struct timespec ts = {0, (long)((sleep_until - now) * 1e6)};
      nanosleep(&ts, NULL);
    }
  }

  qsort(lat, n, sizeof(double), cmp_double);
  double p50 = lat[n / 2], p99 = lat[(size_t)(n * 0.99)], max = lat[n - 1];
  double sum = 0;
  for (size_t i = 0; i < n; ++i) sum += lat[i];
  size_t over2x = 0;
  for (size_t i = 0; i < n; ++i) {
    if (lat[i] > 2.0 * frame_ms) over2x++;
  }
  printf(
      "submit-latency: n=%zu p50=%.2fms p99=%.2fms max=%.2fms mean=%.2fms "
      "over2x(>%.1fms)=%zu/%zu\n",
      n, p50, p99, max, sum / n, 2.0 * frame_ms, over2x, n);

  vkDestroyFence(dev, fence, NULL);
  vkDestroyCommandPool(dev, pool, NULL);
  vkDestroyBuffer(dev, buffer, NULL);
  vkFreeMemory(dev, mem, NULL);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(instance, NULL);
  free(lat);
  free(qp);
  return 0;
}
