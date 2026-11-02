// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// H16 load sections from an HWX; the header gives the overview, the
// comments below the byte layouts.
//
// Every value is read through a bounds-checked view of the file. A field
// that would lie past the end of the file is a refusal, never a read out
// of bounds. Names are NUL-terminated strings inside their field or
// command and must be valid UTF-8.

#include "mlx/backend/omarchy/ane/h16_hwx.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace mlx::core::omarchy::ane {

namespace {

// ---- file layout constants -------------------------------------------

constexpr uint32_t kHwxMagic = 0xbeefface;
constexpr size_t kHwxHeaderBytes = 32;
constexpr size_t kHwxSubtypeAt = 8;
constexpr size_t kHwxCommandCountAt = 16;

constexpr uint32_t kCmdSegment = 0x19;
constexpr uint32_t kCmdDescriptor = 4; // program descriptor or tensor
constexpr uint32_t kCmdSurfaceName = 0x40;
constexpr uint32_t kDescriptorKindProgram = 4;
constexpr uint32_t kDescriptorKindTensor = 3;

constexpr uint32_t kProtInput = 1;
constexpr uint32_t kProtOutput = 2;
constexpr uint32_t kProtReadWrite = 3;

// Generic section: header, then one entry per buffer.
constexpr size_t kGenericHeaderBytes = 0x208;
constexpr size_t kGenericEntryBytes = 0x30;
constexpr size_t kGenericCountAt = 0x204;
constexpr uint32_t kGenericMagic = 1;
constexpr uint32_t kGenericVersion = 0x10;
constexpr uint32_t kGenericEntryPresent = 1;
constexpr uint32_t kGenericEntryTail = 0xffff;

// Operation section: a record count, then one fixed-size record.
constexpr size_t kOperationRecordBytes = 0x40c;
constexpr size_t kOperationRefsAt = 0x0c;
constexpr size_t kOperationRefsMax = (kOperationRecordBytes - kOperationRefsAt) / 8;

constexpr size_t kProcedureBytes = 0x38;
constexpr size_t kTextPropertiesBytes = 0x28;

// Buffer IDs: the output is 4, inputs, scratch and kernel buffers count
// up from 5; IDs end before 0x3d.
constexpr uint32_t kOutputBufferId = 4;
constexpr uint32_t kBufferIdEnd = 0x3d;
// Tag of the buffer H16_LOAD itself places in BAR slot 1.
constexpr uint32_t kResourceKernel = 1007;
constexpr uint32_t kConstantsSlot = 1;

// Register image byte ranges a task record may write ([begin, end)).
constexpr std::array<std::pair<uint32_t, uint32_t>, 9> kRegisterImage{{
    {0x0000, 0x005c},
    {0x4100, 0x41a4},
    {0x44d0, 0x44d4},
    {0x4500, 0x453c},
    {0x4900, 0x4930},
    {0x4d00, 0x4e44},
    {0x5100, 0x5154},
    {0x5500, 0x5620},
    {0x5900, 0x5930},
}};

// ---- refusal texts -----------------------------------------------------

const char* const kNotHwx = "not an H16G or H17 HWX";
const char* const kTruncatedCommand = "truncated load command";
const char* const kPastEnd = "HWX field lies past the end of the file";
const char* const kBadName = "HWX name is not valid UTF-8";
const char* const kSurfaceProt = "surface segment is neither input nor output";
const char* const kScratchProt = "scratch segment is not read/write";
const char* const kMissingParts =
    "HWX lacks a task stream, program descriptor or surfaces";
const char* const kOneOutput = "an H16 program has exactly one output";
const char* const kNoPrologue = "task stream lacks the prologue of its target";
const char* const kPadding = "nonzero padding in the task stream";
const char* const kOverrun = "task overruns the stream";
const char* const kBadRecord = "task record outside the register image";
const char* const kNoTask = "task stream has no task";
const char* const kNoBinding = "program binds no buffer through a BAR slot";
const char* const kTooManyBuffers = "program needs more buffer IDs than exist";

[[noreturn]] void refuse(const std::string& why) {
  throw AneH16HwxError(why);
}

// ---- byte views ----------------------------------------------------------

class View {
 public:
  // `past_end` is the refusal for a field that would lie past the end.
  View(const uint8_t* data, size_t size, const char* past_end = kPastEnd)
      : data_(data), size_(size), past_end_(past_end) {}

  size_t size() const {
    return size_;
  }
  const uint8_t* data() const {
    return data_;
  }

  uint32_t u32(size_t at) const {
    need(at, 4);
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) {
      v = (v << 8) | data_[at + i];
    }
    return v;
  }

  uint64_t u64(size_t at) const {
    return uint64_t(u32(at)) | (uint64_t(u32(at + 4)) << 32);
  }

  // [begin, end) cut to the file: a range that starts or ends beyond the
  // file is shortened, possibly to nothing.
  View cut(uint64_t begin, uint64_t end) const {
    const size_t b = size_t(std::min<uint64_t>(begin, size_));
    const size_t e = size_t(std::clamp<uint64_t>(end, b, size_));
    return View(data_ + b, e - b, past_end_);
  }

  bool all_zero() const {
    return std::all_of(data_, data_ + size_, [](uint8_t b) { return b == 0; });
  }

  // The NUL-terminated string at the start of the view (the whole view
  // when it holds no NUL), checked to be UTF-8.
  std::string text() const {
    const auto* nul = static_cast<const uint8_t*>(std::memchr(data_, 0, size_));
    const size_t n = nul ? size_t(nul - data_) : size_;
    if (!utf8(data_, n)) {
      refuse(kBadName);
    }
    return std::string(reinterpret_cast<const char*>(data_), n);
  }

 private:
  void need(size_t at, size_t n) const {
    if (at > size_ || size_ - at < n) {
      refuse(past_end_);
    }
  }

  // Strict UTF-8: shortest form only, no surrogates, nothing above U+10FFFF.
  static bool utf8(const uint8_t* s, size_t n) {
    size_t i = 0;
    while (i < n) {
      const uint8_t c = s[i];
      if (c < 0x80) {
        ++i;
        continue;
      }
      size_t len;
      uint8_t lo = 0x80, hi = 0xbf;
      if (c >= 0xc2 && c <= 0xdf) {
        len = 2;
      } else if (c >= 0xe0 && c <= 0xef) {
        len = 3;
        if (c == 0xe0) lo = 0xa0;
        if (c == 0xed) hi = 0x9f;
      } else if (c >= 0xf0 && c <= 0xf4) {
        len = 4;
        if (c == 0xf0) lo = 0x90;
        if (c == 0xf4) hi = 0x8f;
      } else {
        return false;
      }
      if (n - i < len) {
        return false;
      }
      if (s[i + 1] < lo || s[i + 1] > hi) {
        return false;
      }
      for (size_t k = 2; k < len; ++k) {
        if (s[i + k] < 0x80 || s[i + k] > 0xbf) {
          return false;
        }
      }
      i += len;
    }
    return true;
  }

  const uint8_t* data_;
  size_t size_;
  const char* past_end_;
};

void put32(std::vector<uint8_t>& out, size_t at, uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    out[at + i] = uint8_t(v >> (8 * i));
  }
}

void put64(std::vector<uint8_t>& out, size_t at, uint64_t v) {
  put32(out, at, uint32_t(v));
  put32(out, at + 4, uint32_t(v >> 32));
}

uint32_t prologue_of(uint32_t subtype) {
  return subtype == kAneHwxSubtypeH16G ? 1 : 0;
}

bool in_register_image(uint32_t word_index) {
  const uint64_t byte = uint64_t(word_index) * 4;
  return std::any_of(kRegisterImage.begin(), kRegisterImage.end(), [&](auto r) {
    return byte >= r.first && byte < r.second;
  });
}

std::string hex(uint64_t v) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%#llx", static_cast<unsigned long long>(v));
  return buf;
}

// ---- load commands -----------------------------------------------------

struct TensorInfo {
  uint32_t direction{0};
  unsigned __int128 span{0};
};

// What the load commands yield before the cross-checks.
struct Collected {
  AneH16Parsed parsed;
  std::optional<AneH16Region> text;
  std::optional<size_t> descriptor;
  std::map<std::string, TensorInfo> tensors;
};

// One segment command (LC_SEGMENT_64 layout: name at +8, vmaddr +24,
// vmsize +32, initprot +60, nsects +64, sections of 80 bytes from +72;
// a section has its name at +0, addr +32, size +40, file offset +48).
void read_segment(const View& file, size_t at, Collected& out) {
  const std::string segment = file.cut(at + 8, at + 24).text();
  file.u64(at + 24); // vmaddr: present in every segment, not used here
  const uint64_t vmsize = file.u64(at + 32);
  const uint32_t prot = file.u32(at + 60);
  const uint32_t nsects = file.u32(at + 64);
  auto& p = out.parsed;
  for (uint32_t i = 0; i < nsects; ++i) {
    const uint64_t sec = uint64_t(at) + 72 + uint64_t(80) * i;
    const std::string name = file.cut(sec, sec + 16).text();
    AneH16Region region;
    region.vm = file.u64(size_t(sec) + 32);
    region.size = file.u64(size_t(sec) + 40);
    region.offset = file.u32(size_t(sec) + 48);

    const std::string next = std::to_string(p.kernels.size());
    if (segment == "__FVMLIB") {
      if (prot != kProtInput && prot != kProtOutput) {
        refuse(kSurfaceProt);
      }
      AneH16Surface surface;
      surface.vm = region.vm;
      surface.allocation = vmsize;
      surface.direction = prot;
      p.surfaces.push_back(std::move(surface));
    } else if (segment == "__TEXT" && name == "__text") {
      out.text = region;
    } else if (segment == "__TEXT" && name == "__const") {
      p.constants = region.vm;
    } else if (segment == "__KERN_" + next && name == "__kern_" + next) {
      p.kernels.push_back(region);
    } else if (segment == "__DATA" && name == "__bss") {
      if (prot != kProtReadWrite) {
        refuse(kScratchProt);
      }
      p.scratch = AneH16Region{region.vm, region.size, 0};
    }
  }
}

// A tensor descriptor: direction +0x14, element count +0x28, batch +0x50,
// offset of its name within the command +0x7c.
void read_tensor(const View& file, size_t at, uint32_t cmdsize, Collected& out) {
  const uint32_t name_at = file.u32(at + 0x7c);
  const uint32_t count = file.u32(at + 0x28);
  const uint64_t batch = file.u64(at + 0x50);
  const std::string name =
      file.cut(uint64_t(at) + name_at, uint64_t(at) + cmdsize).text();
  TensorInfo info;
  info.direction = file.u32(at + 0x14);
  info.span = static_cast<unsigned __int128>(count) * batch;
  out.tensors[name] = info;
}

Collected collect(const View& file) {
  if (file.size() < kHwxHeaderBytes || file.u32(0) != kHwxMagic) {
    refuse(kNotHwx);
  }
  Collected out;
  out.parsed.subtype = file.u32(kHwxSubtypeAt);
  if (out.parsed.subtype != kAneHwxSubtypeH16G &&
      out.parsed.subtype != kAneHwxSubtypeH17) {
    refuse(kNotHwx);
  }
  size_t at = kHwxHeaderBytes;
  const uint32_t commands = file.u32(kHwxCommandCountAt);
  for (uint32_t c = 0; c < commands; ++c) {
    if (file.size() - at < 8) {
      refuse(kTruncatedCommand);
    }
    const uint32_t cmd = file.u32(at);
    const uint32_t cmdsize = file.u32(at + 4);
    if (cmdsize < 8 || cmdsize > file.size() - at) {
      refuse(kTruncatedCommand);
    }
    if (cmd == kCmdSegment) {
      read_segment(file, at, out);
    } else if (cmd == kCmdDescriptor) {
      const uint32_t kind = file.u32(at + 8);
      if (kind == kDescriptorKindProgram) {
        out.descriptor = at;
      } else if (kind == kDescriptorKindTensor) {
        read_tensor(file, at, cmdsize, out);
      }
    } else if (cmd == kCmdSurfaceName) {
      const uint64_t vm = file.u64(at + 16);
      const std::string name =
          file.cut(uint64_t(at) + 24, uint64_t(at) + cmdsize).text();
      for (auto& surface : out.parsed.surfaces) {
        if (surface.vm == vm) {
          surface.name = name;
        }
      }
    }
    at += cmdsize;
  }
  return out;
}

} // namespace

AneH16Parsed h16_parse_hwx(const uint8_t* hwx, size_t size) {
  // A load command whose fields run past the file is a truncated command.
  Collected got = collect(View(hwx, size, kTruncatedCommand));
  AneH16Parsed& p = got.parsed;
  if (!got.text || !got.descriptor || p.surfaces.size() < 2) {
    refuse(kMissingParts);
  }
  p.text = *got.text;
  p.descriptor = *got.descriptor;

  // Every surface needs a tensor of its own name and direction whose span
  // is nonzero and fits the surface allocation.
  for (auto& surface : p.surfaces) {
    const TensorInfo* tensor = nullptr;
    if (surface.name) {
      auto it = got.tensors.find(*surface.name);
      if (it != got.tensors.end()) {
        tensor = &it->second;
      }
    }
    if (!tensor || tensor->direction != surface.direction || tensor->span == 0 ||
        tensor->span > surface.allocation) {
      refuse(
          "surface " + (surface.name ? *surface.name : std::string("None")) +
          " has no matching tensor");
    }
    surface.span = uint64_t(tensor->span);
  }
  const auto outputs = std::count_if(
      p.surfaces.begin(), p.surfaces.end(),
      [](const AneH16Surface& s) { return s.direction == kProtOutput; });
  if (outputs != 1) {
    refuse(kOneOutput);
  }
  return p;
}

// Task stream: a 16-byte prologue (u32 marker, 12 zero bytes), then tasks
// on 16-byte boundaries. A task's first word holds its length in words in
// bits 16..26 (0 marks 16 bytes of zero padding). Its header is 9 words,
// or 10 when bits 0..1 of word 8 are both set. After the header come
// register records: a record word, then its values.
//  - bit 31 set: a masked record, 1 + popcount(bits 15..30) values;
//  - bit 31 clear: (bits 15..20) + 1 consecutive values; with bit 29 set
//    the values come from the BAR slot in bits 23..28.
// Bits 0..14 index the first register word, which must lie in the
// register image.
std::pair<uint32_t, std::vector<uint32_t>> h16_walk_tasks(
    const uint8_t* stream,
    size_t size,
    uint32_t prologue) {
  const View s(stream, size);
  if (size < 16 || size % 4 != 0 || s.u32(0) != prologue ||
      !s.cut(4, 16).all_zero()) {
    refuse(kNoPrologue);
  }
  uint32_t tasks = 0;
  std::set<uint32_t> slots;
  size_t at = 16;
  while (at < size) {
    const uint32_t words = (s.u32(at) >> 16) & 0x7ff;
    if (words == 0) {
      if (!s.cut(at, uint64_t(at) + 16).all_zero()) {
        refuse(kPadding);
      }
      at += 16;
      continue;
    }
    const size_t end = at + size_t(4) * words;
    if (at + 36 > size) {
      refuse(kOverrun); // the header word that sizes the header is missing
    }
    const uint32_t header = (s.u32(at + 32) & 3) == 3 ? 10 : 9;
    if (words < header || end > size) {
      refuse(kOverrun);
    }
    size_t rec = at + size_t(4) * header;
    while (rec < end) {
      const uint32_t word = s.u32(rec);
      rec += 4;
      size_t values;
      if (word & (1u << 31)) {
        values = 1 + size_t(__builtin_popcount((word >> 15) & 0xffff));
      } else {
        values = ((word >> 15) & 0x3f) + 1;
        if (word & (1u << 29)) {
          slots.insert((word >> 23) & 0x3f);
        }
      }
      if (!in_register_image(word & 0x7fff) || rec + 4 * values > end) {
        refuse(kBadRecord);
      }
      rec += 4 * values;
    }
    at = (end + 15) & ~size_t(15);
    ++tasks;
  }
  if (tasks == 0) {
    refuse(kNoTask);
  }
  return {tasks, std::vector<uint32_t>(slots.begin(), slots.end())};
}

AneH16Sections h16_build_sections(const uint8_t* hwx, size_t size) {
  const View file(hwx, size);
  AneH16Parsed p = h16_parse_hwx(hwx, size);
  const uint64_t text_end = p.text.size > UINT64_MAX - p.text.offset
      ? UINT64_MAX
      : p.text.offset + p.text.size;
  const View stream = file.cut(p.text.offset, text_end);
  const auto [tasks, slots] =
      h16_walk_tasks(stream.data(), stream.size(), prologue_of(p.subtype));

  // The program descriptor gives BAR slot n its base at +0x10 * (n + 1).
  std::map<uint32_t, uint64_t> base_of;
  for (uint32_t slot : slots) {
    base_of[slot] = file.u64(p.descriptor + size_t(0x10) * (slot + 1));
  }
  const bool constants_in_bar = p.constants &&
      std::any_of(base_of.begin(), base_of.end(),
                  [&](const auto& kv) { return kv.second == *p.constants; });
  // H16_LOAD places one buffer in BAR slot 1. With more than one kernel
  // section, or a kernel section and constants both read, every kernel
  // section becomes a buffer of its own.
  const bool kernel_buffers =
      p.kernels.size() > 1 || (!p.kernels.empty() && constants_in_bar);

  AneH16Sections out;
  out.subtype = p.subtype;
  out.tasks = tasks;

  uint32_t next_input = kOutputBufferId + 1;
  auto take_input = [&]() {
    if (next_input >= kBufferIdEnd) {
      refuse(kTooManyBuffers);
    }
    return next_input++;
  };

  struct Entry {
    uint32_t id;
    uint32_t direction;
    uint64_t bytes;
  };
  std::vector<Entry> entries;
  for (auto& surface : p.surfaces) {
    surface.id = surface.direction == kProtOutput ? kOutputBufferId : take_input();
    entries.push_back({surface.id, surface.direction, surface.allocation});
  }
  if (p.scratch) {
    out.scratch_id = take_input();
    out.scratch_bytes = p.scratch->size;
    entries.push_back({*out.scratch_id, kProtInput, p.scratch->size});
  }
  if (kernel_buffers) {
    for (const auto& kernel : p.kernels) {
      out.kernel_ids.push_back(take_input());
      entries.push_back({out.kernel_ids.back(), kProtInput, kernel.size});
    }
  }

  // Generic: header {magic, version, ..., count at 0x204}; each entry
  // {present, id, type (0 in, 1 out), _, direction, ..., bytes at +0x20,
  // 0xffff at +0x28}.
  auto& generic = out.sections[0];
  generic.assign(kGenericHeaderBytes + kGenericEntryBytes * entries.size(), 0);
  put32(generic, 0, kGenericMagic);
  put32(generic, 4, kGenericVersion);
  put32(generic, kGenericCountAt, uint32_t(entries.size()));
  for (size_t k = 0; k < entries.size(); ++k) {
    const size_t e = kGenericHeaderBytes + kGenericEntryBytes * k;
    put32(generic, e + 0x00, kGenericEntryPresent);
    put32(generic, e + 0x04, entries[k].id);
    put32(generic, e + 0x08, entries[k].direction - 1);
    put32(generic, e + 0x10, entries[k].direction);
    put64(generic, e + 0x20, entries[k].bytes);
    put32(generic, e + 0x28, kGenericEntryTail);
  }

  // BAR bindings, slot ascending. A zero base is a slot without memory.
  for (const auto& [slot, base] : base_of) {
    if (base == 0) {
      continue;
    }
    auto surface = std::find_if(
        p.surfaces.begin(), p.surfaces.end(),
        [&](const AneH16Surface& s) { return s.vm == base; });
    auto kernel = std::find_if(
        p.kernels.begin(), p.kernels.end(),
        [&](const AneH16Region& k) { return k.vm == base; });
    if (surface != p.surfaces.end()) {
      out.refs.emplace_back(slot, surface->id);
    } else if (p.scratch && base == p.scratch->vm) {
      out.refs.emplace_back(slot, *out.scratch_id);
    } else if (kernel != p.kernels.end()) {
      out.refs.emplace_back(
          slot,
          kernel_buffers ? out.kernel_ids[size_t(kernel - p.kernels.begin())]
                         : kResourceKernel);
    } else if (
        p.constants && base == *p.constants && slot == kConstantsSlot &&
        (kernel_buffers || p.kernels.empty())) {
      out.refs.emplace_back(slot, kResourceKernel);
    } else {
      refuse(
          "BAR slot " + std::to_string(slot) + " base " + hex(base) +
          " names no surface, scratch, kernel or constants");
    }
  }
  if (out.refs.empty() || out.refs.size() > kOperationRefsMax) {
    refuse(kNoBinding);
  }

  // Operation: record count 1, then the record: binding count at +0x08,
  // (slot, tag) pairs from +0x0c.
  auto& operation = out.sections[1];
  operation.assign(4 + kOperationRecordBytes, 0);
  put32(operation, 0, 1);
  put32(operation, 4 + 0x08, uint32_t(out.refs.size()));
  for (size_t j = 0; j < out.refs.size(); ++j) {
    put32(operation, 4 + kOperationRefsAt + 8 * j, out.refs[j].first);
    put32(operation, 4 + kOperationRefsAt + 8 * j + 4, out.refs[j].second);
  }

  // Procedure: one procedure; record table at 0x18, 0x20 bytes.
  auto& procedure = out.sections[2];
  procedure.assign(kProcedureBytes, 0);
  put32(procedure, 0x00, 1);
  put64(procedure, 0x08, 0x18);
  put64(procedure, 0x10, 0x20);
  constexpr std::array<uint32_t, 8> kProcedureRecord{
      1, 3, 0, 0, 1, 0, 0xffffffff, 4};
  for (size_t i = 0; i < kProcedureRecord.size(); ++i) {
    put32(procedure, 0x18 + 4 * i, kProcedureRecord[i]);
  }

  // Text properties: one segment of `tasks` blocks over the whole stream.
  auto& properties = out.sections[3];
  properties.assign(kTextPropertiesBytes, 0);
  put32(properties, 0x00, 1);
  put32(properties, 0x0c, tasks);
  put64(properties, 0x20, p.text.size);

  out.surfaces = std::move(p.surfaces);
  return out;
}

AneMappedFile::AneMappedFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(
        "[omarchy-ane] cannot open " + path + ": " + std::strerror(errno));
  }
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    const int err = errno;
    ::close(fd);
    throw std::runtime_error(
        "[omarchy-ane] cannot stat " + path + ": " + std::strerror(err));
  }
  size_ = size_t(st.st_size);
  if (size_ > 0) {
    void* map = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
      const int err = errno;
      ::close(fd);
      throw std::runtime_error(
          "[omarchy-ane] cannot map " + path + ": " + std::strerror(err));
    }
    data_ = static_cast<const uint8_t*>(map);
  }
  ::close(fd);
}

AneMappedFile::~AneMappedFile() {
  if (data_) {
    ::munmap(const_cast<uint8_t*>(data_), size_);
  }
}

} // namespace mlx::core::omarchy::ane
