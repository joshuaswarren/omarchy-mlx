#!/bin/bash
# usage: REPO=<checkout> gate-mutations.sh   runs gate-harness.sh on the old header, the new header, and seven mutations of the new header; each mutation must fail at least one check.
G="$(dirname "$0")/gate-harness.sh"
echo "== old header (origin/main): the test body must not compile"; "$G" old 2>&1 | tail -3
echo "== new header: all checks must pass"; "$G" new 2>&1 | tail -4
run() { echo "== mutation $1"; GATE_SED="$2" "$G" new 2>&1 | tail -${3:-40} | cut -c1-170; }
F='/inline bool sdpa_split_pv_wide(/,$'
run "G13G limit 16 -> 17" 's/if (device_name.find("G13G") != std::string::npos) return 16u;/if (device_name.find("G13G") != std::string::npos) return 17u;/' 6
run "G13C limit 64 -> 66" 's/if (device_name.find("G13C") != std::string::npos) return 64u;/if (device_name.find("G13C") != std::string::npos) return 66u;/' 6
run "G13C limit 64 -> 63" 's/if (device_name.find("G13C") != std::string::npos) return 64u;/if (device_name.find("G13C") != std::string::npos) return 63u;/' 6
run "block count rounded down instead of up" "$F"' s/ + 127u) \/ 128u/ + 0u) \/ 128u/' 8
run "64-key blocks instead of 128" "$F"' s/ + 127u) \/ 128u/ + 63u) \/ 64u/' 8
run "zero guard workgroups != 0 removed" "$F"' s/return workgroups != 0 \&\& workgroups <= limit;/return workgroups <= limit;/' 6
run "32-bit workgroup count (wraps)" "$F"' s/uint64_t workgroups = static_cast<uint64_t>(kv_heads) \* ((static_cast<uint64_t>(k_len)/uint32_t workgroups = static_cast<uint32_t>(kv_heads) * ((static_cast<uint32_t>(k_len)/' 6
run "key-value heads ignored (keys only)" "$F"' s/uint64_t workgroups = static_cast<uint64_t>(kv_heads) \*/uint64_t workgroups = 2 */' 8
