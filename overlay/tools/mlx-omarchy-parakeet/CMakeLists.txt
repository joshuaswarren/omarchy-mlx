# Pinned-reference Parakeet CLI. A python program: installed, never
# compiled; `download` is network-only, `transcribe` drives the full
# runtime (GPU through libmlx, ANE through the shipped worker).
install(
  PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/mlx_omarchy_parakeet.py
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME mlx-omarchy-parakeet)

# Runtime assets for the pinned ANE encoder islands: compiled bundles,
# the strict libane the worker dlopens, and the hash pin manifest. The
# ANE exists only on aarch64 Apple-silicon hosts, so other
# architectures install the CLI without the (arm64-only) payloads.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/share/mlx-omarchy
    DESTINATION ${CMAKE_INSTALL_DATADIR})
endif()
