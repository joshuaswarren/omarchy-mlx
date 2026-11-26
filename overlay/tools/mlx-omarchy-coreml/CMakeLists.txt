# Host-only Core ML front-end CLI. A python program: installed, never
# compiled; it opens no device and links nothing.
install(
  PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/mlx_omarchy_coreml.py
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME mlx-omarchy-coreml)
