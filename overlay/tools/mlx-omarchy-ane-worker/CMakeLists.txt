# Bounded ANE worker CLI (the fd-protocol serve implementation). ANE
# device builds (MLX_OMARCHY_ANE_DEVICE): host builds carry no libane
# surface at all. libane is dlopen'd at runtime from the path the
# caller passes (--libane), so only its header is needed at build time;
# the vendored libane/ane.h matches the pinned omarchy-ane 6fa243ac
# checkout the shipped libane-strict.so is built from.
if(MLX_OMARCHY_ANE_DEVICE AND NOT TARGET mlx-omarchy-ane-worker)
  set(OMARCHY_ANE_INCLUDE_DIR "" CACHE PATH
      "omarchy-ane libane headers (default: the vendored pinned copy)")
  if(NOT OMARCHY_ANE_INCLUDE_DIR)
    set(OMARCHY_ANE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libane)
  endif()
  add_executable(
    mlx-omarchy-ane-worker
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mlx/backend/omarchy/ane/worker_libane.cpp)
  target_link_libraries(mlx-omarchy-ane-worker PRIVATE mlx ${CMAKE_DL_LIBS})
  target_include_directories(
    mlx-omarchy-ane-worker PRIVATE ${OMARCHY_ANE_INCLUDE_DIR})
  target_compile_definitions(mlx-omarchy-ane-worker
                             PRIVATE MLX_OMARCHY_ANE_DEVICE=1)
  set_target_properties(
    mlx-omarchy-ane-worker PROPERTIES OUTPUT_NAME mlx-omarchy-ane-worker
                                      INSTALL_RPATH "$ORIGIN/../lib")
  install(TARGETS mlx-omarchy-ane-worker
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../mlx/backend/omarchy/ane/mlx-omarchy-ane.conf
          DESTINATION lib/tmpfiles.d)
endif()
