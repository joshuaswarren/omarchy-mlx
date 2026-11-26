add_executable(mlx-omarchy-info ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(mlx-omarchy-info PRIVATE mlx)
set_target_properties(mlx-omarchy-info PROPERTIES
  OUTPUT_NAME mlx-omarchy-info
  INSTALL_RPATH "$ORIGIN/../lib")

install(TARGETS mlx-omarchy-info RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
