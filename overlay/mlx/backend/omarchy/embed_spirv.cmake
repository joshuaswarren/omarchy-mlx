if(NOT INPUT OR NOT OUTPUT OR NOT NAME)
  message(FATAL_ERROR "INPUT, OUTPUT, and NAME are required")
endif()

file(READ "${INPUT}" bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${bytes}")
file(
  WRITE "${OUTPUT}"
  "#pragma once\n\n#include <cstddef>\n\nnamespace mlx::core::omarchy::shaders {\n\nalignas(4) inline constexpr unsigned char ${NAME}[] = {${bytes}};\ninline constexpr size_t ${NAME}_size = sizeof(${NAME});\n\n} // namespace mlx::core::omarchy::shaders\n")
