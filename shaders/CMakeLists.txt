add_executable(generate_shader_header generate_shader_header.cc)
target_compile_features(generate_shader_header PUBLIC cxx_std_20)
if(MSVC)
  target_compile_options(generate_shader_header PRIVATE /W4 /WX)
else()
  target_compile_options(generate_shader_header PRIVATE -Wall -Wextra -pedantic
                                                        -Werror)
endif()
target_link_libraries(generate_shader_header CLI11::CLI11 fmt::fmt-header-only)

add_library(shaders INTERFACE)
target_include_directories(shaders INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

function(add_shader)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCE_FILE)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(SHADER_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_SOURCE_FILE}")
  get_filename_component(SHADER_NAME ${SHADER_FILE_PATH} NAME_WE)
  set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.hpp")

  add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    COMMAND
      "$<TARGET_FILE_DIR:generate_shader_header>/$<TARGET_FILE_NAME:generate_shader_header>"
      --input ${SHADER_FILE_PATH} --output ${OUTPUT_FILE}
    COMMENT "Generate shader header: ${SHADER_NAME}"
    DEPENDS generate_shader_header ${SHADER_FILE_PATH})
  set_source_files_properties(${OUTPUT_FILE} PROPERTIES GENERATED TRUE)

  add_custom_target(${SHADER_NAME}_header DEPENDS ${OUTPUT_FILE})
  add_dependencies(shaders ${SHADER_NAME}_header)
endfunction()

add_shader(SOURCE_FILE vertex.glsl)
add_shader(SOURCE_FILE fragment_display.glsl)
add_shader(SOURCE_FILE fragment_oversampled_cubemap.glsl)
