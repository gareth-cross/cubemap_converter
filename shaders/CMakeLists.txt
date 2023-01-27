
# List of shader files:
set(SHADER_FILES vertex.glsl fragment_display.glsl fragment_cubemap.glsl)

# Iterate over all of them:
foreach (shader_file ${SHADER_FILES})
    # Split the extensions:
    set(shader_file_path "${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}")
    get_filename_component(shader_name ${shader_file_path} NAME_WE)
    message(STATUS "Discovered shader: ${shader_name}")

    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.hpp")
    list(APPEND SHADER_HEADER_FILES ${output_file})
    add_custom_command(
            OUTPUT ${output_file}
            MAIN_DEPENDENCY ${shader_file_path}
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_header.py"
            COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_header.py
            --input ${shader_file_path}
            --output ${output_file}
            VERBATIM
    )
endforeach ()

# Create a target for the shader codegen:
add_custom_target(shaders_codegen DEPENDS ${SHADER_HEADER_FILES})

# Create a target that exports the headers:
add_library(shaders INTERFACE)
add_dependencies(shaders shaders_codegen)
target_include_directories(shaders INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
