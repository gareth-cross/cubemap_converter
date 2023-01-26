# Find OpenGL
find_package(OpenGL REQUIRED)

# Add GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

# Add glm
add_subdirectory(glm)

# Add libfmt
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt)

# Add cli11
add_subdirectory(cli11)
