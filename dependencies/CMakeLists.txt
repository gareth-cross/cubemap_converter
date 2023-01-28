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

# Add zlib (unused)
function (add_zlib)
    SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable building shared library" FORCE)
    add_subdirectory(zlib)
endfunction()

# Add libpng (unused)
function (add_libpng)
    SET(PNG_BUILD_ZLIB ON CACHE BOOL "We are building zlib" FORCE)
    SET(PNG_SHARED OFF CACHE BOOL "Do not build shared library." FORCE)
    SET(PNG_EXECUTABLES OFF CACHE BOOL "Do not build executables." FORCE)
    SET(PNG_TESTS OFF CACHE BOOL "Do not build tests." FORCE)
    set(ZLIB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/zlib" "${CMAKE_CURRENT_BINARY_DIR}/zlib")
    add_subdirectory(libpng)
endfunction()
