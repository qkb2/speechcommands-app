cmake_minimum_required(VERSION 3.29 FATAL_ERROR)
project(speechcommands CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set options for executorch build (based on linux/llm preset)
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)
option(EXECUTORCH_XNNPACK_ENABLE_KLEIDI "" OFF)

# Set options for KissFFT build
option(KISSFFT_PKGCONFIG "Build pkg-config files" OFF)
option(KISSFFT_STATIC "Build kissfft as static (ON) or shared library (OFF)" OFF)
option(KISSFFT_TEST "Build kissfft tests" OFF)
option(KISSFFT_TOOLS "Build kissfft command-line tools" OFF)

# Add subdirectories
add_subdirectory("third-party/executorch")
add_subdirectory("third-party/kissfft")

# Create executable
file(GLOB_RECURSE APP_SOURCES
    src/*.cpp
)

add_executable(speechcommands_app ${APP_SOURCES})

# Include directories
target_include_directories(speechcommands_app 
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  third-party/kissfft
  third-party/executorch
)

# Link libraries
target_link_libraries(
  speechcommands_app
  PRIVATE executorch
          extension_module_static
          extension_tensor
          optimized_native_cpu_ops_lib
          xnnpack_backend
          kissfft
)

# Set output directory
set_target_properties(speechcommands_app
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
