cmake_minimum_required(VERSION 3.29 FATAL_ERROR)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/arm-pico-toolchain.cmake)
project(speechcommands C CXX ASM)

set(PICO_BOARD pico2_w)

include(third-party/pico-sdk/pico_sdk_init.cmake)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

pico_sdk_init()

# Set options for executorch build (based on linux/llm preset)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR OFF)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER OFF)
set(EXECUTORCH_BUILD_ARM_BAREMETAL ON)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set(EXECUTORCH_BUILD_CORTEX_M ON)
set(EXECUTORCH_ENABLE_LOGGING ON)
set(C10_USING_CUSTOM_GENERATED_MACROS ON)
set(EXECUTORCH_ENABLE_LOGGING OFF)
set(EXECUTORCH_PAL_DEFAULT minimal)
set(EXECUTORCH_SELECT_OPS_MODEL ${CMAKE_CURRENT_LIST_DIR}/tiny_kws2nq.pte)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")

# Add subdirectories
add_subdirectory("third-party/executorch")

# Create executable
file(GLOB_RECURSE APP_SOURCES
    src/*.cpp
)

add_executable(speechcommands_app ${APP_SOURCES})

# Include directories
target_include_directories(speechcommands_app 
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  third-party/executorch
  third-party/executorch/runtime/core/portable_type/c10
  ${CMAKE_CURRENT_SOURCE_DIR}/third-party
)

target_compile_options(speechcommands_app PRIVATE
    -mcpu=cortex-m33 -mthumb -mfloat-abi=soft
    -Os -ffunction-sections -fdata-sections
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

# Link libraries
target_link_libraries(speechcommands_app PRIVATE
    executorch
    executorch_core
    portable_ops_lib
    portable_kernels
    pico_stdlib
)

# Set output directory
set_target_properties(speechcommands_app
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    OUTPUT_NAME "speechcommands_app.elf"
)

# Enable USB serial
pico_enable_stdio_usb(speechcommands_app 1)
pico_enable_stdio_uart(speechcommands_app 0)

# Generate UF2
pico_add_extra_outputs(speechcommands_app)