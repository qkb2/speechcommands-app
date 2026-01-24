# ARM Toolchain for Pico 2 W (Cortex-M33, soft float)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_C_COMPILER "arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-none-eabi-g++")
set(CMAKE_ASM_COMPILER "arm-none-eabi-gcc")

# Must match Pico SDK settings: Cortex-M33, soft float ABI
set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-m33 -mthumb -mfloat-abi=soft")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-m33 -mthumb -mfloat-abi=soft -fno-exceptions -fno-rtti")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)