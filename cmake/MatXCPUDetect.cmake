# =========================================================
# MatXCPUDetect.cmake
# CPU vendor auto-detection via try_run
# =========================================================

set(MATX_CPU_PLATFORM "" CACHE STRING "Override CPU platform: AMD | INTEL | OTHER (leave empty for auto-detect)")

if (MATX_CPU_PLATFORM STREQUAL "")

    set(CPU_DETECT_SOURCE ${CMAKE_BINARY_DIR}/detect_cpu.cpp)

    file(WRITE ${CPU_DETECT_SOURCE} "
    #include <iostream>
    #if defined(_MSC_VER)
    #include <intrin.h>
    #else
    #include <cpuid.h>
    #endif

    int main() {
    char vendor[13] = {0};

    #if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    ((int*)vendor)[0] = regs[1];
    ((int*)vendor)[1] = regs[3];
    ((int*)vendor)[2] = regs[2];
    #else
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    ((unsigned int*)vendor)[0] = ebx;
    ((unsigned int* )vendor)[1] = edx;
    ((unsigned int*)vendor)[2] = ecx;
    #endif

    std::cout << vendor;
    return 0;
    }
    ")

    try_run(
        RUN_RESULT
        COMPILE_RESULT
        ${CMAKE_BINARY_DIR}
        ${CPU_DETECT_SOURCE}
        RUN_OUTPUT_VARIABLE CPU_VENDOR
    )

    string(STRIP "${CPU_VENDOR}" CPU_VENDOR)

    if (CPU_VENDOR STREQUAL "AuthenticAMD")
        set(CPU_PLATFORM "AMD")
    elseif (CPU_VENDOR STREQUAL "GenuineIntel")
        set(CPU_PLATFORM "INTEL")
    else()
        set(CPU_PLATFORM "OTHER")
    endif()

else()
    set(CPU_PLATFORM "${MATX_CPU_PLATFORM}")
    set(CPU_VENDOR "(manual override)")
endif()

message(STATUS "CPU vendor detected: ${CPU_VENDOR}")
message(STATUS "CPU platform: ${CPU_PLATFORM}")
