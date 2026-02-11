/*
 * Copyright (C) 2022-2024, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef _AU_CONFIG_H_INCLUDED
#define _AU_CONFIG_H_INCLUDED 2

// clang-format off

/* #undef AU_CONFIG_LITTLE_ENDIAN */

// CPU Identification
/* #undef AU_ENABLE_AOCL_CPUID */

// AOCL Foundations Release Version
/* #undef AU_PACKAGE_VERSION */
/* #undef AU_PACKAGE_SUFFIX */

// AOCL Foundation Build type
#define AU_BUILD_TYPE_RELEASE 1
#define AU_BUILD_TYPE_DEBUG 0
#define AU_BUILD_TYPE_DEVELOPER 0
#define AU_CPU_ARCH_X86 1

// Compiler detection

/* #undef AU_COMPILER_IS_GNU */
#define AU_COMPILER_IS_CLANG 1
/* #undef AU_COMPILER_IS_MSVC */

/* #undef AU_TARGET_OS_IS_LINUX */
#define AU_TARGET_OS_IS_WINDOWS

/* #undef AU_TARGET_ENDIAN_BIG */
/* #undef AU_TARGET_ENDIAN_LITTLE */

#define AU_WARN_DEPRECATION
// enable assertion
/* #undef AU_ENABLE_ASSERTIONS */
// clang-format on
#endif // _AU_CONFIG_H_INCLUDED
