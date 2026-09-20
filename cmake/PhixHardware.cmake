# ===========================================================================
# PhixHardware.cmake — toolchain + GPU detection for portable builds.
#
# Two entry points, in call order:
#
#   phix_bootstrap_cuda_compiler()    BEFORE project() — locate nvcc when it
#                                     is not on PATH, sanitise CUDAARCHS
#   phix_resolve_cuda_arch(<out_var>) AFTER  project() — decide the target
#                                     compute capability and validate it
#                                     against both the GPU and the toolkit
#
# Goal: `cmake .. && make` succeeds, with a runnable binary, on any machine
# that has an NVIDIA GPU and a CUDA toolkit — no machine-specific flags.
# ===========================================================================

# ---------------------------------------------------------------------------
# phix_bootstrap_cuda_compiler()
#
# CMake enables the CUDA language inside project(); if nvcc is not on PATH it
# aborts with an opaque "No CMAKE_CUDA_COMPILER could be found".  Probe the
# usual install locations first and fail with an actionable message instead.
#
# Also clears an EMPTY CUDAARCHS environment variable: CMake seeds
# CMAKE_CUDA_ARCHITECTURES from it, and an empty value breaks the compiler
# test before any of our code runs.
# ---------------------------------------------------------------------------
macro(phix_bootstrap_cuda_compiler)
    if(DEFINED ENV{CUDAARCHS} AND "$ENV{CUDAARCHS}" STREQUAL "")
        message(STATUS "PhiX: ignoring empty CUDAARCHS environment variable")
        unset(ENV{CUDAARCHS})
    endif()

    if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        find_program(PHIX_NVCC nvcc
            HINTS
                ENV CUDA_PATH
                ENV CUDA_HOME
                ENV CUDA_TOOLKIT_ROOT_DIR
                /usr/local/cuda
                /opt/cuda
            PATH_SUFFIXES bin)
        if(PHIX_NVCC)
            set(CMAKE_CUDA_COMPILER "${PHIX_NVCC}")
            message(STATUS "PhiX: CUDA compiler ${PHIX_NVCC}")
        else()
            message(FATAL_ERROR
                "PhiX: no CUDA compiler (nvcc) found.\n"
                "  PhiX is a CUDA-only framework — there is no CPU fallback.\n"
                "  Install the CUDA toolkit, then either put nvcc on PATH or\n"
                "  point CMake at it explicitly:\n"
                "    cmake -DCMAKE_CUDA_COMPILER=/path/to/bin/nvcc ..\n"
                "  Toolkit install options:\n"
                "    conda install -c nvidia cuda-toolkit    (no root needed)\n"
                "    sudo apt install nvidia-cuda-toolkit    (Debian/Ubuntu)\n"
                "    https://developer.nvidia.com/cuda-downloads")
        endif()
    endif()
endmacro()

# ---------------------------------------------------------------------------
# _phix_probe_gpu_arch(<out_var>)
#
# Compute capabilities of the local GPUs as a de-duplicated list of CMake
# architecture numbers ("12.0" -> "120").  Empty when there is no usable
# nvidia-smi (GPU-less build host, container without device access, or a
# driver too old to answer --query-gpu=compute_cap).
# ---------------------------------------------------------------------------
function(_phix_probe_gpu_arch out_var)
    set(_archs "")
    find_program(PHIX_NVIDIA_SMI nvidia-smi)
    if(PHIX_NVIDIA_SMI)
        execute_process(
            COMMAND ${PHIX_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _out
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REPLACE "\n" ";" _lines "${_out}")
        foreach(_line IN LISTS _lines)
            string(STRIP "${_line}" _line)
            # old drivers answer "Field ... is not a valid field to query."
            if(_line MATCHES "^([0-9]+)\\.([0-9]+)$")
                list(APPEND _archs "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _archs)
    endif()
    set(${out_var} "${_archs}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# _phix_check_toolkit_supports(<archs>)
#
# nvcc drops old architectures between major releases (CUDA 13 no longer
# builds sm_50..sm_70) and only learns new ones once they ship (sm_120 needs
# CUDA >= 12.8).  Ask the toolkit directly rather than carrying a version
# table.  Silently skipped when --list-gpu-arch is unavailable.
# ---------------------------------------------------------------------------
function(_phix_check_toolkit_supports archs)
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
        OUTPUT_VARIABLE _out
        RESULT_VARIABLE _rc
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _rc EQUAL 0 OR _out STREQUAL "")
        return()
    endif()

    string(REPLACE "\n" ";" _lines "${_out}")
    set(_supported "")
    foreach(_line IN LISTS _lines)
        if(_line MATCHES "compute_([0-9]+)")
            list(APPEND _supported "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    foreach(_arch IN LISTS archs)
        if(NOT _arch IN_LIST _supported)
            list(SORT _supported COMPARE NATURAL)
            string(REPLACE ";" " " _pretty "${_supported}")
            message(FATAL_ERROR
                "PhiX: this CUDA toolkit cannot build for sm_${_arch}.\n"
                "  ${CMAKE_CUDA_COMPILER}\n"
                "  (CUDA ${CMAKE_CUDA_COMPILER_VERSION}) supports: ${_pretty}\n"
                "  sm_${_arch} is either newer than the toolkit (upgrade CUDA)\n"
                "  or an architecture it has dropped (use an older CUDA).")
        endif()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# phix_resolve_cuda_arch(<out_var>)
#
# PHIX_CUDA_ARCH = "auto" (default): adopt the local GPU's compute capability.
# PHIX_CUDA_ARCH = 75 / 86 / "75;86": honour it, but abort when it does not
# match the local GPU — a mismatched build compiles and links cleanly, then
# dies at the first kernel launch with "no kernel image is available", and a
# stale build/ dir configured for a previous GPU is an easy trap.  Cross-
# compilation stays possible via -DPHIX_ALLOW_ARCH_MISMATCH=ON.
# ---------------------------------------------------------------------------
function(phix_resolve_cuda_arch out_var)
    _phix_probe_gpu_arch(_gpu)

    if(PHIX_CUDA_ARCH STREQUAL "auto" OR PHIX_CUDA_ARCH STREQUAL "")
        if(_gpu)
            list(GET _gpu 0 _arch)
            list(LENGTH _gpu _n)
            if(_n GREATER 1)
                string(REPLACE ";" " " _pretty "${_gpu}")
                message(WARNING
                    "PhiX: GPUs of differing compute capability present "
                    "(${_pretty}); building for sm_${_arch} only. Set "
                    "-DPHIX_CUDA_ARCH=\"${_pretty}\" (space -> ';') to cover all.")
            endif()
            message(STATUS "PhiX: auto-detected GPU architecture sm_${_arch}")
        else()
            set(_arch "75")
            message(WARNING
                "PhiX: no GPU detected (nvidia-smi missing or unusable) — "
                "falling back to sm_${_arch}. Binaries will not launch on a "
                "different architecture; pass -DPHIX_CUDA_ARCH=<cc> to target "
                "the deployment GPU explicitly.")
        endif()
    else()
        set(_arch "${PHIX_CUDA_ARCH}")
        foreach(_a IN LISTS _arch)
            if(NOT _a MATCHES "^[0-9]+$")
                message(FATAL_ERROR
                    "PhiX: PHIX_CUDA_ARCH must be 'auto' or compute-capability "
                    "digits (e.g. 75, 86, 120), got '${_a}'.")
            endif()
        endforeach()
        if(_gpu)
            set(_hit FALSE)
            foreach(_a IN LISTS _arch)
                if(_a IN_LIST _gpu)
                    set(_hit TRUE)
                endif()
            endforeach()
            if(NOT _hit AND NOT PHIX_ALLOW_ARCH_MISMATCH)
                string(REPLACE ";" " " _pretty_gpu "${_gpu}")
                string(REPLACE ";" " " _pretty_req "${_arch}")
                list(GET _gpu 0 _first)
                message(FATAL_ERROR
                    "PhiX: PHIX_CUDA_ARCH=${_pretty_req} does not match the "
                    "local GPU (sm_${_pretty_gpu}).\n"
                    "  Kernels would build fine and then fail at launch with "
                    "'no kernel image is available'.\n"
                    "  Fix:      cmake -DPHIX_CUDA_ARCH=${_first} ..   (or drop "
                    "the flag — 'auto' is the default)\n"
                    "  Override: -DPHIX_ALLOW_ARCH_MISMATCH=ON        (cross-"
                    "compiling for another machine)")
            endif()
        endif()
        message(STATUS "PhiX: using requested GPU architecture sm_${_arch}")
    endif()

    _phix_check_toolkit_supports("${_arch}")
    set(${out_var} "${_arch}" PARENT_SCOPE)
endfunction()
