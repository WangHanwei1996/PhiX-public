cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Hardware / toolchain detection.  nvcc must be located BEFORE project()
# enables the CUDA language, so this include comes first.
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(PhixHardware)
phix_bootstrap_cuda_compiler()

project(PhiX LANGUAGES CXX CUDA)

enable_testing()

# ---------------------------------------------------------------------------
# Build type — default to Release (an empty CMAKE_BUILD_TYPE compiles the
# host side with no optimisation at all)
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "PhiX: CMAKE_BUILD_TYPE not set — defaulting to Release")
endif()

# ---------------------------------------------------------------------------
# Profiling helpers
#   PHIX_LINEINFO=ON     nvcc -lineinfo (source-correlated kernel profiling)
#   PHIX_ENABLE_NVTX=ON  compile PHIX_NVTX_RANGE() into real NVTX ranges
# ---------------------------------------------------------------------------
option(PHIX_LINEINFO    "Add -lineinfo to CUDA compiles (profiling)" OFF)
option(PHIX_ENABLE_NVTX "Enable NVTX ranges (perf/Perf.h)"           OFF)

# ---------------------------------------------------------------------------
# Precision policy (core/Real.h)
#   DOUBLE (default): Real = double — full test suite + applications
#   FLOAT           : Real = float  — library + float smoke tests + benchmark
#                     (apps and 1e-12-tolerance suites are not configured)
# ---------------------------------------------------------------------------
set(PHIX_PRECISION "DOUBLE" CACHE STRING "Field scalar type: DOUBLE or FLOAT")
if(PHIX_PRECISION STREQUAL "FLOAT")
    add_compile_definitions(PHIX_REAL_FLOAT)
    message(STATUS "PhiX: FLOAT precision build (Real = float)")
elseif(NOT PHIX_PRECISION STREQUAL "DOUBLE")
    message(FATAL_ERROR "PHIX_PRECISION must be DOUBLE or FLOAT (got '${PHIX_PRECISION}')")
endif()
if(PHIX_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()
if(PHIX_ENABLE_NVTX)
    add_compile_definitions(PHIX_ENABLE_NVTX)
endif()

# ---------------------------------------------------------------------------
# C++ / CUDA standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
# CMAKE_CUDA_STANDARD 17 requires CMake 3.18+; set the flag directly for 3.16 compatibility
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
endif()

# ---------------------------------------------------------------------------
# CUDA architecture
# Auto-detected from the local GPU (nvidia-smi) by default — a plain
# `cmake ..` produces binaries that run on the machine that built them.
# Override at configure time, e.g. to build for a different deployment GPU:
#   cmake -DPHIX_CUDA_ARCH=86 -DPHIX_ALLOW_ARCH_MISMATCH=ON ..
# ---------------------------------------------------------------------------
set(PHIX_CUDA_ARCH "auto" CACHE STRING
    "CUDA compute capability: 'auto' (probe the local GPU) or e.g. 75, 86, 120")
option(PHIX_ALLOW_ARCH_MISMATCH
    "Permit PHIX_CUDA_ARCH to differ from the local GPU (cross-compiling)" OFF)

phix_resolve_cuda_arch(PHIX_ARCH_RESOLVED)

# CMAKE_CUDA_ARCHITECTURES requires CMake 3.18+; fall back to -arch flag for 3.16 compatibility
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set(CMAKE_CUDA_ARCHITECTURES ${PHIX_ARCH_RESOLVED})
else()
    list(LENGTH PHIX_ARCH_RESOLVED _phix_n_arch)
    if(_phix_n_arch GREATER 1)
        message(FATAL_ERROR "PhiX: multi-architecture builds require CMake >= 3.18")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${PHIX_ARCH_RESOLVED}")
endif()

# ---------------------------------------------------------------------------
# Extended lambdas — required for [] __device__ (...){} in Equation DSL
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# cuFFT (spectral elasticity / future spectral solvers)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# nlohmann/json (header-only) — IO/ConfigFile.h.  Preference order:
#   1. an installed CMake package        2. a bare header on the include path
#   3. download the release tarball      (PHIX_FETCH_JSON=OFF to forbid)
# ---------------------------------------------------------------------------
option(PHIX_FETCH_JSON "Download nlohmann/json when it is not installed" ON)

find_package(nlohmann_json 3.0 QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "PhiX: nlohmann/json ${nlohmann_json_VERSION} (installed package)")
    set(PHIX_JSON_TARGET nlohmann_json::nlohmann_json)
else()
    find_path(PHIX_JSON_INCLUDE_DIR nlohmann/json.hpp)
    if(PHIX_JSON_INCLUDE_DIR)
        message(STATUS "PhiX: nlohmann/json header at ${PHIX_JSON_INCLUDE_DIR}")
        add_library(phix_json INTERFACE)
        target_include_directories(phix_json INTERFACE ${PHIX_JSON_INCLUDE_DIR})
        set(PHIX_JSON_TARGET phix_json)
    elseif(PHIX_FETCH_JSON)
        message(STATUS "PhiX: nlohmann/json not installed — downloading v3.11.3")
        include(FetchContent)
        FetchContent_Declare(nlohmann_json
            URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
            URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
        FetchContent_MakeAvailable(nlohmann_json)
        set(PHIX_JSON_TARGET nlohmann_json::nlohmann_json)
    else()
        message(FATAL_ERROR
            "PhiX: nlohmann/json not found (IO/ConfigFile.h needs it).\n"
            "  Install it:  sudo apt install nlohmann-json3-dev\n"
            "               conda install -c conda-forge nlohmann_json\n"
            "  or drop the single header on the include path, or configure\n"
            "  with -DPHIX_FETCH_JSON=ON to download it.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Build-time version stamp (core/Version.h).  cmake/PhixVersion.cmake
# regenerates build/generated/phix_version.h on every build (newest
# changelog/v* version + git SHA, "-dirty" when tracked files have
# uncommitted changes); the header is only rewritten when its content
# changes, so incremental builds stay incremental.
# ---------------------------------------------------------------------------
set(PHIX_VERSION_HEADER "${CMAKE_BINARY_DIR}/generated/phix_version.h")
add_custom_target(phix_version_stamp
    COMMAND ${CMAKE_COMMAND}
        -DPHIX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DPHIX_VERSION_OUT=${PHIX_VERSION_HEADER}
        -P ${CMAKE_SOURCE_DIR}/cmake/PhixVersion.cmake
    BYPRODUCTS ${PHIX_VERSION_HEADER}
    VERBATIM)

# ---------------------------------------------------------------------------
# PhiX core library  (static)
# ---------------------------------------------------------------------------
add_library(phix STATIC
    src/core/Check.cu
    src/core/Version.cpp
    src/mesh/Patch.cpp
    src/mesh/Mesh.cpp
    src/field/FieldLayout.cpp
    src/field/ScalarField.cu
    src/field/VectorField.cu
    src/field/FaceField.cu
    src/field/GibbsSimplex.cu
    src/field/IndexField.cu
    src/field/NoiseGenerator.cu
    src/field/Reduce.cu
    src/scheme/Schemes.cu
    src/scheme/SchemeCatalog.cu
    src/operators/Laplacian.cu
    src/operators/Gradient.cu
    src/operators/GradSq.cu
    src/operators/Advection.cu
    src/operators/Anisotropy.cu
    src/operators/FaceOps.cu
    src/boundary/Boundary.cu
    src/boundary/BCFactory.cpp
    src/numerics/Symbolic.cu
    src/numerics/SymbolicBackend.cu
    src/numerics/Composite.cu
    src/numerics/Preparation.cu
    src/numerics/Expression.cu
    src/numerics/System.cu
    src/equation/Equation.cu
    src/equation/EvalPlan.cu
    src/equation/Expr.cu
    src/equation/EquationSystem.cu
    src/equation/VectorEquation.cu
    src/solver/Solver.cu
    src/solver/LinearSolver.cu
    src/solver/Preconditioner.cu
    src/solver/Multigrid.cu
    src/solver/SemiImplicitSolver.cu
    src/solver/VectorSolver.cu
    src/solver/MovingFrame.cu
    src/IO/ConfigFile.cpp
    src/IO/FieldIO.cpp
    src/IO/OutputWriter.cpp
    src/IO/PFHubWriter.cpp
    src/material/FreeEnergyTable.cu
    src/material/KKS.cu
    src/material/KKSAntiTrapping.cu
    src/material/BinaryAlloy.cu
    src/lbm/LBM.cu
    src/diagnostics/Interface.cu
    src/mechanics/ElasticityFFT.cu
    src/field/DomainMask.cu
    src/field/MeshMap.cu
)

target_include_directories(phix PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
target_include_directories(phix PRIVATE "${CMAKE_BINARY_DIR}/generated")
add_dependencies(phix phix_version_stamp)

target_link_libraries(phix PUBLIC CUDA::cufft CUDA::nvrtc CUDA::cuda_driver ${PHIX_JSON_TARGET})

# Ensure CUDA device-code linking is enabled for the library
set_target_properties(phix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------------------------
# Module tests + convergence suite (DOUBLE builds only — tolerances assume
# double precision); float smoke tests replace them in FLOAT builds
# ---------------------------------------------------------------------------
if(PHIX_PRECISION STREQUAL "DOUBLE")
    add_subdirectory(test/moduleTest)
    add_subdirectory(test/convergence)
else()
    add_subdirectory(test/floatSmoke)
endif()

# ---------------------------------------------------------------------------
# Integration tests (both precisions)
# ---------------------------------------------------------------------------
add_subdirectory(test/integration/preparation2d)

# ---------------------------------------------------------------------------
# Performance benchmarks (both precisions)
# ---------------------------------------------------------------------------
add_subdirectory(test/benchmark)

# ---------------------------------------------------------------------------
# Tutorials
# ---------------------------------------------------------------------------
add_subdirectory(tutorials/quickstart)
add_subdirectory(tutorials/numerics2d)
add_subdirectory(tutorials/equations2d)

# ---------------------------------------------------------------------------
# Sample tests + Applications (DOUBLE builds only — solver sources use
# double-typed host code and double-calibrated tolerances)
# ---------------------------------------------------------------------------
if(PHIX_PRECISION STREQUAL "DOUBLE")

add_subdirectory(test/sampleTest)

# Solvers used by the paper: PFHub benchmarks 1, 2, 4, 5, 6 (Cahn-Hillard_double-well,
# CH_4AC_Ostwald, CH_elastic_precipitate, LBM_channel, CH_Poisson_electro), the
# hand-assembled dendrite solver superseded in Sec. 5.3 (dendrite_growth/2D), the
# three-phase model behind the fusion benchmark (MPF_AC_DW).
add_subdirectory("applications/solvers/dendrite_growth/2D")
add_subdirectory("applications/solvers/MPF_AC_DW")
add_subdirectory("applications/solvers/Cahn-Hillard_double-well/2D")
add_subdirectory("applications/solvers/CH_4AC_Ostwald/2D")
add_subdirectory("applications/solvers/CH_elastic_precipitate/2D")
add_subdirectory("applications/solvers/LBM_channel/2D")
add_subdirectory("applications/solvers/CH_Poisson_electro/2D")

endif()
