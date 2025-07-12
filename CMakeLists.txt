# Set the minimum required version of CMake and define the project
cmake_minimum_required(VERSION 3.16)
project(SOFIE_Benchmarks CXX)

# Set the C++ standard to 17, which is required by ONNXRuntime
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find the ROOT Library ---
# This automatically finds your ROOT installation because you sourced thisroot.sh
find_package(ROOT REQUIRED COMPONENTS ROOTTMVASofie)

# --- Find the ONNXRuntime Library ---
# YOU MUST CHANGE THIS PATH to where you unzipped the ONNXRuntime folder
set(ONNXRUNTIME_DIR "/Users/olia/libs/onnxruntime-osx-arm64-1.17.1")

# Find the header files and library files inside that directory
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_DIR}/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib)

# Check if both were found
if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find ONNXRuntime in ${ONNXRUNTIME_DIR}")
endif()

# --- Define Your Executable Programs ---

# 1. Define the executable for the SOFIE test
add_executable(run_sofie_test particlenet/run_inference_particle_net.C)

# Link it with the ROOT libraries it needs
target_link_libraries(run_sofie_test PRIVATE ROOT::ROOTTMVASofie)

# 2. Define the executable for the ONNXRuntime test
add_executable(run_onnx_test particlenet/run_ONNXRuntimeInference_particle_net.cxx)

# Link it with the ONNXRuntime library
target_include_directories(run_onnx_test PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(run_onnx_test PRIVATE ${ONNXRUNTIME_LIBRARY})

add_executable(run_atlas_test atlas-tracking/run_atlas_model.C)
target_link_libraries(run_atlas_test PRIVATE ROOT::ROOTTMVASofie)


# --- Print a message to confirm ---
message(STATUS "Build configured for SOFIE and ONNXRuntime tests.")