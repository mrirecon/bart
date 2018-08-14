# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>

option(BART_CREATE_PYTHON_MODULE "Compile the pyBART Python module" OFF)
option(BART_DISABLE_PNG "Disable the use of the PNG library" OFF)
option(BART_ENABLE_MEM_CFL "Enable the use of in-memory CFL files" OFF)
option(BART_FPIC "Compile using position-independent code" OFF)
option(BART_FFTWTHREADS "Enable multi-threaded FFTW libraries" ON)
option(BART_GENERATE_DOC "Automatically generate documentation after building BART" ON)
option(BART_LOCAL_FFTW "Use a modified version of FFTW where all functions are prefixed with local_ (useful for static FFTW linking)" OFF)
option(BART_MEMONLY_CFL "Use only in-memory CFL files" OFF)
option(BART_MATLAB "Specify if the optional matlab programs should be built" OFF)
option(BART_NO_LAPACKE "Compile without LAPACKE installed on the system" OFF)
option(BART_REDEFINE_PRINTF_FOR_TRACE "Replace debug_print* functions with macros for better log tracing (e.g. with external loggers)" OFF)

# ------------------------------------------------------------------------------

option(BART_LOG_BACKEND "Enable delegating all outputs to external logging backend" OFF)
include(CMakeDependentOption)
cmake_dependent_option(BART_LOG_SIEMENS_BACKEND "Use the Siemens logging backend" OFF
  "BART_LOG_BACKEND" OFF)
cmake_dependent_option(BART_LOG_ORCHESTRA_BACKEND "Use the Orchestra logging backend" OFF
  "BART_LOG_BACKEND" OFF)
cmake_dependent_option(BART_LOG_GADGETRON_BACKEND "Use the Gadgetron logging backend" OFF
  "BART_LOG_BACKEND" OFF)

# ------------------------------------------------------------------------------

option(USE_CUDA "Provide support for CUDA processing" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)

# ==============================================================================

if(BART_ENABLE_MEM_CFL OR USE_CUDA OR BART_LOG_BACKEND)
  enable_language(CXX)
endif()

# ------------------------------------------------------------------------------

if(BART_CREATE_PYTHON_MODULE)
  message(STATUS "Generating the pyBART Python module")
endif()

# --------------------------------------

if(BART_DISABLE_PNG)
  set(PNG_DEFINITIONS "")
  set(PNG_INCLUDE_DIRS "")
else()
  find_package(PNG REQUIRED)
  add_definitions(${PNG_DEFINITIONS})
  include_directories(${PNG_INCLUDE_DIRS})
endif()

# --------------------------------------

if(BART_MEMONLY_CFL)
  message(STATUS "Forcing BART to use only in-memory CFLs")
  if(NOT BART_ENABLE_MEM_CFL)
    message(STATUS "Setting BART_ENABLE_MEM_CFL = ON")
    set(BART_ENABLE_MEM_CFL ON)
  endif(NOT BART_ENABLE_MEM_CFL)
  add_definitions(-DMEMONLY_CFL)
endif(BART_MEMONLY_CFL)

if(BART_ENABLE_MEM_CFL)
  message(STATUS "Using in-memory CFL files")
  add_definitions(-DUSE_MEM_CFL)
endif(BART_ENABLE_MEM_CFL)

# --------------------------------------

if(BART_FPIC)
  message(STATUS "ENABLED compilation with -fPIC")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# --------------------------------------

if(BART_GENERATE_DOC)
  message(STATUS "Automatically generating documentation after building BART")
endif()

# --------------------------------------

if(BART_LOCAL_FFTW)
  set(FFTWF_FOUND 1)
  set(FFTW_INCLUDE ${CURRENT_LIST_DIR/src})
  set(FFTWF_THREADS_LIB "${CMAKE_CURRENT_LIST_DIR}/lib/libfftw3f_threads.a")
  set(FFTWF_LIB "${CMAKE_CURRENT_LIST_DIR}/lib/libfftw3f.a" ${FFTWF_THREADS_LIB})
  set(FFTWF_LIBRARIES ${FFTWF_LIB})
  add_definitions(-DUSE_LOCAL_FFTW)
else(BART_LOCAL_FFTW)
  set(USE_FFTWF ON) # Only find single precision fftw
  find_package(FFTW REQUIRED)
endif(BART_LOCAL_FFTW)
message(STATUS "FFTWF_LIBRARIES: ${FFTWF_LIBRARIES}")

if(BART_FFTWTHREADS)
  # manually add the pthread library in case one is using static fftw libraries...
  find_library(PTHREAD_LIB pthread)
  list(APPEND FFTWF_LIBRARIES ${PTHREAD_LIB})
  add_definitions(-DFFTWTHREADS)
else(BART_FFTWTHREADS)
  # remove the threaded library when not using threads
  list(REMOVE_ITEM FFTWF_LIBRARIES ${FFTWF_THREADS_LIB})
endif(BART_FFTWTHREADS)

# --------------------------------------

if(BART_NO_LAPACKE)
  set(ATLAS_NO_LAPACKE ON)
  set(OpenBLAS_NO_LAPACKE ON)
  set(BLAS_NO_LAPACKE ON)
  add_definitions(-DNOLAPACKE)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/src/lapacke)
  message(STATUS "Compiling with NOLAPACKE")
endif(BART_NO_LAPACKE)

# --------------------------------------

if(BART_REDEFINE_PRINTF_FOR_TRACE)
  add_definitions(-DREDEFINE_PRINTF_FOR_TRACE)
  message(STATUS "Redefining debug_print* functions as macros for better log tracing")
endif(BART_REDEFINE_PRINTF_FOR_TRACE)

# ==============================================================================

if(BART_LOG_SIEMENS_BACKEND)
  message(STATUS "Delegating all outputs to the Siemens logging backend")
endif(BART_LOG_SIEMENS_BACKEND)

if(BART_LOG_ORCHESTRA_BACKEND)
  message(STATUS "Delegating all outputs to the Orchestra logging backend")
endif(BART_LOG_ORCHESTRA_BACKEND)

if(BART_LOG_GADGETRON_BACKEND)
  message(STATUS "Delegating all outputs to the Gadgetron logging backend")
endif(BART_LOG_GADGETRON_BACKEND)

# ==============================================================================

if(USE_CUDA)
  find_package(CUDA)
  add_definitions(-DUSE_CUDA)
  CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/src)
  # set(CUDA_NVCC_FLAGS "-DUSE_CUDA;-Xcompiler;-fPIC;-Xcompiler;-fopenmp;-O3;-arch=sm_20;-m64;-ccbin ${CMAKE_C_COMPILER}")
  set(CUDA_NVCC_FLAGS "-DUSE_CUDA;-Xcompiler;-fPIC;-Xcompiler;-fopenmp;-O3;${GPUARCH_FLAGS};-ccbin ${CMAKE_C_COMPILER}")
  macro(bart_add_executable target_name)
    CUDA_ADD_EXECUTABLE(${target_name} ${ARGN})
    CUDA_ADD_CUFFT_TO_TARGET(${target_name})
    CUDA_ADD_CUBLAS_TO_TARGET(${target_name})
    target_link_libraries(${target_name} ${CUDA_LIBRARIES})
  endmacro(bart_add_executable)
  macro(bart_add_library target_name)
    CUDA_ADD_LIBRARY(${target_name} ${ARGN})
    CUDA_ADD_CUFFT_TO_TARGET(${target_name})
    CUDA_ADD_CUBLAS_TO_TARGET(${target_name})
    target_link_libraries(${target_name} ${CUDA_LIBRARIES})
  endmacro(bart_add_library)  
else()
  macro(bart_add_executable target_name)
    add_executable(${target_name} ${ARGN})
  endmacro(bart_add_executable)
  macro(bart_add_library target_name)
    add_library(${target_name} ${ARGN})
  endmacro(bart_add_library)
endif(USE_CUDA)

# ------------------------------------------------------------------------------

if(USE_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif(USE_OPENMP)

# ==============================================================================

