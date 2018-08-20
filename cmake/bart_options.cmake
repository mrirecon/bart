# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>

option(BART_CREATE_PYTHON_MODULE "Compile the pyBART Python module" OFF)
option(BART_DISABLE_PNG "Disable the use of the PNG library" OFF)
option(BART_ENABLE_MEM_CFL "Enable the use of in-memory CFL files" OFF)
option(BART_FPIC "Compile using position-independent code" OFF)
option(BART_FFTWTHREADS "Enable multi-threaded FFTW libraries" ON)
option(BART_GENERATE_DOC "Automatically generate documentation after building BART" OFF)
option(BART_LOCAL_FFTW "Use a modified version of FFTW where all functions are prefixed with local_ (useful for static FFTW linking)" OFF)
option(BART_MEMONLY_CFL "Use only in-memory CFL files" OFF)
option(BART_MATLAB "Specify if the optional matlab programs should be built" OFF)
option(BART_NO_LAPACKE "Compile without LAPACKE installed on the system" OFF)
option(BART_REDEFINE_PRINTF_FOR_TRACE "Replace debug_print* functions with macros for better log tracing (e.g. with external loggers)" OFF)

find_package(ISMRMRD QUIET CONFIG) ## if you can find ISMRMRD by default, then default configuration is ON
option(BART_ISMRMRD "Use external ISMRMRD package for reading/writing" ${ISMRMRD_FOUND})

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
if(CMAKE_VERSION VERSION_GREATER 3.9.99)
  cmake_dependent_option(USE_CUDA_NATIVE "Use native CMake support for CUDA (instead of relying on the FindCUDA module)" ON
    "USE_CUDA" OFF)
endif()
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
  set(PNG_TGT "")
else()
  find_package(PNG REQUIRED)
  if(CMAKE_VERSION VERSION_LESS 3.5)
    add_library(PNG::PNG SHARED IMPORTED GLOBAL)
    set_target_properties(PNG::PNG
      PROPERTIES
      IMPORTED_LOCATION "${PNG_LIBRARY}"
      INTERFACE_DEFINITIONS "${PNG_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${PNG_LIBRARIES}"
      )
  endif()
  set(PNG_TGT "PNG::PNG")
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
  # manually add the pthread library in case one is using static fftw libraries...
  set(CMAKE_THREAD_PREFER_PTHREAD ON)
  find_package(Threads REQUIRED)
  
  add_definitions(-DUSE_LOCAL_FFTW)
  
  add_library(FFTW::FFTWF STATIC IMPORTED GLOBAL)
  set_target_properties(FFTW::FFTWF
    PROPERTIES
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src"
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/libfftw3f.a")

  add_library(FFTW::FFTWF_MT STATIC IMPORTED GLOBAL)  
  set_target_properties(FFTW::FFTWF_MT
    PROPERTIES
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src"
    INTERFACE_LINK_LIBRARIES "Threads::Threads"
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/libfftw3f_threads.a")
else()
  find_package(FFTW 3 REQUIRED COMPONENTS FFTWF FFTWF_MT)
endif()
set(FFTW_TGT "FFTW::FFTWF;FFTW::FFTWF_MT")

if(BART_FFTWTHREADS)
  add_definitions(-DFFTWTHREADS)
else()
  # remove the threaded library when not using threads
  list(REMOVE_ITEM FFTW_TGT "FFTW::FFTWF_MT")
endif()

# --------------------------------------

if(BART_NO_LAPACKE)
  set(ATLAS_NO_LAPACKE ON)
  set(OpenBLAS_NO_LAPACKE ON)
  set(BLAS_NO_LAPACKE ON)
  add_definitions(-DNOLAPACKE)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/src/lapacke)
  message(STATUS "Compiling with NOLAPACKE")
endif()

# --------------------------------------

if(BART_REDEFINE_PRINTF_FOR_TRACE)
  add_definitions(-DREDEFINE_PRINTF_FOR_TRACE)
  message(STATUS "Redefining debug_print* functions as macros for better log tracing")
endif(BART_REDEFINE_PRINTF_FOR_TRACE)

# --------------------------------------

if(BART_ISMRMRD)
  find_package(ISMRMRD REQUIRED)
endif()

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
  # Actually, CMake >= 3.8 supports CUDA natively, but at least on my machine,
  # the command line generated by CMake 3.8.X & 3.9.X contains
  # -isystem /usr/include which causes the compilation to fail.
  # So... only use the new feature for CMake >= 3.10
  if(CMAKE_VERSION VERSION_LESS 3.10)
    set(BART_USE_NATIVE_CUDA FALSE)
  else()
    if(USE_CUDA_NATIVE)
      set(BART_USE_NATIVE_CUDA TRUE)
    else()      
      set(BART_USE_NATIVE_CUDA FALSE)
    endif()
  endif()
  
  if(NOT BART_USE_NATIVE_CUDA)
    find_package(CUDA REQUIRED)
    
    add_definitions(-DUSE_CUDA)
    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/src)
    # set(CUDA_NVCC_FLAGS "-DUSE_CUDA;-Xcompiler;-fPIC;-Xcompiler;-fopenmp;-O3;-arch=sm_20;-m64;-ccbin ${CMAKE_C_COMPILER}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;-DUSE_CUDA;-Xcompiler;-fPIC;-Xcompiler;-fopenmp;-O3;${GPUARCH_FLAGS};-ccbin ${CMAKE_C_COMPILER}")
    macro(bart_add_object_library target_name)
      add_library(${target_name} OBJECT ${ARGN})
      target_include_directories(${target_name} PRIVATE "${CUDA_INCLUDE_DIRS}")
    endmacro()
    macro(bart_add_executable target_name)
      set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC) # was introduced in CMake 3.9
      CUDA_ADD_EXECUTABLE(${target_name} ${ARGN})
      target_link_libraries(${target_name} PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    endmacro()
    macro(bart_add_library target_name)
      set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC) # was introduced in CMake 3.9
      CUDA_ADD_LIBRARY(${target_name} ${ARGN})
      target_link_libraries(${target_name} PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    endmacro()  
  else()
    enable_language(CUDA)
    set (CMAKE_CUDA_STANDARD 11)
    set (CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAlibs REQUIRED COMPONENTS CUBLAS CUFFT)
    
    macro(bart_add_object_library target_name)
      add_library(${target_name} OBJECT ${ARGN})
      target_compile_definitions(${target_name} PRIVATE -DUSE_CUDA)
      target_include_directories(${target_name} PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    endmacro()
    macro(bart_add_executable target_name)
      add_executable(${target_name} ${ARGN})
      target_compile_definitions(${target_name} PRIVATE -DUSE_CUDA)
      target_link_libraries(${target_name} PUBLIC "CUDAlibs::CUBLAS;CUDAlibs::CUFFT")
      set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endmacro()
    macro(bart_add_library target_name)
      add_library(${target_name} ${ARGN})
      target_compile_definitions(${target_name} PRIVATE -DUSE_CUDA)
      target_link_libraries(${target_name} PUBLIC "CUDAlibs::CUBLAS;CUDAlibs::CUFFT")
      set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endmacro()
  endif()
else() # NOT USE_CUDA
  macro(bart_add_object_library target_name)
    add_library(${target_name} OBJECT ${ARGN})
  endmacro()
  macro(bart_add_executable target_name)
    add_executable(${target_name} ${ARGN})
  endmacro()
  macro(bart_add_library target_name)
    add_library(${target_name} ${ARGN})
  endmacro()
endif()

# ------------------------------------------------------------------------------

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")

    if(USE_CUDA)
      get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
      if(CMAKE_VERSION VERSION_LESS 3.3)
	list (FIND languages "CXX" _index)
	if (${_index} GREATER -1)
	  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
	else(DEFINED CMAKE_CUDA_FLAGS)
	  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_C_FLAGS}")
	endif()
      else()
	if("CXX" IN_LIST languages)
	  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
	else(DEFINED CMAKE_CUDA_FLAGS)
	  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_C_FLAGS}")
	endif()
      endif() # CXX in languages
    endif()
  endif() # OpenMP found
endif() # USE_OPENMP

# ==============================================================================

