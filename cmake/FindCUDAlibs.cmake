#.rst:
# FindCUDAlibs
# -------------
#
# Find CUDA include dirs and libraries
#
# Use this module by invoking find_package with the form::
#
#   find_package(CUDAlibs
#     [REQUIRED]             # Fail with error if LAPACKE is not found
#     [COMPONENTS <comp>...] # List of components to look for
#     )                      # Not specifying any components is equivalent to
#                              looking for the BASIC libraries
#
# Valid names for COMPONENTS libraries are::
#
#   ALL         - Find all libraries
#   BASIC       - Equivalent to CUDART;CUBLAS;CUFFT
#
#   CUDART      - CUDA RT library
#   CUBLAS      - CUDA BLAS library
#   CUFFT       - CUDA FFT library
#   CUFFTW      - CUDA FFTW library
#   CUPTI       - CUDA Profiling Tools Interface library.
#   CURAND      - CUDA Random Number Generation library.
#   CUSOLVER    - CUDA Direct Solver library.
#   CUSPARSE    - CUDA Sparse Matrix library.
#   NPP         - NVIDIA Performance Primitives lib.
#   NPPC        - NVIDIA Performance Primitives lib (core).
#   NPPI        - NVIDIA Performance Primitives lib (image processing).
#   NPPIAL      - NVIDIA Performance Primitives lib (image processing).
#   NPPICC      - NVIDIA Performance Primitives lib (image processing).
#   NPPICOM     - NVIDIA Performance Primitives lib (image processing).
#   NPPIDEI     - NVIDIA Performance Primitives lib (image processing).
#   NPPIF       - NVIDIA Performance Primitives lib (image processing).
#   NPPIG       - NVIDIA Performance Primitives lib (image processing).
#   NPPIM       - NVIDIA Performance Primitives lib (image processing).
#   NPPIST      - NVIDIA Performance Primitives lib (image processing).
#   NPPISU      - NVIDIA Performance Primitives lib (image processing).
#   NPPITC      - NVIDIA Performance Primitives lib (image processing).
#   NPPS        - NVIDIA Performance Primitives lib (signal processing).
#   NVBLAS      - NVIDIA BLAS library
#
#  Not specifying COMPONENTS is identical to choosing ALL
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   CUDA_TOOLKIT_ROOT_DIR
#   CUDA_ROOT
#
# The following :prop_tgt:`IMPORTED` targets are defined if required::
#
#   CUDAlibs::CUDART         - Imported target for the CUDA RT library
#   CUDAlibs::CUBLAS         - Imported target for the CUDA cublas library
#   CUDAlibs::CUFFT          - Imported target for the CUDA cufft library
#   CUDAlibs::CUFFTW         - Imported target for the CUDA cufftw library
#   CUDAlibs::CUPTI          - Imported target for the CUDA cupti library
#   CUDAlibs::CURAND         - Imported target for the CUDA curand library
#   CUDAlibs::CUSOLVER       - Imported target for the CUDA cusolver library
#   CUDAlibs::CUSPARSE       - Imported target for the CUDA cusparse library
#   CUDAlibs::NPP            - Imported target for the CUDA npp library
#   CUDAlibs::NPPC           - Imported target for the CUDA nppc library
#   CUDAlibs::NPPI           - Imported target for the CUDA nppi library
#   CUDAlibs::NPPIAL         - Imported target for the CUDA nppial library
#   CUDAlibs::NPPICC         - Imported target for the CUDA nppicc library
#   CUDAlibs::NPPICOM        - Imported target for the CUDA nppicom library
#   CUDAlibs::NPPIDEI        - Imported target for the CUDA nppidei library
#   CUDAlibs::NPPIF          - Imported target for the CUDA nppif library
#   CUDAlibs::NPPIG          - Imported target for the CUDA nppig library
#   CUDAlibs::NPPIM          - Imported target for the CUDA nppim library
#   CUDAlibs::NPPIST         - Imported target for the CUDA nppist library
#   CUDAlibs::NPPISU         - Imported target for the CUDA nppisu library
#   CUDAlibs::NPPITC         - Imported target for the CUDA nppitc library
#   CUDAlibs::NPPS           - Imported target for the CUDA npps library
#   CUDAlibs::NVBLAS         - Imported target for the CUDA nvblas library
#

# ==============================================================================
# Copyright 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ==============================================================================

if(CMAKE_VERSION VERSION_LESS 3.8)
  message(FATAL_ERROR "Cannot use find_package(CUDAlibs ...) with CMake < 3.8! Use find_package(CUDA) instead.")
endif()

# ------------------------------------------------------------------------------

set(_cuda_root_dir_hint)
foreach(_dir ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  get_filename_component(_dirname "${_dir}" DIRECTORY)
  list(APPEND _cuda_root_dir_hint ${_dirname})
endforeach()

set(CUDAlibs_SEARCH_PATHS
  ${_cuda_root_dir_hint}
  ${CUDA_ROOT}
  $ENV{CUDA_ROOT}
  ${CUDA_TOOLKIT_ROOT_DIR}
  $ENV{CUDA_TOOLKIT_ROOT_DIR}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local
  /usr/local/cuda
  /opt
  /opt/local
  )

set(PATH_SUFFIXES_LIST lib/x64 lib64 libx64 lib lib/Win32 lib libWin32)

if(WIN32)
  set(_root_dir "C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA")
  file(GLOB _children RELATIVE ${_root_dir} ${_root_dir}/*)
  list(SORT _children)
  list(REVERSE _children)
  foreach(_child ${_children})
    if(IS_DIRECTORY ${_root_dir}/${_child} AND ${_child} MATCH "v[0-9]+.[0-9]")
      list(APPEND CUDAlibs_SEARCH_PATHS ${_root_dir}/${child})
    endif()
  endforeach()
endif()

# ==============================================================================
# Prepare some helper variables

set(CUDAlibs_REQUIRED_VARS)

# ==============================================================================

macro(cuda_find_library component)
  string(TOLOWER ${COMPONENT} libnames)

  find_library(CUDAlibs_${component}_LIB
    NAMES ${libnames}
    PATHS ${CUDAlibs_SEARCH_PATHS}
    PATH_SUFFIXES ${PATH_SUFFIXES_LIST}
    NO_DEFAULT_PATH)

  if(CUDAlibs_${component}_LIB)
    set(CUDAlibs_${component}_LIB_FOUND 1)
  endif()
  list(APPEND CUDAlibs_REQUIRED_VARS "CUDAlibs_${component}_LIB")
  
  if(CUDAlibs_${component}_LIB_FOUND)
    set(CUDAlibs_${component}_FOUND 1)
  else()
    set(CUDAlibs_${component}_FOUND 0)
  endif()
endmacro()

# ------------------------------------------------------------------------------

set(CUDAlibs_ALL_LIBS "CUDART;CUBLAS;CUFFT;CUFFTW;CUPTI;CURAND;CUSOLVER;CUSPARSE;NPP;NPPC;NPPI;NPPIAL;NPPICC;NPPICOM;NPPIDEI;NPPIF;NPPIG;NPPIM;NPPIST;NPPISU;NPPITC;NPPS;NVBLAS")

# List dependent libraries for relevant targets
set(CUDAlibs_DEP_CUFFTW "CUFFT")
set(CUDAlibs_DEP_NPPIAL "NPPC")
set(CUDAlibs_DEP_NPPICC "NPPC")
set(CUDAlibs_DEP_NPPICOM "NPPC")
set(CUDAlibs_DEP_NPPIDEI "NPPC")
set(CUDAlibs_DEP_NPPIF "NPPC")
set(CUDAlibs_DEP_NPPIG "NPPC")
set(CUDAlibs_DEP_NPPIM "NPPC")
set(CUDAlibs_DEP_NPPIST "NPPC")
set(CUDAlibs_DEP_NPPISU "NPPC")
set(CUDAlibs_DEP_NPPITC "NPPC")
set(CUDAlibs_DEP_NPPS "NPPC")
set(CUDAlibs_DEP_NVBLAS "CUBLAS")

# ------------------------------------------------------------------------------

if(NOT CUDAlibs_FIND_COMPONENTS OR CUDAlibs_FIND_COMPONENTS STREQUAL "BASIC")
  set(CUDAlibs_FIND_COMPONENTS "CUDART;CUBLAS;CUFFT")
elseif(CUDAlibs_FIND_COMPONENTS STREQUAL "ALL")
  set(CUDAlibs_FIND_COMPONENTS ${CUDAlibs_ALL_LIBS})
endif()

foreach(COMPONENT ${CUDAlibs_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)

  if (${UPPERCOMPONENT} IN_LIST CUDAlibs_ALL_LIBS)
    cuda_find_library(${UPPERCOMPONENT})
    foreach(_dep "${CUDAlibs_DEP_${UPPERCOMPONENT}}")
      if(NOT "${_dep}" STREQUAL "")
	cuda_find_library(${_dep})
      endif()
    endforeach()
  else()
    message(FATAL_ERROR "Unknown component: ${COMPONENT}")
  endif()

  mark_as_advanced(
    CUDAlibs_${UPPERCOMPONENT}_LIB
    CUDAlibs_${UPPERCOMPONENT}_INCLUDE_DIR)
endforeach()

# ==============================================================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDAlibs
  FOUND_VAR CUDAlibs_FOUND
  REQUIRED_VARS ${CUDAlibs_REQUIRED_VARS}
  HANDLE_COMPONENTS)

# ==============================================================================

if(CUDAlibs_FOUND)
  # Inspired by FindBoost.cmake
  foreach(COMPONENT ${CUDAlibs_FIND_COMPONENTS})
    string(TOUPPER ${COMPONENT} UPPERCOMPONENT)

    if(NOT TARGET CUDAlibs::${UPPERCOMPONENT} AND CUDAlibs_${UPPERCOMPONENT}_FOUND)
      get_filename_component(LIB_EXT "${CUDAlibs_${UPPERCOMPONENT}_LIB}" EXT)
      if(LIB_EXT STREQUAL ".a" OR LIB_EXT STREQUAL ".lib")
        set(LIB_TYPE STATIC)
      else()
        set(LIB_TYPE SHARED)
      endif()

      add_library(CUDAlibs::${UPPERCOMPONENT} ${LIB_TYPE} IMPORTED GLOBAL)
      set_target_properties(CUDAlibs::${UPPERCOMPONENT} PROPERTIES
	IMPORTED_LOCATION "${CUDAlibs_${UPPERCOMPONENT}_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

      # ------------------------------------------------------------------------
      
      set(_dependent_libraries)
      foreach(_dep "${CUDAlibs_DEP_${UPPERCOMPONENT}}")
	list(APPEND _dependent_libraries ${_dep})
      endforeach()

      if(NOT "${_dependent_libraries}" STREQUAL "")
	message(STATUS "Adding dependencies: ${_dependent_libraries}")
	set_target_properties(CUDAlibs::${UPPERCOMPONENT} PROPERTIES
          INTERFACE_LINK_LIBRARIES "${_dependent_libraries}")
      endif()
    endif()
  endforeach()

  # ----------------------------------------------------------------------------

  if(NOT CUDAlibs_FIND_QUIETLY)
    message(STATUS "Found CUDAlibs and defined the following imported targets:")
    foreach(_comp ${CUDAlibs_FIND_COMPONENTS})
      message(STATUS "  - CUDAlibs::${_comp}")
    endforeach()
  endif()
endif()

# ==============================================================================

mark_as_advanced(
  CUDAlibs_FOUND
  )

