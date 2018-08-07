#.rst:
# FindLAPACKE
# -------------
#
# Find LAPACKE include dirs and libraries
#
# Use this module by invoking find_package with the form::
#
#   find_package(Boost
#     [REQUIRED]             # Fail with error if LAPACKE is not found
#     [COMPONENTS <libs>...] # List of libraries to look for
#     )
#
# Valid names for COMPONENTS libraries are::
#
#   ALL                      - Find all libraries
#   LAPACKE_H                - Find the lapacke.h header file
#   LAPACKE                  - Find a LAPACKE library
#   LAPACK                   - Find a LAPACK library
#   CBLAS                    - Find a CBLAS library
#   BLAS                     - Find a BLAS library
#
#  Not specifying COMPONENTS is identical to choosing ALL
#
# This module defines::
#
#   LAPACKE_FOUND            - True if headers and requested libraries were found
#   LAPACKE_INCLUDE_DIRS     - LAPACKE include directories
#   LAPACKE_LIBRARIES        - Boost component libraries to be linked
#
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   LAPACKE_ROOT             - Preferred installation prefix for LAPACKE
#
#
# The following :prop_tgt:`IMPORTED` targets are also defined::
#
#   LAPACKE::LAPACKE         - Imported target for the LAPACKE library
#   LAPACKE::LAPACK          - Imported target for the LAPACK library
#   LAPACKE::CBLAS           - Imported target for the CBLAS library
#   LAPACKE::BLAS            - Imported target for the BLAS library
#

# ==============================================================================
# Copyright 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ==============================================================================

set(LAPACKE_SEARCH_PATHS
  ${LAPACKE_ROOT}
  $ENV{LAPACKE_ROOT}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local/
  /usr/local/opt # homebrew on mac
  /opt
  /opt/local
  /opt/LAPACKE
  )

set(PATH_SUFFIXES_LIST
  lib64
  lib
  lib/x86_64-linux-gnu
  lib32
)

if(APPLE)
  list(APPEND PATH_SUFFIXES_LIST lapack/lib openblas/lib)
endif()

# ==============================================================================
# Prepare some helper variables

set(LAPACKE_INCLUDE_DIRS)
set(LAPACKE_LIBRARIES)
set(LAPACKE_REQUIRED_VARS)
set(LAPACKE_FIND_ALL_COMPONENTS 0)

# ==============================================================================

macro(_find_library_with_header component libnames incnames)
  find_library(LAPACKE_${component}_LIB
    NAMES ${libnames}
    PATHS ${LAPACKE_SEARCH_PATHS}
    PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
  if(LAPACKE_${component}_LIB)    
    set(LAPACKE_${component}_LIB_FOUND 1)
  endif()
  list(APPEND LAPACKE_REQUIRED_VARS "LAPACKE_${component}_LIB")

  # If necessary, look for the header file as well
  if(NOT "${incnames}" STREQUAL "")
    find_path(LAPACKE_${component}_INCLUDE_DIR
      NAMES ${incnames}
      PATHS ${LAPACKE_SEARCH_PATHS}
      PATH_SUFFIXES include lapack/include)
    list(APPEND LAPACKE_REQUIRED_VARS "LAPACKE_${component}_INCLUDE_DIR")
    if(LAPACKE_${component}_LIB)
      set(LAPACKE_${component}_INC_FOUND 1)
    endif()
  else()
    set(LAPACKE_${component}_INC_FOUND 1)
  endif()
  
  if(LAPACKE_${component}_LIB_FOUND AND LAPACKE_${component}_INC_FOUND)
    set(LAPACKE_${component}_FOUND 1)
  else()
    set(LAPACKE_${component}_FOUND 0)
  endif()
endmacro()

# ------------------------------------------------------------------------------

if(NOT LAPACKE_FIND_COMPONENTS OR LAPACKE_FIND_COMPONENTS STREQUAL "ALL")
  set(LAPACKE_FIND_ALL_COMPONENTS 1)
  set(LAPACKE_FIND_COMPONENTS "LAPACKE;LAPACK;CBLAS;BLAS")
endif(NOT LAPACKE_FIND_COMPONENTS OR LAPACKE_FIND_COMPONENTS STREQUAL "ALL")

foreach(COMPONENT ${LAPACKE_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
  if(UPPERCOMPONENT STREQUAL "LAPACKE")
    _find_library_with_header(${UPPERCOMPONENT} lapacke lapacke.h)
  elseif(UPPERCOMPONENT STREQUAL "LAPACKE_H")
    find_path(LAPACKE_${UPPERCOMPONENT}_INCLUDE_DIR
      NAMES lapacke.h
      PATHS ${LAPACKE_SEARCH_PATHS}
      PATH_SUFFIXES include lapack/include)
    list(APPEND LAPACKE_REQUIRED_VARS "LAPACKE_${UPPERCOMPONENT}_INCLUDE_DIR")
    if(LAPACKE_${UPPERCOMPONENT}_LIB)
      set(LAPACKE_${UPPERCOMPONENT}_INC_FOUND 1)
    endif()
  elseif(UPPERCOMPONENT STREQUAL "LAPACK")
    _find_library_with_header(${UPPERCOMPONENT} lapack "")
  elseif(UPPERCOMPONENT STREQUAL "CBLAS")
    _find_library_with_header(${UPPERCOMPONENT} cblas cblas.h)
  elseif(UPPERCOMPONENT STREQUAL "BLAS")
    _find_library_with_header(${UPPERCOMPONENT} blas "")
  else()
    message(FATAL_ERROR "Unknown component: ${COMPONENT}")
  endif()
  mark_as_advanced(
    LAPACKE_${UPPERCOMPONENT}_LIB
    LAPACKE_${UPPERCOMPONENT}_INCLUDE_DIR)
endforeach()

# ==============================================================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE
  FOUND_VAR LAPACKE_FOUND
  REQUIRED_VARS ${LAPACKE_REQUIRED_VARS})

# ==============================================================================

if(LAPACKE_FOUND)
  foreach(COMPONENT ${LAPACKE_FIND_COMPONENTS})
    string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
    list(APPEND LAPACKE_INCLUDE_DIRS ${LAPACKE_${UPPERCOMPONENT}_INCLUDE_DIR})
    list(APPEND LAPACKE_LIBRARIES ${LAPACKE_${UPPERCOMPONENT}_LIB})
  endforeach()
  
  if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR
      "${CMAKE_C_COMPILER_ID}" MATCHES ".*GNU.*" OR
      "${CMAKE_C_COMPILER_ID}" MATCHES ".*Intel.*"
      ) #NOT MSVC
    set(MATH_LIB "m")
    list(APPEND LAPACKE_LIBRARIES m)
  endif()
  
  if(NOT "${LAPACKE_INCLUDE_DIRS}" STREQUAL "")
    list(REMOVE_DUPLICATES LAPACKE_INCLUDE_DIRS)
  endif()
endif()

# ------------------------------------------------------------------------------

set(LAPACKE_IMPORTED_TARGET_LIST)
# Inspired by FindBoost.cmake
foreach(COMPONENT ${LAPACKE_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
  if(NOT TARGET LAPACKE::${UPPERCOMPONENT} AND LAPACKE_${UPPERCOMPONENT}_FOUND)
    add_library(LAPACKE::${UPPERCOMPONENT} UNKNOWN IMPORTED)
    if(LAPACKE_INCLUDE_DIRS)
      set_target_properties(LAPACKE::${UPPERCOMPONENT} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${LAPACKE_${UPPERCOMPONENT}_LIB}")
      set_target_properties(LAPACKE::${UPPERCOMPONENT} PROPERTIES
        IMPORTED_LOCATION "${LAPACKE_${UPPERCOMPONENT}_LIB}")
    endif()
    set_target_properties(LAPACKE::${UPPERCOMPONENT} PROPERTIES
      INTERFACE_LINK_LIBRARIES "${MATH_LIB}")
    list(APPEND LAPACKE_IMPORTED_TARGET_LIST LAPACKE::${UPPERCOMPONENT})
  endif()
endforeach()

# ==============================================================================

if(NOT LAPACKE_FIND_QUIETLY)
  message(STATUS "LAPACKE_FOUND         :${LAPACKE_FOUND}:  - set to true if the library is found")
  message(STATUS "LAPACKE_INCLUDE_DIRS  :${LAPACKE_INCLUDE_DIRS}: - list of required include directories")
  message(STATUS "LAPACKE_LIBRARIES     :${LAPACKE_LIBRARIES}: - list of libraries to be linked")
endif()
