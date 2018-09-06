#.rst:
# FindOpenBLAS
# -------------
#
# Find OpenBLAS include dirs and libraries
#
# Use this module by invoking find_package with the form::
#
#   find_package(OpenBLAS
#     [REQUIRED]             # Fail with error if OpenBLAS is not found
#     )
#
#
# This module defines::
#
#   OpenBLAS_FOUND            - set to true if the library is found
#   OpenBLAS_INCLUDE_DIRS     - list of required include directories
#   OpenBLAS_LIBRARIES        - list of libraries to be linked
#   OpenBLAS_HAS_PARALLEL_LIBRARIES - determine if there are parallel libraries compiled
#   OpenBLAS_PARALLEL_LIBRARIES - list of libraries for parallel implementations
#   OpenBLAS_VERSION_MAJOR    - major version number
#   OpenBLAS_VERSION_MINOR    - minor version number
#   OpenBLAS_VERSION_PATCH    - patch version number
#   OpenBLAS_VERSION_STRING   - version number as a string (ex: "0.2.18")
#
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   OpenBLAS_ROOT             - Preferred installation prefix for OpenBLAS
#   OpenBLAS_DIR              - (for compatibility purposes)
#
#
#   OpenBLAS::OpenBLAS        - Imported target for the OpenBLAS library
#

# ==============================================================================
# Copyright 2016 Hans J. Johnson <hans-johnson@uiowa.edu>
# Copyright 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ==============================================================================

set(OpenBLAS_SEARCH_PATHS
  ${OpenBLAS_ROOT}
  $ENV{OpenBLAS_ROOT}
  ${OpenBLAS_DIR}
  $ENV{OpenBLAS_DIR}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local/
  /usr/local/opt # homebrew on mac
  /opt
  /opt/local
  /opt/OpenBLAS
  )

set(PATH_SUFFIXES_LIST
  lib64
  lib
  lib/x86_64-linux-gnu
  lib32
)

if(APPLE)
  list(APPEND PATH_SUFFIXES_LIST openblas/lib)
endif()

# ==============================================================================
# Prepare some helper variables

set(OpenBLAS_INCLUDE_DIRS)
set(OpenBLAS_LIBRARIES)
set(OpenBLAS_HAS_PARALLEL_LIBRARIES FALSE)

# ==============================================================================

## First try to find OpenBLAS with NO_MODULE,
## As of 20160706 version 0.2.18 there is limited cmake support for OpenBLAS
## that is not as complete as this version, if found, use it
## to identify the OpenBLAS_VERSION_STRING and improve searching.
find_package(OpenBLAS NO_MODULE QUIET)
if(OpenBLAS_VERSION)
  set(OpenBLAS_VERSION_STRING ${OpenBLAS_VERSION})
  unset(OpenBLAS_VERSION) # Use cmake conventional naming
endif()

# ==============================================================================
### First search for headers

find_path(OpenBLAS_CBLAS_INCLUDE_DIR 
  NAMES cblas.h 
  PATHS ${OpenBLAS_SEARCH_PATHS} 
  PATH_SUFFIXES include include/openblas)

# ==============================================================================
### Second, search for libraries

find_library(OpenBLAS_LIB 
  NAMES openblas
  PATHS ${OpenBLAS_SEARCH_PATHS}
  PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

if(OpenBLAS_LIB)
  get_filename_component(OpenBLAS_LIB_DIR i${OpenBLAS_LIB} DIRECTORY)

  include(CheckLibraryExists)
  check_library_exists(${OpenBLAS_LIB} LAPACKE_cheev  "" HAVE_LAPACKE_CHEEV)
  check_library_exists(${OpenBLAS_LIB} LAPACKE_zgesdd "" HAVE_LAPACKE_ZGESDD)

  if(NOT HAVE_LAPACKE_CHEEV OR NOT HAVE_LAPACKE_ZGESDD)
    message(WARNING "OpenBLAS has no LAPACKE symbols. Attempting to look for a LAPACKE library")
    find_package(LAPACKE QUIET COMPONENTS LAPACKE)
    if(LAPACKE_FOUND)
      message(STATUS "Found LAPACKE: ${LAPACKE_LIBRARIES}")
      list(APPEND OpenBLAS_LIBRARIES ${LAPACKE_LIBRARIES})
    else()
      message(WARNING "Unable to find a LAPACKE library. Consider configuring CMake with BART_NO_LAPACKE=ON")
    endif()
  else()
    # Make sure we find lapacke.h
    find_package(LAPACKE QUIET COMPONENTS LAPACKE_H)
  endif()
endif()

## Find the named parallel version of openblas
set(OpenBLAS_SEARCH_VERSIONS ${OpenBLAS_VERSION_STRING} 0.3.2 0.3.1 0.3.0 0.2.19 0.2.18 0.2.17 0.2.16)
list(REMOVE_DUPLICATES OpenBLAS_SEARCH_VERSIONS)
foreach(checkVersion ${OpenBLAS_SEARCH_VERSIONS})
  find_library(OpenBLAS_PARALLEL_LIB 
    NAMES openblasp-r${checkVersion}
    PATHS ${OpenBLAS_LIB_DIR} ${OpenBLAS_SEARCH_PATHS}
    PATH_SUFFIXES ${PATH_SUFFIXES_LIST}
    )
  if(EXISTS ${OpenBLAS_PARALLEL_LIB})
    if(NOT OpenBLAS_VERSION_STRING)
      set(OpenBLAS_VERSION_STRING ${checkVersion})
    endif()
    set(OpenBLAS_HAS_PARALLEL_LIBRARIES ON)
    break()
  endif()
endforeach()

# WARNING: We may not be able to determine the version of some OpenBLAS
set(OpenBLAS_VERSION_MAJOR 0)
set(OpenBLAS_VERSION_MINOR 0)
set(OpenBLAS_VERSION_PATCH 0)
if(OpenBLAS_VERSION_STRING)
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\1" OpenBLAS_VERSION_MAJOR "${OpenBLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\2" OpenBLAS_VERSION_MINOR "${OpenBLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\3" OpenBLAS_VERSION_PATCH "${OpenBLAS_VERSION_STRING}")
endif()

# ==============================================================================
# Checks 'REQUIRED', 'QUIET' and versions.

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS FOUND_VAR OpenBLAS_FOUND
  REQUIRED_VARS 
  OpenBLAS_CBLAS_INCLUDE_DIR
  OpenBLAS_LIB
  LAPACKE_INCLUDE_DIRS
  VERSION_VAR OpenBLAS_VERSION_STRING
)

# ------------------------------------------------------------------------------

if (OpenBLAS_FOUND)
  list(APPEND OpenBLAS_INCLUDE_DIRS
    ${OpenBLAS_CBLAS_INCLUDE_DIR}
    ${LAPACKE_INCLUDE_DIRS})
  list(REMOVE_DUPLICATES OpenBLAS_INCLUDE_DIRS)
  
  if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*GNU.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*Intel.*"
      ) #NOT MSVC
    set(MATH_LIB m)
  endif()
  list(APPEND OpenBLAS_LIBRARIES ${OpenBLAS_LIB} ${MATH_LIB})

  if(OpenBLAS_HAS_PARALLEL_LIBRARIES)
    list(APPEND OpenBLAS_PARALLEL_LIBRARIES ${OpenBLAS_PARALLEL_LIB})
  endif()

  # ----------------------------------------------------------------------------

  if(NOT TARGET OpenBLAS::OpenBLAS)
    get_filename_component(LIB_EXT "${LAPACKE_${UPPERCOMPONENT}_LIB}" EXT)
    if(LIB_EXT STREQUAL ".a" OR LIB_EXT STREQUAL ".lib")
      set(LIB_TYPE STATIC)
    else()
      set(LIB_TYPE SHARED)
    endif()
    add_library(OpenBLAS::OpenBLAS ${LIB_TYPE} IMPORTED GLOBAL)
    set(_tmp_dep_libs "${MATH_LIB};${LAPACKE_LIBRARIES}")
    list(REMOVE_DUPLICATES _tmp_dep_libs)
    set_target_properties(OpenBLAS::OpenBLAS
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${_tmp_dep_libs}")
    
    if(OpenBLAS_HAS_PARALLEL_LIBRARIES)
      set_target_properties(OpenBLAS::OpenBLAS
	PROPERTIES
	IMPORTED_LOCATION "${OpenBLAS_PARALLEL_LIB}")
    else()
      set_target_properties(OpenBLAS::OpenBLAS
	PROPERTIES
	IMPORTED_LOCATION "${OpenBLAS_LIB}")
    endif()
  endif()

  # ----------------------------------------------------------------------------

  if(NOT OpenBLAS_FIND_QUIETLY)
    get_target_property(_lib OpenBLAS::OpenBLAS IMPORTED_LOCATION)
    get_target_property(_dep_libs OpenBLAS::OpenBLAS INTERFACE_LINK_LIBRARIES)
    message(STATUS "Found OpenBLAS and defined the OpenBLAS::OpenBLAS imported target:")
    message(STATUS "  - include:      ${OpenBLAS_INCLUDE_DIRS}")
    message(STATUS "  - library:      ${_lib}")
    message(STATUS "  - dependencies: ${_dep_libs}")
  endif()
endif()

# ==============================================================================

mark_as_advanced(
  OpenBLAS_FOUND
  OpenBLAS_INCLUDE_DIRS
  OpenBLAS_LIBRARIES
  OpenBLAS_HAS_PARALLEL_LIBRARIES
  OpenBLAS_PARALLEL_LIBRARIES
  OpenBLAS_VERSION_MAJOR
  OpenBLAS_VERSION_MINOR
  OpenBLAS_VERSION_PATCH
  OpenBLAS_VERSION_STRING
  )

