#.rst:
# FindATLAS
# -------------
#
# Find ATLAS include dirs and libraries
#
# Use this module by invoking find_package with the form::
#
#   find_package(ATLAS
#     [REQUIRED]             # Fail with error if ATLAS is not found
#     )
#
#   [OPTIONAL]  set(ATLAS_REQUIRE_THREADED [TRUE|FALSE]) to find threaded versions of the libraries
#
# This module defines::
#
#   ATLAS_FOUND            - set to true if the library is found
#   ATLAS_INCLUDE_DIRS     - list of required include directories
#   ATLAS_LIBRARIES        - list of libraries to be linked
#   ATLAS_VERSION_MAJOR    - major version number
#   ATLAS_VERSION_MINOR    - minor version number
#   ATLAS_VERSION_PATCH    - patch version number
#   ATLAS_VERSION_STRING   - version number as a string (ex: "0.2.18")
#
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   ATLAS_ROOT             - Preferred installation prefix for ATLAS
#   ATLAS_DIR              - Preferred installation prefix for ATLAS
#
#
#   ATLAS::ATLAS           - Imported target for the ATLAS library
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

set(ATLAS_SEARCH_PATHS
  ${ATLAS_ROOT}
  $ENV{ATLAS_ROOT}
  ${ATLAS_DIR}
  $ENV{ATLAS_DIR}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local/
  /usr/local/opt # homebrew on mac
  /opt
  /opt/local
  /opt/ATLAS
  )

set(PATH_SUFFIXES_LIST
  lib64/atlas-sse3 #openSUSE 13.2 (Harlequin)
  lib64/atlas-sse2 #openSUSE 13.2 (Harlequin)
  lib64/atlas-sse  #openSUSE 13.2 (Harlequin)
  lib64/atlas      #openSUSE 13.2 (Harlequin)
  lib64
  lib/atlas-sse3 #openSUSE 13.2 (Harlequin)
  lib/atlas-sse2 #openSUSE 13.2 (Harlequin)
  lib/atlas-sse  #openSUSE 13.2 (Harlequin)
  lib/atlas      #openSUSE 13.2 (Harlequin)
  lib
)

if(APPLE)
  list(APPEND PATH_SUFFIXES_LIST openblas/lib)
endif()

# ==============================================================================
# Prepare some helper variables

set(ATLAS_INCLUDE_DIRS)
set(ATLAS_LIBRARIES)

set(ATLAS_THREAD_PREFIX "")
if(ATLAS_REQUIRE_THREADED)
  set(ATLAS_THREAD_PREFIX "pt")
endif()

# ==============================================================================

## First try to find ATLAS with NO_MODULE,
## As of 20160706 version 0.2.18 there is limited cmake support for ATLAS
## that is not as complete as this version, if found, use it
## to identify the ATLAS_VERSION_STRING and improve searching.
find_package(ATLAS NO_MODULE QUIET)
if(ATLAS_VERSION)
  set(ATLAS_VERSION_STRING ${ATLAS_VERSION})
  unset(ATLAS_VERSION) # Use cmake conventional naming
endif()

# ==============================================================================
### First search for headers

find_path(ATLAS_CBLAS_INCLUDE_DIR 
  NAMES cblas.h 
  PATHS ${ATLAS_SEARCH_PATHS} 
  PATH_SUFFIXES include include/openblas)
	   
# ==============================================================================
### Second, search for libraries

find_library(ATLAS_LIB
  NAMES atlas
  PATHS ${ATLAS_SEARCH_PATHS}
  PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

find_library(CBLAS_LIB 
  NAMES ${ATLAS_THREAD_PREFIX}cblas
  PATHS ${LAPACKE_SEARCH_PATHS}
  PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

find_library(LAPACK_LIB 
  NAMES ${ATLAS_THREAD_PREFIX}lapack
  PATHS ${LAPACKE_SEARCH_PATHS}
  PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

if(NOT DEFINED ATLAS_NO_LAPACKE OR NOT ATLAS_NO_LAPACKE)
  if(ATLAS_LIB)
    include(CheckLibraryExists)
    check_library_exists(${ATLAS_LIB} LAPACKE_cheev  "" HAVE_LAPACKE_CHEEV)
    check_library_exists(${ATLAS_LIB} LAPACKE_zgesdd "" HAVE_LAPACKE_ZGESDD)
    
    if(NOT HAVE_LAPACKE_CHEEV OR NOT HAVE_LAPACKE_ZGESDD)
      message(WARNING "ATLAS has no LAPACKE symbols. Attempting to look for a LAPACKE library")
      find_package(LAPACKE QUIET REQUIRED COMPONENTS lapacke)
      list(APPEND ATLAS_LIBRARIES ${LAPACKE_LIBRARIES})
    endif()
  endif()
  set(LAPACKE_LIB_VAR "LAPACKE_LIBRARIES")
else()
  set(LAPACKE_LIB_VAR "")
endif()

find_library(F77BLAS_LIB 
  NAMES ${ATLAS_THREAD_PREFIX}f77blas
  PATHS ${LAPACKE_SEARCH_PATHS}
  PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

# WARNING: We may not be able to determine the version of some ATLAS
set(ATLAS_VERSION_MAJOR 0)
set(ATLAS_VERSION_MINOR 0)
set(ATLAS_VERSION_PATCH 0)
if(ATLAS_VERSION_STRING)
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\1" ATLAS_VERSION_MAJOR "${ATLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\2" ATLAS_VERSION_MINOR "${ATLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\3" ATLAS_VERSION_PATCH "${ATLAS_VERSION_STRING}")
elseif(ATLAS_LIB)
  set(ATLAS_VERSION_STRING "ATLAS.UNKOWN.VERSION")
else()
  set(ATLAS_VERSION_STRING)
endif()

# ==============================================================================
# Checks 'REQUIRED', 'QUIET' and versions.

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ATLAS FOUND_VAR ATLAS_FOUND
  REQUIRED_VARS
  ATLAS_CBLAS_INCLUDE_DIR
  ${LAPACKE_INCLUDE_VAR}
  ${LAPACKE_LIB_VAR}
  LAPACK_LIB
  F77BLAS_LIB
  CBLAS_LIB
  ATLAS_LIB
  VERSION_VAR ATLAS_VERSION_STRING
  )

# ------------------------------------------------------------------------------

if (ATLAS_FOUND)
  list(APPEND ATLAS_INCLUDE_DIRS
    ${ATLAS_CBLAS_INCLUDE_DIR}
    ${ATLAS_CBLAS_INCLUDE_DIR})
  list(REMOVE_DUPLICATES ATLAS_INCLUDE_DIRS)
  
  if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*GNU.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*Intel.*"
      ) #NOT MSVC
    set(MATH_LIB m)
  endif()
  
  list(APPEND ATLAS_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACK_LIB} ${F77BLAS_LIB} ${CBLAS_LIB} ${ATLAS_LIB} ${MATH_LIB})
  
  if(NOT TARGET ATLAS::ATLAS)
    get_filename_component(LIB_EXT "${ATLAS_LIB}" EXT)
    if(LIB_EXT STREQUAL ".a" OR LIB_EXT STREQUAL ".lib")
      set(LIB_TYPE STATIC)
    else()
      set(LIB_TYPE SHARED)
    endif()
    add_library(ATLAS::ATLAS ${LIB_TYPE} IMPORTED GLOBAL)
    set(_tmp_dep_libs "${LAPACKE_LIBRARIES};${LAPACK_LIB};${F77BLAS_LIB};${CBLAS_LIB};${MATH_LIB}")
    list(REMOVE_DUPLICATES _tmp_dep_libs)
    set_target_properties(ATLAS::ATLAS
      PROPERTIES
      IMPORTED_LOCATION "${ATLAS_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${ATLAS_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${_tmp_dep_libs}")
  endif()
  
  if(NOT ATLAS_FIND_QUIETLY)
    get_target_property(_dep_libs ATLAS::ATLAS INTERFACE_LINK_LIBRARIES)

    set(_version "${ATLAS_VERSION_STRING}")
    if(_version STREQUAL "ATLAS.UNKOWN.VERSION")
      set(_version)
    else()
      set(_version " (${_version})")
    endif()
    
    message(STATUS "Found ATLAS${_version} and defined the ATLAS::ATLAS imported target:")
    message(STATUS "  - include:      ${ATLAS_INCLUDE_DIRS}")
    message(STATUS "  - library:      ${ATLAS_LIB}")
    message(STATUS "  - dependencies: ${_dep_libs}")
  endif()
endif()

# ==============================================================================

mark_as_advanced(
  ATLAS_FOUND
  ATLAS_INCLUDE_DIRS
  ATLAS_LIBRARIES
  ATLAS_VERSION_MAJOR
  ATLAS_VERSION_MINOR
  ATLAS_VERSION_PATCH
  ATLAS_VERSION_STRING
  )

# ==============================================================================

