#.rst:
# FindATLAS
# -------------
#
# Find the ATLAS library
#
# Using ATLAS:
#
# ::
#
#   [OPTIONAL]  set(ATLAS_REQUIRE_THREADED [TRUE|FALSE]) to find threaded versions of the libraries
#   find_package(ATLAS REQUIRED)
#   include_directories(${ATLAS_INCLUDE_DIRS})
#   add_executable(foo foo.cc)
#   target_link_libraries(foo ${ATLAS_LIBRARIES})
#   -- OR --
#   target_link_libraries(foo ${ATLAS_PARALLEL_LIBRARIES})
#
# This module sets the following variables:
#
# ::
#
#   ATLAS_FOUND - set to true if the library is found
#   ATLAS_INCLUDE_DIRS - list of required include directories
#   ATLAS_LIBRARIES - list of libraries to be linked
#   ATLAS_VERSION_MAJOR - major version number
#   ATLAS_VERSION_MINOR - minor version number
#   ATLAS_VERSION_PATCH - patch version number
#   ATLAS_VERSION_STRING - version number as a string (ex: "0.2.18")

#=============================================================================
# Copyright 2016 Hans J. Johnson <hans-johnson@uiowa.edu>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#=============================================================================
#

set(ATLAS_SEARCH_PATHS
  ${ATLAS_DIR}
  $ENV{ATLAS_DIR}
  $ENV{CMAKE_PREFIX_PATH}
  ${CMAKE_PREFIX_PATH}
  /usr
  /usr/local
  /usr/local/opt/openblas  ## Mac Homebrew install path
  /opt/ATLAS
)

set(CMAKE_PREFIX_PATH ${ATLAS_SEARCH_PATHS})
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

## First try to find ATLAS with NO_MODULE,
## As of 20160706 version 0.2.18 there is limited cmake support for ATLAS
## that is not as complete as this version, if found, use it
## to identify the ATLAS_VERSION_STRING and improve searching.
find_package(ATLAS NO_MODULE QUIET)
if(ATLAS_VERSION)
  set(ATLAS_VERSION_STRING ${ATLAS_VERSION})
  unset(ATLAS_VERSION) # Use cmake conventional naming
endif()
##################################################################################################
### First search for headers
find_path(ATLAS_CBLAS_INCLUDE_DIR 
             NAMES cblas.h 
             PATHS ${ATLAS_SEARCH_PATHS} 
             PATH_SUFFIXES include include/openblas)
find_path(ATLAS_LAPACKE_INCLUDE_DIR 
             NAMES lapacke.h 
             PATHS ${ATLAS_SEARCH_PATHS} 
             PATH_SUFFIXES include)

##################################################################################################
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

### Second, search for libraries
find_library(ATLAS_LIB 
                 NAMES atlas
                 PATHS ${ATLAS_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

set(ATLAS_THREAD_PREFIX "")
if(ATLAS_REQUIRE_THREADED)
  set(ATLAS_THREAD_PREFIX "pt")
endif()
find_library(CBLAS_LIB 
                 NAMES ${ATLAS_THREAD_PREFIX}cblas
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
find_library(LAPACK_LIB 
                 NAMES ${ATLAS_THREAD_PREFIX}lapack
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
find_library(LAPACKE_LIB 
                 NAMES ${ATLAS_THREAD_PREFIX}lapacke
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
find_library(F77BLAS_LIB 
                 NAMES ${ATLAS_THREAD_PREFIX}f77blas
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

# ------------------------------------------------------------------------
#  Extract version information
# ------------------------------------------------------------------------

# WARNING: We may not be able to determine the version of some ATLAS
set(ATLAS_VERSION_MAJOR 0)
set(ATLAS_VERSION_MINOR 0)
set(ATLAS_VERSION_PATCH 0)
if(ATLAS_VERSION_STRING)
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\1" ATLAS_VERSION_MAJOR "${ATLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\2" ATLAS_VERSION_MINOR "${ATLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\3" ATLAS_VERSION_PATCH "${ATLAS_VERSION_STRING}")
else()
  set(ATLAS_VERSION_STRING "ATLAS.UNKOWN.VERSION")
endif()

#======================
# Checks 'REQUIRED', 'QUIET' and versions.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ATLAS FOUND_VAR ATLAS_FOUND
  REQUIRED_VARS
                ATLAS_CBLAS_INCLUDE_DIR
                ATLAS_LAPACKE_INCLUDE_DIR
                LAPACKE_LIB
                LAPACK_LIB
                F77BLAS_LIB
                CBLAS_LIB
                ATLAS_LIB
  VERSION_VAR ATLAS_VERSION_STRING
)
if (ATLAS_FOUND)
  set(ATLAS_INCLUDE_DIRS ${ATLAS_CBLAS_INCLUDE_DIR} ${ATLAS_CBLAS_INCLUDE_DIR})
  list(REMOVE_DUPLICATES ATLAS_INCLUDE_DIRS)
  if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*GNU.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*Intel.*"
      ) #NOT MSVC
    set(MATH_LIB m)
  endif()
  list(APPEND ATLAS_LIBRARIES ${LAPACKE_LIB} ${LAPACK_LIB} ${F77BLAS_LIB} ${CBLAS_LIB} ${ATLAS_LIB} ${MATH_LIB})
endif()

mark_as_advanced(
  ATLAS_FOUND
  ATLAS_INCLUDE_DIRS
  ATLAS_LIBRARIES
  ATLAS_VERSION_MAJOR
  ATLAS_VERSION_MINOR
  ATLAS_VERSION_PATCH
  ATLAS_VERSION_STRING
)

## For debugging
message(STATUS "ATLAS_FOUND                  :${ATLAS_FOUND}:  - set to true if the library is found")
message(STATUS "ATLAS_INCLUDE_DIRS           :${ATLAS_INCLUDE_DIRS}: - list of required include directories")
message(STATUS "ATLAS_LIBRARIES              :${ATLAS_LIBRARIES}: - list of libraries to be linked")
message(STATUS "ATLAS_VERSION_MAJOR          :${ATLAS_VERSION_MAJOR}: - major version number")
message(STATUS "ATLAS_VERSION_MINOR          :${ATLAS_VERSION_MINOR}: - minor version number")
message(STATUS "ATLAS_VERSION_PATCH          :${ATLAS_VERSION_PATCH}: - patch version number")
message(STATUS "ATLAS_VERSION_STRING         :${ATLAS_VERSION_STRING}: - version number as a string")
