#.rst:
# FindBLAS
# -------------
#
# Find a BLAS library
#
# Using BLAS:
#
# This module sets the following variables:
#
# ::
#
#   BLAS_FOUND - set to true if the library is found
#   BLAS_INCLUDE_DIRS - list of required include directories
#   BLAS_LIBRARIES - list of libraries to be linked

#=============================================================================
# Copyright 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#=============================================================================
#

set(BLAS_SEARCH_PATHS
  ${BLAS_DIR}
  $ENV{BLAS_DIR}
  $ENV{CMAKE_PREFIX_PATH}
  ${CMAKE_PREFIX_PATH}
  /usr
  /usr/local
  /usr/local/opt/openblas  # Mac Homebrew install path for OpenBLAS
  /opt/local               # Macports general install prefix
)

set(PATH_SUFFIXES_LIST
  lib64
  lib32
  lib
  lib/x86_64-linux-gnu
)

set(CMAKE_PREFIX_PATH ${BLAS_SEARCH_PATHS})

find_library(LAPACK_LIB 
  NAMES lapack
  PATHS ${BLAS_SEARCH_PATHS}
  PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

find_library(BLAS_LIB 
  NAMES blas
  PATHS ${BLAS_SEARCH_PATHS}
  PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

if(DEFINED BLAS_FIND_CBLAS AND BLAS_FIND_CBLAS)
  find_library(CBLAS_LIB
    NAMES cblas
    PATHS ${BLAS_SEARCH_PATHS}
    PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
  set(CBLAS_LIB_VAR "CBLAS_LIB")
else(DEFINED BLAS_FIND_CBLAS AND BLAS_FIND_CBLAS)
  set(CBLAS_LIB_VAR "")
endif(DEFINED BLAS_FIND_CBLAS AND BLAS_FIND_CBLAS)

find_path(BLAS_CBLAS_INCLUDE_DIR
  NAMES cblas.h 
  PATHS ${BLAS_SEARCH_PATHS} 
  PATH_SUFFIXES include)

if(NOT DEFINED BLAS_NO_LAPACKE OR NOT BLAS_NO_LAPACKE)
  find_path(BLAS_LAPACKE_INCLUDE_DIR 
               NAMES lapacke.h 
               PATHS ${BLAS_SEARCH_PATHS} 
               PATH_SUFFIXES include)  
  set(LAPACKE_INCLUDE_VAR "BLAS_LAPACKE_INCLUDE_DIR")
else(NOT DEFINED BLAS_NO_LAPACKE OR NOT BLAS_NO_LAPACKE)
  set(LAPACKE_INCLUDE_VAR "")
endif(NOT DEFINED BLAS_NO_LAPACKE OR NOT BLAS_NO_LAPACKE)

#======================
# Checks 'REQUIRED', 'QUIET' and versions.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLAS FOUND_VAR BLAS_FOUND
  REQUIRED_VARS
  BLAS_CBLAS_INCLUDE_DIR
  ${LAPACKE_INCLUDE_VAR}
  LAPACK_LIB
  BLAS_LIB
  ${CBLAS_LIB_VAR}
)

if (BLAS_FOUND)
  set(BLAS_INCLUDE_DIRS ${BLAS_CBLAS_INCLUDE_DIR})
  list(REMOVE_DUPLICATES BLAS_INCLUDE_DIRS)
  set(BLAS_LIBRARIES)
  list(APPEND BLAS_LIBRARIES ${LAPACK_LIB} ${BLAS_LIB} ${CBLAS_LIB})
  if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*GNU.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*Intel.*"
      ) #NOT MSVC
    list(APPEND BLAS_LIBRARIES m)
  endif()
endif()

mark_as_advanced(
  BLAS_FOUND
  BLAS_INCLUDE_DIRS
  BLAS_LIBRARIES
)

## For debugging
message(STATUS "BLAS_FOUND         :${BLAS_FOUND}:  - set to true if the library is found")
message(STATUS "BLAS_INCLUDE_DIRS  :${BLAS_INCLUDE_DIRS}: - list of required include directories")
message(STATUS "BLAS_LIBRARIES     :${BLAS_LIBRARIES}: - list of libraries to be linked")
