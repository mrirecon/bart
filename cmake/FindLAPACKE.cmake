#.rst:
# FindLAPACKE
# -------------
#
# Find the LAPACKE library
#
# Using LAPACKE:
#
# ::
#
#   find_package(LAPACKE REQUIRED)
#   include_directories(${LAPACKE_INCLUDE_DIRS})
#   add_executable(foo foo.cc)
#   target_link_libraries(foo ${LAPACKE_LIBRARIES})
#
# This module sets the following variables:
#
# ::
#
#   LAPACKE_FOUND - set to true if the library is found
#   LAPACKE_INCLUDE_DIRS - list of required include directories
#   LAPACKE_LIBRARIES - list of libraries to be linked
#   LAPACKE_VERSION_MAJOR - major version number
#   LAPACKE_VERSION_MINOR - minor version number
#   LAPACKE_VERSION_PATCH - patch version number
#   LAPACKE_VERSION_STRING - version number as a string (ex: "0.2.18")

#=============================================================================
# Copyright 2016 Hans J. Johnson <hans-johnson@uiowa.edu>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#=============================================================================
#
set(LAPACKE_SEARCH_PATHS
  ${LAPACKE_DIR}
  $ENV{LAPACKE_DIR}
  $ENV{CMAKE_PREFIX_PATH}
  ${CMAKE_PREFIX_PATH}
  /usr
  /usr/local
  /usr/local/opt/lapack  ## Mac Homebrew install path
  /opt/LAPACKE
)
message(STATUS "LAPACKE_SEARCH_PATHS: ${LAPACKE_SEARCH_PATHS}")

set(CMAKE_PREFIX_PATH ${LAPACKE_SEARCH_PATHS})
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

## First try to find LAPACKE with NO_MODULE,
## As of 20160706 version 0.2.18 there is limited cmake support for LAPACKE
## that is not as complete as this version, if found, use it
## to identify the LAPACKE_VERSION_STRING and improve searching.
find_package(LAPACKE NO_MODULE QUIET)
if(LAPACKE_FOUND)
  if(EXISTS ${LAPACKE_DIR}/lapacke-config-version.cmake)
    include(${LAPACKE_DIR}/lapacke-config-version.cmake)
    set(LAPACKE_VERSION_STRING ${PACKAGE_VERSION})
    unset(PACKAGE_VERSION) # Use cmake conventional naming
  endif()
  find_package(LAPACK NO_MODULE QUIET) #Require matching versions here!
  find_package(BLAS NO_MODULE QUIET)   #Require matching versions here!
endif()

##################################################################################################
### First search for headers
find_path(LAPACKE_CBLAS_INCLUDE_DIR 
             NAMES cblas.h 
             PATHS ${LAPACKE_SEARCH_PATHS} 
             PATH_SUFFIXES include include/lapack)
find_path(LAPACKE_LAPACKE_INCLUDE_DIR 
             NAMES lapacke.h 
             PATHS ${LAPACKE_SEARCH_PATHS} 
             PATH_SUFFIXES include)

##################################################################################################
### Second, search for libraries
set(PATH_SUFFIXES_LIST
  lib64
  lib
)
find_library(LAPACKE_LIB 
                 NAMES lapacke
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
find_library(CBLAS_LIB 
                 NAMES cblas
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
find_library(LAPACK_LIB 
                 NAMES lapack
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})
find_library(BLAS_LIB 
                 NAMES blas
                 PATHS ${LAPACKE_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

## TODO: Get version components
# ------------------------------------------------------------------------
#  Extract version information
# ------------------------------------------------------------------------

# WARNING: We may not be able to determine the version of some LAPACKE
set(LAPACKE_VERSION_MAJOR 0)
set(LAPACKE_VERSION_MINOR 0)
set(LAPACKE_VERSION_PATCH 0)
if(LAPACKE_VERSION_STRING)
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\1" LAPACKE_VERSION_MAJOR "${LAPACKE_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\2" LAPACKE_VERSION_MINOR "${LAPACKE_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\3" LAPACKE_VERSION_PATCH "${LAPACKE_VERSION_STRING}")
endif()

#======================
# Checks 'REQUIRED', 'QUIET' and versions.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE FOUND_VAR LAPACKE_FOUND
  REQUIRED_VARS LAPACKE_CBLAS_INCLUDE_DIR
                LAPACKE_LAPACKE_INCLUDE_DIR
                LAPACKE_LIB
                LAPACK_LIB
                CBLAS_LIB
                BLAS_LIB
  VERSION_VAR LAPACKE_VERSION_STRING
)

if (LAPACKE_FOUND)
  set(LAPACKE_INCLUDE_DIRS ${LAPACKE_CBLAS_INCLUDE_DIR} ${LAPACKE_CBLAS_INCLUDE_DIR})
  list(REMOVE_DUPLICATES LAPACKE_INCLUDE_DIRS)
  if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*GNU.*" OR
     "${CMAKE_C_COMPILER_ID}" MATCHES ".*Intel.*"
      ) #NOT MSVC
    set(MATH_LIB m)
  endif()
  list(APPEND LAPACKE_LIBRARIES ${LAPACKE_LIB} ${LAPACK_LIB} ${BLAS_LIB} ${CBLAS_LIB})
  # Check for a common combination, and find required gfortran support libraries

  if(1)
    if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" AND "${CMAKE_Fortran_COMPILER_ID}" MATCHES "GNU")
          message(STATUS "\n\n WARNING: ${CMAKE_C_COMPILER} identified as ${CMAKE_C_COMPILER_ID}\n"
                                   "AND: ${CMAKE_Fortran_COMPILER} identified as ${CMAKE_Fortran_COMPILER_ID}\n"
                               "\n"
                               "may be require special configurations.  The most common is the need to"
                               "explicitly link C programs against the gfortran support library.")
                              
    endif()
  else()
    ## This code automated code is hard to determine if it is robust in many different environments.
    # Check for a common combination, and find required gfortran support libraries
    if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" AND "${CMAKE_Fortran_COMPILER_ID}" MATCHES "GNU")
       include(FortranCInterface)
       FortranCInterface_VERIFY() 
       if(NOT FortranCInterface_VERIFIED_C)
          message(FATAL_ERROR "C and fortran compilers are not compatible:\n${CMAKE_Fortran_COMPILER}:${CMAKE_C_COMPILER}")
       endif()
       
       execute_process(COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=libgfortran.a OUTPUT_VARIABLE FORTRANSUPPORTLIB ERROR_QUIET)
       string(STRIP ${FORTRANSUPPORTLIB} FORTRANSUPPORTLIB)
       if(EXISTS "${FORTRANSUPPORTLIB}")
         list(APPEND LAPACKE_LIBRARIES ${FORTRANSUPPORTLIB})
         message(STATUS "Appending fortran support lib: ${FORTRANSUPPORTLIB}")
       else()
         message(FATAL_ERROR "COULD NOT FIND libgfortran.a support library:${FORTRANSUPPORTLIB}:")
       endif()
    endif()
  endif()
  list(APPEND LAPACKE_LIBRARIES ${MATH_LIB})
endif()

mark_as_advanced(
  LAPACKE_FOUND
  LAPACKE_INCLUDE_DIRS
  LAPACKE_LIBRARIES
  LAPACKE_VERSION_MAJOR
  LAPACKE_VERSION_MINOR
  LAPACKE_VERSION_PATCH
  LAPACKE_VERSION_STRING
)

## For debugging
message(STATUS "LAPACKE_FOUND                  :${LAPACKE_FOUND}:  - set to true if the library is found")
message(STATUS "LAPACKE_INCLUDE_DIRS           :${LAPACKE_INCLUDE_DIRS}: - list of required include directories")
message(STATUS "LAPACKE_LIBRARIES              :${LAPACKE_LIBRARIES}: - list of libraries to be linked")
message(STATUS "LAPACKE_VERSION_MAJOR          :${LAPACKE_VERSION_MAJOR}: - major version number")
message(STATUS "LAPACKE_VERSION_MINOR          :${LAPACKE_VERSION_MINOR}: - minor version number")
message(STATUS "LAPACKE_VERSION_PATCH          :${LAPACKE_VERSION_PATCH}: - patch version number")
message(STATUS "LAPACKE_VERSION_STRING         :${LAPACKE_VERSION_STRING}: - version number as a string")
