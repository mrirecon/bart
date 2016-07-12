#.rst:
# FindOpenBLAS
# -------------
#
# Find the OpenBLAS library
#
# Using OpenBLAS:
#
# ::
#
#   find_package(OpenBLAS REQUIRED)
#   include_directories(${OpenBLAS_INCLUDE_DIRS})
#   add_executable(foo foo.cc)
#   target_link_libraries(foo ${OpenBLAS_LIBRARIES})
#   -- OR --
#   target_link_libraries(foo ${OpenBLAS_PARALLEL_LIBRARIES})
#
# This module sets the following variables:
#
# ::
#
#   OpenBLAS_FOUND - set to true if the library is found
#   OpenBLAS_INCLUDE_DIRS - list of required include directories
#   OpenBLAS_LIBRARIES - list of libraries to be linked
#   OpenBLAS_HAS_PARALLEL_LIBRARIES - determine if there are parallel libraries compiled
#   OpenBLAS_PARALLEL_LIBRARIES - list of libraries for parallel implementations
#   OpenBLAS_VERSION_MAJOR - major version number
#   OpenBLAS_VERSION_MINOR - minor version number
#   OpenBLAS_VERSION_PATCH - patch version number
#   OpenBLAS_VERSION_STRING - version number as a string (ex: "0.2.18")

#=============================================================================
# Copyright 2016 Hans J. Johnson <hans-johnson@uiowa.edu>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#=============================================================================
#
set(OpenBLAS_HAS_PARALLEL_LIBRARIES FALSE)

set(OpenBLAS_SEARCH_PATHS
  ${OpenBLAS_DIR}
  $ENV{OpenBLAS_DIR}
  $ENV{CMAKE_PREFIX_PATH}
  ${CMAKE_PREFIX_PATH}
  /usr
  /usr/local
  /usr/local/opt/openblas  ## Mac Homebrew install path
  /opt/OpenBLAS
)

set(CMAKE_PREFIX_PATH ${OpenBLAS_SEARCH_PATHS})
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

## First try to find OpenBLAS with NO_MODULE,
## As of 20160706 version 0.2.18 there is limited cmake support for OpenBLAS
## that is not as complete as this version, if found, use it
## to identify the OpenBLAS_VERSION_STRING and improve searching.
find_package(OpenBLAS NO_MODULE QUIET)
if(OpenBLAS_VERSION)
  set(OpenBLAS_VERSION_STRING ${OpenBLAS_VERSION})
  unset(OpenBLAS_VERSION) # Use cmake conventional naming
endif()
##################################################################################################
### First search for headers
find_path(OpenBLAS_CBLAS_INCLUDE_DIR 
             NAMES cblas.h 
             PATHS ${OpenBLAS_SEARCH_PATHS} 
             PATH_SUFFIXES include include/openblas)
find_path(OpenBLAS_LAPACKE_INCLUDE_DIR 
             NAMES lapacke.h 
             PATHS ${OpenBLAS_SEARCH_PATHS} 
             PATH_SUFFIXES include)

##################################################################################################
### Second, search for libraries
set(PATH_SUFFIXES_LIST
  lib64
  lib
)
find_library(OpenBLAS_LIB 
                 NAMES openblas
                 PATHS ${OpenBLAS_SEARCH_PATHS}
                 PATH_SUFFIXES ${PATH_SUFFIXES_LIST})

if(EXISTS ${OpenBLAS_LIB})
   get_filename_component(OpenBLAS_LIB_DIR i${OpenBLAS_LIB} DIRECTORY)
endif()
## Find the named parallel version of openblas
set(OpenBLAS_SEARCH_VERSIONS ${OpenBLAS_VERSION_STRING} 0.2.19 0.2.18 0.2.17 0.2.16)
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

# ------------------------------------------------------------------------
#  Extract version information
# ------------------------------------------------------------------------

# WARNING: We may not be able to determine the version of some OpenBLAS
set(OpenBLAS_VERSION_MAJOR 0)
set(OpenBLAS_VERSION_MINOR 0)
set(OpenBLAS_VERSION_PATCH 0)
if(OpenBLAS_VERSION_STRING)
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\1" OpenBLAS_VERSION_MAJOR "${OpenBLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\2" OpenBLAS_VERSION_MINOR "${OpenBLAS_VERSION_STRING}")
  string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\3" OpenBLAS_VERSION_PATCH "${OpenBLAS_VERSION_STRING}")
endif()

#======================
# Checks 'REQUIRED', 'QUIET' and versions.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS FOUND_VAR OpenBLAS_FOUND
  REQUIRED_VARS 
                OpenBLAS_CBLAS_INCLUDE_DIR
                OpenBLAS_LAPACKE_INCLUDE_DIR
                OpenBLAS_LIB
  VERSION_VAR OpenBLAS_VERSION_STRING
)

if (OpenBLAS_FOUND)
  set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_CBLAS_INCLUDE_DIR} ${OpenBLAS_CBLAS_INCLUDE_DIR})
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
endif()

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

## For debugging
message(STATUS "OpenBLAS_FOUND                  :${OpenBLAS_FOUND}:  - set to true if the library is found")
message(STATUS "OpenBLAS_INCLUDE_DIRS           :${OpenBLAS_INCLUDE_DIRS}: - list of required include directories")
message(STATUS "OpenBLAS_LIBRARIES              :${OpenBLAS_LIBRARIES}: - list of libraries to be linked")
message(STATUS "OpenBLAS_HAS_PARALLEL_LIBRARIES :${OpenBLAS_HAS_PARALLEL_LIBRARIES}: - determine if there are parallel libraries compiled")
message(STATUS "OpenBLAS_PARALLEL_LIBRARIES     :${OpenBLAS_PARALLEL_LIBRARIES}: - list of libraries for parallel implementations")
message(STATUS "OpenBLAS_VERSION_MAJOR          :${OpenBLAS_VERSION_MAJOR}: - major version number")
message(STATUS "OpenBLAS_VERSION_MINOR          :${OpenBLAS_VERSION_MINOR}: - minor version number")
message(STATUS "OpenBLAS_VERSION_PATCH          :${OpenBLAS_VERSION_PATCH}: - patch version number")
message(STATUS "OpenBLAS_VERSION_STRING         :${OpenBLAS_VERSION_STRING}: - version number as a string")
