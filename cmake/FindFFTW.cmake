#.rst:
# FindFFTW
# -------------
#
# Find FFTW3 include dirs and libraries
#
# Use this module by invoking find_package with the form::
#
#   find_package(Boost
#     [REQUIRED]             # Fail with error if FFTW is not found
#     [VERSION [2,3]]        # Version of FFTW to look for (only considers major version)
#     [COMPONENTS <libs>...] # List of libraries to look for
#     )
#
# Valid names for COMPONENTS libraries are::
#
#   ALL                      - Find all libraries
#   FFTW                     - Find a FFTW (double) library
#   FFTW_MT                  - Find a FFTW multi-threaded (double) library
#   FFTWF                    - Find a FFTW (single) library
#   FFTWF_MT                 - Find a FFTW multi-threaded (single) library
#
#  Not specifying COMPONENTS is identical to choosing ALL
#
# This module defines::
#
#   FFTW_FOUND            - True if headers and requested libraries were found
#   FFTW_VERSION          - Version of the FFTW libraries
#   FFTW_INCLUDE_DIRS     - FFTW include directories
#   FFTW_LIBRARIES        - FFTW component libraries to be linked
#
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   FFTW_ROOT             - Preferred installation prefix for FFTW
#   FFTW_DIR              - Preferred installation prefix for FFTW
#
#
# The following :prop_tgt:`IMPORTED` targets are also defined::
#
#   FFTW::FFTW            - Imported target for the FFTW (double) library
#   FFTW::FFTW_MT         - Imported target for the FFTW multi-thread (double) library
#   FFTW::FFTWF           - Imported target for the FFTW (single) library
#   FFTW::FFTWF_MT        - Imported target for the FFTW multi-thread (single) library
#

# ==============================================================================
# Copyright 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ==============================================================================

set(FFTW_SEARCH_PATHS
  ${FFTW_ROOT}
  $ENV{FFTW_ROOT}
  ${FFTW_DIR}
  $ENV{FFTW_DIR}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /sw
  /usr
  /usr/local/
  /usr/local/opt # homebrew on mac
  /opt
  /opt/local
  /opt/FFTW
  )

set(INC_PATH_SUFFIXES_LIST
  include
  include/fftw
  include/fftw3
)

set(LIB_PATH_SUFFIXES_LIST
  lib64
  lib
  lib/fftw
  lib/x86_64-linux-gnu
  lib32
)

# ==============================================================================
# Prepare some helper variables

set(FFTW_REQUIRED_VARS)
set(FFTW_INCLUDE_DIRS)
set(FFTW_LIBRARIES)

# ==============================================================================

macro(_find_library_with_header component libnames incnames)
  find_library(FFTW_${component}_LIB
    NAMES ${libnames}
    PATHS ${FFTW_SEARCH_PATHS}
    PATH_SUFFIXES ${LIB_PATH_SUFFIXES_LIST})
  if(FFTW_${component}_LIB)    
    set(FFTW_${component}_LIB_FOUND 1)
  endif()
  list(APPEND FFTW_REQUIRED_VARS "FFTW_${component}_LIB")

  # If necessary, look for the header file as well
  if(NOT "${incnames}" STREQUAL "")
    find_path(FFTW_${component}_INCLUDE_DIR
      NAMES ${incnames}
      PATHS ${FFTW_SEARCH_PATHS}
      PATH_SUFFIXES ${INC_PATH_SUFFIXES_LIST})
    list(APPEND FFTW_REQUIRED_VARS "FFTW_${component}_INCLUDE_DIR")
    if(FFTW_${component}_LIB)
      set(FFTW_${component}_INC_FOUND 1)
    endif()
  else()
    set(FFTW_${component}_INC_FOUND 1)
  endif()
  
  if(FFTW_${component}_LIB_FOUND AND FFTW_${component}_INC_FOUND)
    set(FFTW_${component}_FOUND 1)
  else()
    set(FFTW_${component}_FOUND 0)
  endif()
endmacro()

macro(_mangle_names)
endmacro()

# ------------------------------------------------------------------------------

# Make sure that all components are in capitals
set(_tmp_component_list)
foreach(_comp ${FFTW_FIND_COMPONENTS})
  string(TOUPPER ${_comp} _comp)
  list(APPEND _tmp_component_list ${_comp})
endforeach()
set(FFTW_FIND_COMPONENTS ${_tmp_component_list})
set(_tmp_component_list)

# ------------------------------------------------------------------------------

## FFTW can be compiled and subsequently linked against
## various data types.
## There is a single set of include files, and then muttiple libraries,
## One for each type.  I.e. libfftw.a-->double, libfftwf.a-->float


if(NOT FFTW_FIND_COMPONENTS OR FFTW_FIND_COMPONENTS STREQUAL "ALL")
  set(FFTW_FIND_ALL_COMPONENTS 1)
  set(FFTW_FIND_COMPONENTS "FFTW;FFTW_MT;FFTWF;FFTWF_MT")
endif()

if(NOT DEFINED FFTW_FIND_VERSION_MAJOR)
  set(FFTW_FIND_VERSION_MAJOR 3)
endif()

if(FFTW_FIND_VERSION_MAJOR EQUAL 2)
  set(_fftw_suffix "")
elseif(FFTW_FIND_VERSION_MAJOR EQUAL 3)
  set(_fftw_suffix "3")
else()
  message(FATAL_ERROR "Unsupported version number for FFTW")
endif()

# ------------------------------------------------------------------------------

foreach(_comp ${FFTW_FIND_COMPONENTS})
  if(_comp STREQUAL "FFTW")
    _find_library_with_header(${_comp} fftw${_fftw_suffix} fftw${_fftw_suffix}.h)
  elseif(_comp STREQUAL "FFTW_MT")
    _find_library_with_header(${_comp} fftw${_fftw_suffix}_threads fftw${_fftw_suffix}.h)
  elseif(_comp STREQUAL "FFTWF" AND FFTW_FIND_VERSION_MAJOR GREATER 2)
    _find_library_with_header(${_comp} fftw${_fftw_suffix}f fftw${_fftw_suffix}.h)
  elseif(_comp STREQUAL "FFTWF_MT" AND FFTW_FIND_VERSION_MAJOR GREATER 2)
    _find_library_with_header(${_comp} fftw${_fftw_suffix}f_threads fftw${_fftw_suffix}.h)
  else()
    message(FATAL_ERROR "Unknown component (looked for FFTW ${FFTW_FIND_VERSION_MAJOR}): ${_comp}")
  endif()
  mark_as_advanced(
    FFTW_${_comp}_LIB
    FFTW_${_comp}_INCLUDE_DIR)
endforeach()

# ==============================================================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW
  FOUND_VAR FFTW_FOUND
  REQUIRED_VARS ${FFTW_REQUIRED_VARS}
  HANDLE_COMPONENTS)

# ==============================================================================

if(FFTW_FOUND)  
  # Inspired by FindBoost.cmake
  foreach(_comp ${FFTW_FIND_COMPONENTS})
    if(NOT TARGET FFTW::${_comp} AND FFTW_${_comp}_FOUND)
      get_filename_component(LIB_EXT "${FFTW_${_comp}_LIB}" EXT)
      if(LIB_EXT STREQUAL ".a" OR LIB_EXT STREQUAL ".lib")
        set(LIB_TYPE STATIC)
      else()
        set(LIB_TYPE SHARED)
      endif()
      add_library(FFTW::${_comp} ${LIB_TYPE} IMPORTED GLOBAL)
      set_target_properties(FFTW::${_comp}
	PROPERTIES
	IMPORTED_LOCATION "${FFTW_${_comp}_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_${_comp}_INCLUDE_DIR}")
    endif()

    if(FFTW_${_comp}_FOUND)
      set(APPEND FFTW_INCLUDE_DIRS "${FFTW_${_comp}_INCLUDE_DIR}")
      set(APPEND FFTW_LIBRARIES "${FFTW_${_comp}_LIB}")
    endif()
  endforeach()

  # ----------------------------------------------------------------------------

  if(NOT FFTW_FIND_QUIETLY)
    message(STATUS "Found FFTW and defined the following imported targets:")
    foreach(_comp ${FFTW_FIND_COMPONENTS})
      message(STATUS "  - FFTW::${_comp}:")
      message(STATUS "      + include: ${FFTW_${_comp}_INCLUDE_DIR}")
      message(STATUS "      + library: ${FFTW_${_comp}_LIB}")
    endforeach()
  endif()
endif()

# ==============================================================================

mark_as_advanced(
  FFTW_FOUND
  FFTW_INCLUDE_DIRS
  FFTW_LIBRARIES
  )

