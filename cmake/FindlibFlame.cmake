#.rst:
# FindlibFlame
# -------------
#
# Find libFlame include dirs and libraries
#
# Use this module by invoking find_package with the form::
#
#   find_package(libFlame
#     [REQUIRED]             # Fail with error if libFlame is not found
#     )
#
# This module defines::
#
#   libFlame_FOUND            - True if headers and requested libraries were found
#   libFlame_INCLUDE_DIRS     - libFlame include directories
#   libFlame_LIBRARIES        - libFlame component libraries to be linked
#
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   libFlame_ROOT             - Preferred installation prefix
#   libFlame_DIR              - Preferred installation prefix
#
#
# The following :prop_tgt:`IMPORTED` targets are also defined::
#
#   libFlame::libFlame        - Target for specific component dependency
#                               (shared or static library)
#

# ==============================================================================
# Copyright 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ==============================================================================

set(libFlame_SEARCH_PATHS
  ${libFlame_ROOT}
  $ENV{libFlame_ROOT}
  ${libFlame_DIR}
  $ENV{libFlame_DIR}
  $ENV{CMAKE_PREFIX_PATH}
  ${CMAKE_PREFIX_PATH}
  /usr
  /usr/local/
  /usr/local/amd
  /opt
  /opt/amd
  /opt/local
  )

find_library(libFlame_LIBRARY
  NAMES flame
  PATHS ${libFlame_SEARCH_PATHS}
  PATH_SUFFIXES lib lib32 lib64 libflame/lib
  )

if(libFLame_LIBRARY)
  get_filename_component(libFlame_ROOT_HINT ${libFlame_LIBRARY} PATH) # for CMake > 2.8.11 we should really use DIRECTORY
  get_filename_component(libFlame_ROOT_HINT ${libFlame_ROOT_HINT} PATH)  
endif()

find_path(libFlame_INCLUDE_DIR
  NAMES FLAME.h flame.h
  HINTS ${libFlame_ROOT_HINT}
  PATHS ${libFlame_SEARCH_PATHS}
  PATH_SUFFIXES include libflame/include
  )

set(libFlame_LAPACKE_COMPONENTS cblas)

# ==============================================================================

if(libFLame_LIBRARY)
  include(CheckLibraryExists)
  check_library_exists(${libFlame_LIBRARY} zgesdd_ "" HAVE_ZGESDD)
  check_library_exists(${libFlame_LIBRARY} zgelss_ "" HAVE_ZGELSS)
  
  if(HAVE_ZGESDD AND HAVE_ZGELSS)
    set(HAVE_LAPACK_INTERFACE 1)
  else()
    set(HAVE_LAPACK_INTERFACE 0)
  endif()
  
  set(LAPACK_LIB_VAR "")
  if(NOT HAVE_LAPACK_INTERFACE)
    message(STATUS "libFlame has no LAPACK interface, looking for a LAPACK library")
    list(APPEND libFlame_LAPACKE_COMPONENTS lapack)
  endif()

  # ----------------------------------------------------------------------------

  find_package(LAPACKE QUIET REQUIRED COMPONENTS ${libFlame_LAPACKE_COMPONENTS})
endif()

# ==============================================================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libFlame
  FOUND_VAR libFlame_FOUND
  REQUIRED_VARS
  libFlame_INCLUDE_DIR
  libFlame_LIBRARY
  LAPACKE_FOUND
  LAPACKE_LIBRARIES
  LAPACKE_INCLUDE_DIRS
  )

# ==============================================================================

if(libFlame_FOUND)
  set(libFlame_INCLUDE_DIRS)
  list(APPEND libFlame_INCLUDE_DIRS
    ${libFlame_INCLUDE_DIR}
    ${LAPACKE_INCLUDE_DIRS})
  list(REMOVE_DUPLICATES libFlame_INCLUDE_DIRS)

  set(libFlame_LIBRARIES)
  list(APPEND libFlame_LIBRARIES
    ${libFlame_LIBRARY}
    ${LAPACKE_LIBRARIES})
  
  # ----------------------------------------------------------------------------

  if(NOT TARGET libFlame::libFlame)
    get_filename_component(LIB_EXT "${libFlame_LIBRARY}" EXT)
    if(LIB_EXT STREQUAL ".a" OR LIB_EXT STREQUAL ".lib")
      set(LIB_TYPE STATIC)
    else()
      set(LIB_TYPE SHARED)
    endif()
    add_library(libFlame::libFlame ${LIB_TYPE} IMPORTED GLOBAL)
    set_target_properties(libFlame::libFlame PROPERTIES
      IMPORTED_LOCATION "${libFlame_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${libFlame_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${LAPACKE_LIBRARIES}"
      )
  endif()
  
  # ----------------------------------------------------------------------------
  ## For debugging
  if(NOT libFlame_FIND_QUIETLY)
    get_target_property(_dep_libs libFlame::libFlame INTERFACE_LINK_LIBRARIES)
    message(STATUS "Found libFlame and defined the libFlame::libFlame imported target:")
    message(STATUS "  - include:      ${libFlame_INCLUDE_DIR}")
    message(STATUS "  - library:      ${libFlame_LIBRARY}")
    message(STATUS "  - dependencies: ${_dep_libs}")
  endif()
endif()


# ==============================================================================

mark_as_advanced(
  libFlame_LIBRARY
  libFlame_INCLUDE_DIR
  libFlame_CBLAS_INCLUDE_DIR
  )

# ==============================================================================

