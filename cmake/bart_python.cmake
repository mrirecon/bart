# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>

# Interpreter only required to find site-lib directories
if(CMAKE_VERSION VERSION_LESS 3.12)
  find_package(BPython ${_python_version} REQUIRED COMPONENTS Development Interpreter)
else()
  find_package(Python ${_python_version} REQUIRED COMPONENTS Development Interpreter)
endif()

set(Python_TGT Python::Python)

get_target_property(_release_lib ${Python_TGT} IMPORTED_LOCATION_RELEASE)
get_target_property(_lib ${Python_TGT} IMPORTED_LOCATION)
message(STATUS "Found Python ${Python_VERSION}:")
message(STATUS "  - Interpreter (${Python_INTERPRETER_ID}): ${Python_EXECUTABLE}")
message(STATUS "  - Development:")
message(STATUS "    + include:  ${Python_INCLUDE_DIR}")
message(STATUS "    + site-lib: ${Python_SITELIB}")
if(_release_lib)
  message(STATUS "    + library:  ${_release_lib}")
else()
  message(STATUS "    + library:  ${_lib}")
endif()

# ==============================================================================

set(PYBART_PYBIND11_SEARCH_PATHS
  ${PYBART_PYBIND11_ROOT}
  $ENV{PYBART_PYBIND11_ROOT}
  ${PYBART_PYBIND11_DIR}
  $ENV{PYBART_PYBIND11_DIR}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local/
  /opt
  /opt/local
  )

set(_pybind11_file pybind11.h)

get_filename_component(_python_inc_path ${Python_INCLUDE_DIR} DIRECTORY)
get_filename_component(_python_ver_name ${Python_INCLUDE_DIR} NAME)
string(REGEX MATCH "python[0-9].[0-9]" _python_ver ${_python_ver_name})

if(UNIX)
  list(APPEND PYBART_PYBIND11_SEARCH_PATHS "/usr/local/include/${_python_ver}")
  list(APPEND PYBART_PYBIND11_SEARCH_PATHS "/usr/local/include/${_python_ver}m")
endif()

find_path(PYBART_PYBIND11_INCLUDE
  NAMES ${_pybind11_file}
  HINTS ${Python_SITELIB} ${Python_INCLUDE_DIR} ${Python_EXTRA_INCLUDE_DIR}
  PATHS ${PYBART_PYBIND11_SEARCH_PATHS}
  PATH_SUFFIXES pybind11
  NO_DEFAULT_PATH
  )

if(NOT PYBART_PYBIND11_INCLUDE)
  set(_msg "Unable to find pybind11/${_pybind11_file}. Please set the path to this")
  set(_msg "${_msg} file on your system to either a system or CMake variable")
  set(_msg "${_msg} named PYBART_PYBIND11_ROOT or PYBART_PYBIND11_DIR or add the")
  set(_msg "${_msg} the path to the CMAKE_PREFIX_PATH")
  message(FATAL_ERROR "${_msg}")
endif()

# ==============================================================================

set(PYBART_NUMPY_SEARCH_PATHS
  ${PYBART_NUMPY_ROOT}
  $ENV{PYBART_NUMPY_ROOT}
  ${PYBART_NUMPY_DIR}
  $ENV{PYBART_NUMPY_DIR}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local/
  /opt
  /opt/local
  )

set(_numpy_file arrayobject.h)

find_path(PYBART_NUMPY_INCLUDE
  NAMES ${_numpy_file}
  HINTS ${Python_SITELIB}
  PATHS ${PYBART_NUMPY_SEARCH_PATHS}
  PATH_SUFFIXES numpy/core/include/numpy include/numpy
  NO_DEFAULT_PATH
  )

if(NOT PYBART_NUMPY_INCLUDE)
  set(_msg "Unable to find numpy/${_numpy_file}. Please set the path to this")
  set(_msg "${_msg} file on your system to either a system or CMake variable")
  set(_msg "${_msg} named PYBART_NUMPY_ROOT or PYBART_NUMPY_DIR or add the")
  set(_msg "${_msg} the path to the CMAKE_PREFIX_PATH")
  message(FATAL_ERROR "${_msg}")
endif()

# ------------------------------------------------------------------------------

get_filename_component(_numpy_inc_dir ${PYBART_NUMPY_INCLUDE} DIRECTORY)
get_filename_component(_pybind11_inc_dir ${PYBART_PYBIND11_INCLUDE} DIRECTORY)

get_target_property(_interface_inc_dir
  ${Python_TGT}
  INTERFACE_INCLUDE_DIRECTORIES
  )
list(APPEND _interface_inc_dir ${_numpy_inc_dir})
list(APPEND _interface_inc_dir ${_pybind11_inc_dir})

set_target_properties(${Python_TGT}
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_interface_inc_dir}"
  )

# ==============================================================================

set(PYBART_FUNCTION_PROTOTYPE)
set(PYBART_COMMANDS_MODULE_METHODS)
set(PYBART_COMMANDS_IMPLEMENTATION)

foreach(curr_prog ${ALLPROGS})
  if(EXISTS "${PROJECT_SOURCE_DIR}/src/${curr_prog}.c")
    set(PYBART_FUNCTION_PROTOTYPE "${PYBART_FUNCTION_PROTOTYPE}static py::object call_${curr_prog} (const char* cmdline);\n")
    set(PYBART_COMMANDS_MODULE_METHODS "${PYBART_COMMANDS_MODULE_METHODS}          m.def(\"${curr_prog}\", call_${curr_prog}, bart_subcommand_docstring);\n")
    set(PYBART_COMMANDS_IMPLEMENTATION "${PYBART_COMMANDS_IMPLEMENTATION}py::object call_${curr_prog} (const char* cmdline)\n{\n     return call_submain(\"${curr_prog}\", cmdline);\n}\n\n")
  endif()
endforeach()

# ==============================================================================
