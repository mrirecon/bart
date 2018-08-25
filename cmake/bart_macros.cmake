# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>

# Extract the version number based on version.txt or similar formatted file
macro(get_version_from_file filename)
  message(STATUS "Getting version info from ${filename}")
  file(READ ${filename} _version_string)
  string(STRIP ${_version_string} _version_string)
  string(SUBSTRING ${_version_string} 1 -1 _version_string)
  string(REGEX MATCHALL "[0-9]+" _version_list ${_version_string})
  list(GET _version_list 0 _version_major)
  list(GET _version_list 1 _version_minor)
  list(GET _version_list 2 _version_patch)
endmacro()

# ------------------------------------------------------------------------------

# BART Makefile depends on a GCC flag to add an extra include directive from
# the command line: -include src/main.h, which essentially results in
# '#include "src/main.h" being added at the top of each file being compiled.
# However, since this is not portable (GCC/Clang use -include, MSVC /FI, Intel
# does not suppport it), we use a slightly different approach here.
# Instead, we simply create a copy of the file with the added line at the top
# at *build* time. We also make sure that if the original file should be
# modified, CMake would automatically re-generate it.
macro(configure_file_add_header INFILE PREFIX)
  set(_script "${CMAKE_CURRENT_LIST_DIR}/cmake/bart_configure_add_header.cmake")
  set(_output_folder ${CMAKE_CURRENT_BINARY_DIR}/CombinedCode)
  get_filename_component(_name ${INFILE} NAME)
  set(OUTFILE "${_output_folder}/${_name}")

  # We need to add a protected ';' after each element here as it will be
  # interpreted when CMake prepares the command line arguments below
  # In addition, we also need to escape quotes
  set(_prefix)
  foreach(_line ${PREFIX})
    string(REPLACE "\"" "\\\"" _line ${_line})
    list(APPEND _prefix "\"${_line}\"\;")
  endforeach()

  add_custom_command(OUTPUT ${OUTFILE}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_folder}"
    COMMAND ${CMAKE_COMMAND}
    -DINFILE="${INFILE}"
    -DOUTFILE="${OUTFILE}"
    -DPREFIX="${_prefix}"
     -P "${_script}"
    DEPENDS ${INFILE}
    )
endmacro()

# ------------------------------------------------------------------------------

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# Find a compatible compiler flag from a series of list of possible flags
# For each list of whitespace separated compiler flags passed in argument, this
# function will append or save the compatible flags in ${var_prefix}_c and
# ${var_prefix}_cxx (if C++ is enabled)
macro(check_for_compiler_flags var_prefix)
  set(_c_opts)
  set(_cxx_opts)

  foreach(_flag_list ${ARGN})
    separate_arguments(_flag_list)

    foreach(_flag ${_flag_list})
      # Drop the first character (most likely either '-' or '/')
      string(SUBSTRING ${_flag} 1 -1 _flag_name)
      string(REGEX REPLACE "[-:/]" "_" _flag_name ${_flag_name})

      check_c_compiler_flag(${_flag} c_compiler_has_${_flag_name})
      if(c_compiler_has_${_flag_name})
	list(APPEND _c_opts ${_flag})
	break()
      endif()
    endforeach()

    if(CMAKE_CXX_COMPILER_LOADED)
      foreach(_flag ${_flag_list})
	# Drop the first character (most likely either '-' or '/')
	string(SUBSTRING ${_flag} 1 -1 _flag_name)
	string(REGEX REPLACE "[-:/]" "_" _flag_name ${_flag_name})

	check_cxx_compiler_flag(${_flag} cxx_compiler_has_${_flag_name})
	if(cxx_compiler_has_${_flag_name})
	  list(APPEND _cxx_opts ${_flag})
	  break()
	endif()
      endforeach()
    endif()
  endforeach()

  if(DEFINED ${var_prefix}_c)
    list(APPEND ${var_prefix}_c ${_c_opts})
  else()
    set(${var_prefix}_c ${_c_opts})
  endif()
  if(DEFINED ${var_prefix}_cxx)
    list(APPEND ${var_prefix}_cxx ${_cxx_opts})
  else()
    set(${var_prefix}_cxx ${_cxx_opts})
  endif()
endmacro()
