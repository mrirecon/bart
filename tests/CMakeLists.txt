# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>


# ==============================================================================
# Generate a pair of randomly named directories

# string(RANDOM LENGTH 4 ALPHABET "0123456789" _random)
set(_random "cmake")
set(_tests_data_dir "${CMAKE_CURRENT_BINARY_DIR}/data_${_random}")
set(_tests_out_dir "${CMAKE_CURRENT_BINARY_DIR}/out_${_random}")

add_custom_command(
  OUTPUT ${_tests_data_dir} ${_tests_out_dir}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_tests_data_dir}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_tests_out_dir}"
  )

# ==============================================================================
# Main helper variables

set(_test_groups)
set(_test_list_data)
set(_data_file_list)
set(_data_cmd_list)

# ==============================================================================
# This macro parses a XXX.mk file and extracts its tests and related commands
#
# Each call to this macro appends a new group to the _test_groups list
# For each group XXX, a variable _test_list_XXX is created to store the names
# of the tests present in that group.
# For each test TTT in group XXX, a variable TEST_XXX_TTT contains the list of
# commands to execute for that particular test
macro(parse_test_file filename)
  # First get name of the group and add it to the global list
  get_filename_component(_group ${filename} NAME_WE)
  list(APPEND _test_groups ${_group})
  set(_test_list_${_group})

  # Read the file
  file(READ ${filename} _test_file_lines)
  string(REPLACE "\\" ";;" _test_file_lines ${_test_file_lines})
  string(REPLACE "\n" ";;" _test_file_lines ${_test_file_lines})

  # The following algorithm is relying on the fact that we first encounter a line
  # with the test name (of the form tests/test-name:) and then a series of
  # tab-indented lines that start with $(TOOLDIR)/some-command + arguments

  set(_curr_test)
  foreach(_line ${_test_file_lines})
    string(STRIP ${_line} _is_empty)
    if(NOT ${_is_empty} STREQUAL "")
      # Look for $(TOOLDIR) to find out if a line is a command or not
      string(FIND ${_line} "$(TOOLDIR)" _is_cmd)

      # Try to grab the test name
      string(REGEX MATCH "^tests/([^:]+)" _test_name ${_line})

      if(NOT "${_test_name}" STREQUAL "")
	# Got the name -> remove tests/ from the beginning of the string
	string(SUBSTRING ${_test_name} 6 -1 _test_name)
	string(REPLACE "-" "_" _test_name ${_test_name})
	# Add this test to the list of test for this group
	list(APPEND _test_list_${_group} ${_test_name})
	set(_curr_test ${_test_name})
      elseif(_is_cmd GREATER -1 AND NOT _curr_test STREQUAL "")
	# First chars are TAB + $ -> command part of a test
	
	# Separate line by whitespace
	string(REPLACE " " ";" _comp ${_line})

	set(_cmd)
	foreach(_c ${_comp})
	  # Starts with "\t$(TOOLDIR)" => it's the name of the command
	  string(REGEX MATCH "^\t\\$\\(TOOLDIR\\)/([a-zA-z0-9]+)" _cmd_name ${_c})
	  # Contains $(TESTS_OUT) => path to an input file in data diretory
	  string(REGEX MATCH "\\$\\(TESTS_OUT\\)/(.*)" _path ${_c})

	  if(NOT ${_cmd_name} STREQUAL "")
	    string(REPLACE "$(TOOLDIR)/" " $<TARGET_FILE:bart> " _cmd_name ${_cmd_name})
	    set(_cmd "${_cmd} ${_cmd_name}")
	  elseif(NOT ${_path} STREQUAL "")
	    string(REPLACE "$(TESTS_OUT)/" "${_tests_data_dir}/" _path ${_path})
	    file(TO_NATIVE_PATH ${_path} _path)
	    set(_cmd "${_cmd} ${_path}")
	  else()
	    set(_cmd "${_cmd} ${_c}")
	  endif()
	endforeach()

	# If we were called with some extra arguments, apply an extra processing
	# step to the command before adding it to the list
	if(${ARGC} GREATER 1)
	  _parse_cmdline_custom(${_cmd} ${ARGN})
	endif()

	# Cleanup string and add it to the list of commands for the current test
	string(STRIP ${_cmd} _cmd)
	list(APPEND TEST_${_group}_${_curr_test} ${_cmd})
      endif()
    else()
      # Reset current test name if we encounter anything else
      set(_curr_test)
    endif()
  endforeach()
endmacro()

# ------------------------------------------------------------------------------
# This macro parses a XXX.mk file and extracts the commands that generate some
# test data
#
# Each call to this macro may append some name XXX to the list of data files
# (_test_list_data). If a new test data file is found, then the command needed
# to generate the file is saved in the _data_XXX_cmd variable and the path to
# the file in the _data_XXX_file variable.
# Also, if the generation of the current data file is dependent on another one,
# the variable _data_XXX_deps will contain the file that needs to be created
# first.
macro(parse_test_file_for_data filename)
  # First read the file
  file(READ ${filename} _test_file_lines)
  string(REPLACE "\\" ";" _test_file_lines ${_test_file_lines})
  string(REPLACE "\n" ";" _test_file_lines ${_test_file_lines})

  set(_data_name)
  set(_data_cmd)
  foreach(_line ${_test_file_lines})
    string(STRIP ${_line} _is_empty)
    string(LENGTH ${_line} _str_length)
    if(NOT ${_is_empty} STREQUAL "" AND _str_length GREATER 12)
      string(SUBSTRING "${_line}" 0 12 _name_header)
      string(SUBSTRING "${_line}" 0 2  _cmd_header)

      if(_name_header STREQUAL "$(TESTS_OUT)")
	# Found a name line -> extract the name and add it to the list of names
	string(REGEX MATCH "\\$\\(TESTS_OUT\\)/([^:]+)" _data_name ${_line})
	string(REPLACE "$(TESTS_OUT)/" "" _data_name ${_data_name})
	get_filename_component(_bname ${_data_name} NAME_WE)
	list(APPEND _test_list_data ${_bname})
	set(_data_cmd)
      elseif(_data_name AND NOT _data_cmd AND ${_cmd_header} STREQUAL "\t$")
	# Got a command line -> parse it and save the command
	string(REPLACE "$(TOOLDIR)/" "$<TARGET_FILE:bart> " _data_cmd ${_line})

	# If the line contains $(TESTS_OUT) we have a dependency
	string(REGEX MATCH "\\$\\(TESTS_OUT\\)/([^ \t]+)" _data_deps ${_data_cmd})

	if(NOT _data_deps STREQUAL "")
	  # Modify the command to have the correct path
	  string(REPLACE "$(TESTS_OUT)/" "${_tests_data_dir}/" _data_cmd ${_data_cmd})
	  # Cleanup the dependency name and save it
	  string(REPLACE "$(TESTS_OUT)/" "" _data_deps ${_data_deps})
	  set(_data_${_bname}_deps "${_tests_data_dir}/${_data_deps}")
	endif()
	# Cleanup command line and then save it
	string(REPLACE "$@" "${_data_name}" _data_cmd ${_data_cmd})
	string(STRIP ${_data_cmd} _data_${_bname}_cmd)
	# Propery set the name of the data file and save it
	file(TO_NATIVE_PATH "${_tests_data_dir}/${_data_name}" _data_file)
	set(_data_${_bname}_file ${_data_file})
      else()
	set(_data_name)
	set(_data_cmd)
	set(_bname)
      endif()
    endif()
  endforeach()
endmacro()

# ------------------------------------------------------------------------------
# This macro processes the list of test data files to be generated and creates
# custom commands to generate them.
#
# We simply iterate over the list of data files, grab the required information
# and then generate a file to create the CMake custom command
macro(add_commands_for_test_data_generation)
  foreach(_data_name ${_test_list_data})
    # Set relevant helper variables
    set(TEST_DATA_OUTPUT ${_data_${_data_name}_file})
    set(TEST_DATA_CMD ${_data_${_data_name}_cmd})
    set(TEST_DATA_DEPS ${_tests_data_dir};${_data_${_data_name}_deps})
    set(TEST_DATA_DIR ${_tests_data_dir})
    # Generate the file...
    configure_file(${PROJECT_SOURCE_DIR}/cmake/tests_data_generation.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/tests_datagen_${_data_name}.cmake @ONLY)
    # ...and include it
    include(${CMAKE_CURRENT_BINARY_DIR}/tests_datagen_${_data_name}.cmake)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/tests_datagen_${_data_name}.cmake)
    # While we're at it also keep a list of all the test data files
    list(APPEND _data_file_list ${_data_${_data_name}_file})
  endforeach()
endmacro()

# ------------------------------------------------------------------------------
# This macro processes the list of tests for a particular group and generates
# a CMake script file in order to execute them
macro(create_integration_test group)
  # First, create a custom target to execute all the test in the group
  set(_grp_test_tgt integration_${group})
  add_custom_target(${_grp_test_tgt}
    DEPENDS ${_data_file_list})

  # Then iterate over all the tests in the group
  foreach(_test ${_test_list_${group}})
    # Skip GPU-related test if we did not compile with CUDA support
    string(REGEX MATCH ".*gpu.*" _test_require_gpu ${_test})
    if(USE_CUDA OR _test_require_gpu STREQUAL "")
      # Build the CMake script file content
      set(_cmd_list "set(_error_flag 0)\n")
      list(APPEND _cmd_list "execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue \"Executing ${_test}\")\n")
      foreach(_cmd ${TEST_${group}_${_test}})
	set(_exec_in)
	
	# Set environment variables before calling the command if required
	if(NOT ${_test_${group}_env_name} STREQUAL "")
	  foreach(_name ${_test_${group}_env_name})
            set(_exec_in "${_exec_in}\nset(ENV{${_name}} \"${_test_${group}_env_val}\")")
	  endforeach()
	endif()

	string(REPLACE "\"" "\\\"" _cmd_escaped ${_cmd})
	
	# For each command in the test, we make a call execute_process()
	set(_exec_in "${_exec_in}\nexecute_process(\nCOMMAND ${_cmd}\nRESULT_VARIABLE res_var\nWORKING_DIRECTORY ${_tests_out_dir}\nOUTPUT_VARIABLE _output\nERROR_VARIABLE _output)\nif(NOT \"\${res_var}\" STREQUAL \"0\")\n  message(ERROR \"Command ${_cmd_escaped} failed with return \${res_var}:\\n \${_output} \")\nset(_error_flag 1)\nendif()\n\n")
	list(APPEND _cmd_list "${_exec_in}")
      endforeach()
      list(APPEND _cmd_list "if(_error_flag)\nexecute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red \"FAILED\")\nelse()\nexecute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green \"PASSED\")\nendif()\n")
      string(REPLACE ";" "\n" _cmd_list ${_cmd_list})

      # Finally, we write the file...
      file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_test}.cmake CONTENT "${_cmd_list}")
      # ...and create a target to execute it.
      add_custom_command(TARGET ${_grp_test_tgt}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/${_test}.cmake)
    endif()
  endforeach()
endmacro()

# ==============================================================================
# Macro to handle some special cases
set(_python_cmd_match "python")
macro(_parse_cmdline_custom cmd type)
  if(${type} STREQUAL ${_python_cmd_match})
    string(REGEX MATCH "PYTHONPATH.*" _is_python ${cmd})
    if(NOT _is_python STREQUAL "")
      string(REGEX REPLACE "PYTHONPATH[^ \t]+(.*)" "${Python_EXECUTABLE}\\1" _cmd "${cmd}")
      string(REGEX REPLACE "\\$\\(TOOLDIR\\)/(.*)" "${PROJECT_SOURCE_DIR}/\\1" _cmd "${_cmd}")
    endif()
  endif()
endmacro()

# ==============================================================================
# Macros to manually add some tests (ie. not based on some *.mk file)
macro(add_new_test_group group)
  list(APPEND _test_groups ${group})
endmacro()

macro(add_test_to_group group test_name)
  list(APPEND _test_list_${group} ${test_name})
  set(TEST_${group}_${test_name} ${ARGN})
endmacro()

macro(add_env_var_to_group group name value)
  set(_test_${group}_env_name "${name}")
  set(_test_${group}_env_val "${value}")
endmacro()

# ==============================================================================

# Now that we have all our macros, we simply iterate over all the XXX.mk files
# in the folder, while skipping some incompatible tests
#   - join: calls commands like `seq 1 300` which may not work if not on Linux
#   - estdelay: one call to `$(TOOLDIR)/estdelay` in the middle of another
#               command -> impossible to do with the current approach
#
# NB: the GPU-only tests are already ignored when parsing the tests file
#     since they all contain the "gpu" string in their name
file(GLOB _file_list ${CMAKE_CURRENT_LIST_DIR}/*.mk)
foreach(_file ${_file_list})
  get_filename_component(_bname ${_file} NAME_WE)
  if(_bname MATCHES "python")
    if(CMAKE_VERSION VERSION_LESS 3.12)
      find_package(BPython ${_python_version} QUIET COMPONENTS Interpreter)
    else()
      find_package(Python ${_python_version} QUIET COMPONENTS Interpreter)
    endif()

    if(Python_FOUND)
      set(_test_${_bname}_env_name "PYTHONPATH")
      if(WIN32)
	set(_test_${_bname}_env_val "${PROJECT_SOURCE_DIR}/python;$ENV{PYTHONPATH}")
      else()
	set(_test_${_bname}_env_val "${PROJECT_SOURCE_DIR}/python:$ENV{PYTHONPATH}")
      endif()
      parse_test_file(${_file} ${_python_cmd_match})
    else()
      message(WARNING "Python interpreter not found: unable to perform python test")
    endif()
  elseif(_bname STREQUAL "join"
      OR _bname STREQUAL "estdelay")
    # Do nothing
  else()
    parse_test_file_for_data(${_file})
    parse_test_file(${_file})
  endif()
endforeach()

if(BART_CREATE_PYTHON_MODULE)
  if(CMAKE_VERSION VERSION_LESS 3.12)
    find_package(BPython ${_python_version} QUIET COMPONENTS Interpreter)
  else()
    find_package(Python ${_python_version} QUIET COMPONENTS Interpreter)
  endif()
  if(Python_FOUND)
    include(${CMAKE_CURRENT_LIST_DIR}/pyBART.cmake)
  else()
    message(WARNING "Python interpreter not found: unable to perform pyBART test")
  endif()
endif()

add_commands_for_test_data_generation()

# ==============================================================================
# Finally add conveniene targets to launch all tests

add_custom_target(integration)

foreach(_group ${_test_groups})
  create_integration_test(${_group})
  add_dependencies(integration integration_${_group})
endforeach()

add_custom_target(integration_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${_tests_data_dir}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${_tests_out_dir}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
