set(_required_vars INFILE OUTFILE PREFIX)
foreach(_var ${_required_vars})
  if(NOT DEFINED "${_var}")
    message(FATAL_ERROR "Variable ${_var} is undefined!")
  endif()
endforeach()

if(NOT EXISTS "${INFILE}")
  message(FATAL_ERROR "\n\nMissing input file '${INFILE}' for generating '${OUTFILE}'\n\n")
endif()

# ==============================================================================

file(READ "${INFILE}" _file_content)

set(_prefix_string)
foreach(_line ${PREFIX})
  set(_prefix_string "${_prefix_string}\n${_line}")
endforeach()

file(WRITE "${OUTFILE}" "${_prefix_string}\n\n\n${_file_content}")
