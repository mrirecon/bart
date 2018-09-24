# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>

add_new_test_group(pyBART)

if(WIN32)
  add_env_var_to_group(pyBART PYTHONPATH "${PROJECT_BINARY_DIR};$ENV{PYTHONPATH}")
else()
  add_env_var_to_group(pyBART PYTHONPATH "${PROJECT_BINARY_DIR}:$ENV{PYTHONPATH}")
endif()

macro(pyBART_add_test_to_group group test_name)
  list(APPEND _test_list_${group} ${test_name})

  set(_cmd_list)
  foreach(_cmd ${ARGN})
    set(_cmd_list "${_cmd_list}${_cmd}\n")
  endforeach()
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.py CONTENT "${_cmd_list}")
  
  set(TEST_${group}_${test_name}
    "${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/pyBART_run_test.py ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.py")
endmacro()

pyBART_add_test_to_group(pyBART "test_pyBART_import"
  "import pyBART"
  )

pyBART_add_test_to_group(pyBART "test_pyBART_automatic_conversion"
  "import pyBART"
  "import numpy as np"
  "a = np.array([['0','1','2'],['3','4','5']])"
  "pyBART.register_memory('test.mem', a)"
  )

pyBART_add_test_to_group(pyBART "test_pyBART_simple"
  "import pyBART"
  "import numpy as np"
  "a = np.array([ [0,1,2], [3,4,5] ])"
  "pyBART.register_memory('test.mem', a)"
  "pyBART.bart('fmac test.mem test.mem test2.mem')"
  "b = pyBART.load_cfl('test2.mem')"
  "print(np.squeeze(b))"
  "pyBART.cleanup_memory()"
  )
