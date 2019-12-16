
# - Try to find the Python module NumPy
#
# This module defines:
#  NUMPY_INCLUDE_DIR: include path for arrayobject.h

# Copyright (c) 2009-2012 Arnaud Barré <arnaud.barre@gmail.com>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FIND_QUIETLY TRUE)
endif()

if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "\"PYTHON_EXECUTABLE\" varabile not set before FindNumPy.cmake was run.")
endif()

# Look for the include path
# WARNING: The variable PYTHON_EXECUTABLE is defined by the script FindPythonInterp.cmake
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print (numpy.get_include()); print (numpy.version.version)"
                 OUTPUT_VARIABLE NUMPY_OUTPUT
                 ERROR_VARIABLE NUMPY_ERROR)
if (NOT NUMPY_ERROR)
  STRING(REPLACE "\n" ";" NUMPY_OUTPUT ${NUMPY_OUTPUT})
  LIST(GET NUMPY_OUTPUT 0 PYTHON_NUMPY_INCLUDE_DIRS)
  LIST(GET NUMPY_OUTPUT 1 PYTHON_NUMPY_VERSION)
endif(NOT NUMPY_ERROR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy DEFAULT_MSG PYTHON_NUMPY_VERSION PYTHON_NUMPY_INCLUDE_DIRS)

set(PYTHON_NUMPY_INCLUDE_DIR ${PYTHON_NUMPY_INCLUDE_DIRS}
    CACHE PATH "Location of NumPy include files.")
mark_as_advanced(PYTHON_NUMPY_INCLUDE_DIR)