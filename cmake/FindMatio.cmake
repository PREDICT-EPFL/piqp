# - Find Matio
# Find the native Matio headers and libraries.
#
#  MATIO_DIR - set this to where to look for matio
#
#  MATIO_INCLUDE_DIRS - where to find nana/nana.h, etc.
#  MATIO_LIBRARIES    - List of libraries when using nana.
#  MATIO_FOUND        - True if nana found.

# Look for the header file.
find_path(MATIO_INCLUDE_DIR
  HINTS ${MATIO_DIR}/include
  NAMES matio.h)
mark_as_advanced(MATIO_INCLUDE_DIR)

# Look for the library.
find_library(MATIO_LIBRARY
  HINTS ${MATIO_DIR}/lib ${MATIO_DIR}/lib64
  NAMES matio)
find_library(Z_LIBRARY
  HINTS "${MATIO_DIR}"
  NAMES z)
mark_as_advanced(MATIO_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set MATIO_FOUND to TRUE if 
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Matio DEFAULT_MSG MATIO_LIBRARY MATIO_INCLUDE_DIR)

if(Matio_FOUND)
  set(MATIO_LIBRARIES ${MATIO_LIBRARY} ${Z_LIBRARY})
  set(MATIO_INCLUDE_DIRS ${MATIO_INCLUDE_DIR})
  if(NOT TARGET Matio::Matio)
    add_library(Matio::Matio UNKNOWN IMPORTED)
    set_target_properties(Matio::Matio PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MATIO_INCLUDE_DIR}")
    set_property(TARGET Matio::Matio APPEND PROPERTY IMPORTED_LOCATION "${MATIO_LIBRARY}")
  endif()
else()
  set(MATIO_LIBRARIES)
  set(MATIO_INCLUDE_DIRS)
endif()
