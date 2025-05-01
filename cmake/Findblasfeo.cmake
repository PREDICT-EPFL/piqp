# - Find blasfeo

# Look for the header file.
find_path(blasfeo_INCLUDE_DIR
  HINTS ${blasfeo_DIR}/include "/opt/blasfeo/include"
  NAMES blasfeo_target.h)
mark_as_advanced(blasfeo_INCLUDE_DIR)

# Look for the library.
find_library(blasfeo_LIBRARY
  HINTS ${blasfeo_DIR}/lib  "/opt/blasfeo/lib"
  NAMES libblasfeo.a blasfeo)

# handle the QUIETLY and REQUIRED arguments and set blasfeo_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(blasfeo DEFAULT_MSG blasfeo_LIBRARY blasfeo_INCLUDE_DIR)

if (NOT blasfeo_FOUND)
  # try to find blasfeo installed via cmake
  find_package(blasfeo CONFIG)
  set(blasfeo_LIBRARIES blasfeo)
endif ()

if (blasfeo_FOUND)
  message(STATUS "Found blasfeo: ${blasfeo_INCLUDE_DIR} ${blasfeo_LIBRARY}")
  set(blasfeo_LIBRARIES ${blasfeo_LIBRARY})
  set(blasfeo_INCLUDE_DIRS ${blasfeo_INCLUDE_DIR})
  if (NOT TARGET blasfeo)
    add_library(blasfeo UNKNOWN IMPORTED)
    set_target_properties(blasfeo PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${blasfeo_INCLUDE_DIR}")
    set_property(TARGET blasfeo APPEND PROPERTY IMPORTED_LOCATION "${blasfeo_LIBRARY}")
  endif ()
else ()
  message(STATUS "Could not find blasfeo")
  set(blasfeo_LIBRARIES)
  set(blasfeo_INCLUDE_DIRS)
endif ()
