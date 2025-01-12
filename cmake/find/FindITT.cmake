# https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-0/instrumentation-tracing-technology-api-reference.html

include(FindPackageHandleStandardArgs)

find_path(ITT_INCLUDE_DIR ittnotify.h
    PATHS ${VTUNE_PROFILER_DIR} $ENV{VTUNE_PROFILER_DIR})
find_library(ITT_LIBRARY ittnotify
    PATHS ${VTUNE_PROFILER_DIR} $ENV{VTUNE_PROFILER_DIR})

mark_as_advanced(ITT_INCLUDE_DIR CUTEST_DIR ITT_LIBRARY)
find_package_handle_standard_args(ITT REQUIRED_VARS ITT_LIBRARY ITT_INCLUDE_DIR)

if (ITT_FOUND AND NOT TARGET ITT::ITT)
    add_library(ITT::ITT UNKNOWN IMPORTED)
    target_include_directories(ITT::ITT INTERFACE ${ITT_INCLUDE_DIR})
    set_target_properties(ITT::ITT PROPERTIES IMPORTED_LOCATION ${ITT_LIBRARY})
endif()
