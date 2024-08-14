string(TIMESTAMP KOQKATOO_BUILD_TIME UTC)
execute_process(
    COMMAND git log -1 --format=%H
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE KOQKATOO_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
configure_file(${CMAKE_CURRENT_LIST_DIR}/koqkatoo-build-time.cpp.in
    koqkatoo-build-time.cpp @ONLY)
