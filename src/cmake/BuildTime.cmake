string(TIMESTAMP BATMAT_BUILD_TIME UTC)
set(COMMIT_TXT "${CMAKE_CURRENT_LIST_DIR}/../../commit.txt")
if (EXISTS "${COMMIT_TXT}")
    file(STRINGS ${COMMIT_TXT} BATMAT_COMMIT_HASH LIMIT_COUNT 1)
    message("Read Git commit hash from file: ${BATMAT_COMMIT_HASH}")
else()
    execute_process(
        COMMAND git log -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE BATMAT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(
        COMMAND git status . --short --no-branch --untracked-files=no
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE BATMAT_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (NOT BATMAT_GIT_STATUS STREQUAL "" AND NOT BATMAT_COMMIT_HASH STREQUAL "")
        string(APPEND BATMAT_COMMIT_HASH "-dirty")
    endif()
endif()
configure_file(${CMAKE_CURRENT_LIST_DIR}/batmat-build-time.cpp.in
    batmat-build-time.cpp @ONLY)
