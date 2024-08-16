include(${PROJECT_SOURCE_DIR}/cmake/Debug.cmake)

# Set the runtime linker/loader search paths to make koqkatoo stand-alone
cmake_path(RELATIVE_PATH KOQKATOO_INSTALL_LIBDIR
           BASE_DIRECTORY KOQKATOO_INSTALL_BINDIR
           OUTPUT_VARIABLE KOQKATOO_INSTALL_LIBRELBINDIR)

function(koqkatoo_add_if_target_exists OUT)
    foreach(TGT IN LISTS ARGN)
        if (TARGET ${TGT})
            list(APPEND ${OUT} ${TGT})
        endif()
    endforeach()
    set(${OUT} ${${OUT}} PARENT_SCOPE)
endfunction()

include(CMakePackageConfigHelpers)

set(KOQKATOO_INSTALLED_COMPONENTS)
macro(koqkatoo_install_config PKG COMP)
    # Install the target CMake definitions
    install(EXPORT koqkatoo${PKG}Targets
        FILE koqkatoo${PKG}Targets.cmake
        DESTINATION "${KOQKATOO_INSTALL_CMAKEDIR}"
            COMPONENT ${COMP}
        NAMESPACE koqkatoo::)
    # Add all targets to the build tree export set
    export(EXPORT koqkatoo${PKG}Targets
        FILE "${PROJECT_BINARY_DIR}/koqkatoo${PKG}Targets.cmake"
        NAMESPACE koqkatoo::)
    # Generate the config file that includes the exports
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PKG}Config.cmake.in"
        "${PROJECT_BINARY_DIR}/koqkatoo${PKG}Config.cmake"
        INSTALL_DESTINATION "${KOQKATOO_INSTALL_CMAKEDIR}"
        NO_SET_AND_CHECK_MACRO)
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/koqkatoo${PKG}ConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMinorVersion)
    # Install the koqkatooConfig.cmake and koqkatooConfigVersion.cmake
    install(FILES
        "${PROJECT_BINARY_DIR}/koqkatoo${PKG}Config.cmake"
        "${PROJECT_BINARY_DIR}/koqkatoo${PKG}ConfigVersion.cmake"
        DESTINATION "${KOQKATOO_INSTALL_CMAKEDIR}"
            COMPONENT ${COMP})
    list(APPEND KOQKATOO_OPTIONAL_COMPONENTS ${PKG})
endmacro()

macro(koqkatoo_install_cmake FILES COMP)
    # Install a CMake script
    install(FILES ${FILES}
        DESTINATION "${KOQKATOO_INSTALL_CMAKEDIR}"
            COMPONENT ${COMP})
endmacro()

set(KOQKATOO_INSTALLED_TARGETS_MSG "\nSummary of koqkatoo components and targets to install:\n\n")

# Install the koqkatoo core libraries
set(KOQKATOO_CORE_HIDDEN_TARGETS warnings blas-lapack-lib common_options _linalg-compact _linalg-compact-microkernels)
set(KOQKATOO_CORE_TARGETS config koqkatoo linalg linalg-compact ocp)
if (KOQKATOO_CORE_TARGETS)
    install(TARGETS ${KOQKATOO_CORE_HIDDEN_TARGETS} ${KOQKATOO_CORE_TARGETS}
        EXPORT koqkatooCoreTargets
        RUNTIME DESTINATION "${KOQKATOO_INSTALL_BINDIR}"
            COMPONENT lib
        LIBRARY DESTINATION "${KOQKATOO_INSTALL_LIBDIR}"
            COMPONENT lib
            NAMELINK_COMPONENT dev
        ARCHIVE DESTINATION "${KOQKATOO_INSTALL_LIBDIR}"
            COMPONENT dev
        FILE_SET headers DESTINATION "${KOQKATOO_INSTALL_INCLUDEDIR}"
            COMPONENT dev)
    koqkatoo_install_config(Core dev)
    list(JOIN KOQKATOO_CORE_TARGETS ", " TGTS)
    string(APPEND KOQKATOO_INSTALLED_TARGETS_MSG " * Core:  ${TGTS}\n")
    list(APPEND KOQKATOO_INSTALL_TARGETS ${KOQKATOO_CORE_TARGETS})
endif()

# Install the extra targets
set(KOQKATOO_EXTRA_TARGETS)
if (KOQKATOO_EXTRA_TARGETS)
    install(TARGETS warnings common_options ${KOQKATOO_EXTRA_TARGETS}
        EXPORT koqkatooExtraTargets
        RUNTIME DESTINATION "${KOQKATOO_INSTALL_BINDIR}"
            COMPONENT lib
        LIBRARY DESTINATION "${KOQKATOO_INSTALL_LIBDIR}"
            COMPONENT lib
            NAMELINK_COMPONENT dev
        ARCHIVE DESTINATION "${KOQKATOO_INSTALL_LIBDIR}"
            COMPONENT dev
        FILE_SET headers DESTINATION "${KOQKATOO_INSTALL_INCLUDEDIR}"
            COMPONENT dev)
    koqkatoo_install_config(Extra dev)
    list(JOIN KOQKATOO_EXTRA_TARGETS ", " TGTS)
    string(APPEND KOQKATOO_INSTALLED_TARGETS_MSG " * Extra:  ${TGTS}\n")
    list(APPEND KOQKATOO_INSTALL_TARGETS ${KOQKATOO_EXTRA_TARGETS})
endif()

# Install the debug files
foreach(target IN LISTS KOQKATOO_CORE_TARGETS KOQKATOO_EXTRA_TARGETS)
    get_target_property(target_type ${target} TYPE)
    if (${target_type} STREQUAL "SHARED_LIBRARY")
        koqkatoo_install_debug_syms(${target} debug
                                  ${KOQKATOO_INSTALL_LIBDIR}
                                  ${KOQKATOO_INSTALL_BINDIR})
    elseif (${target_type} STREQUAL "EXECUTABLE")
        koqkatoo_install_debug_syms(${target} debug
                                  ${KOQKATOO_INSTALL_BINDIR}
                                  ${KOQKATOO_INSTALL_BINDIR})
    endif()
endforeach()

# Make stand-alone
if (KOQKATOO_STANDALONE)
    foreach(target IN LISTS KOQKATOO_CORE_TARGETS)
        set_target_properties(${TGT} PROPERTIES
            INSTALL_RPATH "$ORIGIN;$ORIGIN/${KOQKATOO_INSTALL_LIBRELBINDIR}")
    endforeach()
endif()

# Generate the main config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${PROJECT_BINARY_DIR}/koqkatooConfig.cmake"
    INSTALL_DESTINATION "${KOQKATOO_INSTALL_CMAKEDIR}"
    NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/koqkatooConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion)
# Install the main koqkatooConfig.cmake and koqkatooConfigVersion.cmake files
install(FILES
    "${PROJECT_BINARY_DIR}/koqkatooConfig.cmake"
    "${PROJECT_BINARY_DIR}/koqkatooConfigVersion.cmake"
    DESTINATION "${KOQKATOO_INSTALL_CMAKEDIR}"
        COMPONENT dev)

# Print the components and targets we're going to install
message(${KOQKATOO_INSTALLED_TARGETS_MSG})
