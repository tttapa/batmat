include(${PROJECT_SOURCE_DIR}/cmake/Debug.cmake)

# Set the runtime linker/loader search paths to make batmat stand-alone
cmake_path(RELATIVE_PATH BATMAT_INSTALL_LIBDIR
           BASE_DIRECTORY BATMAT_INSTALL_BINDIR
           OUTPUT_VARIABLE BATMAT_INSTALL_LIBRELBINDIR)

function(batmat_add_if_target_exists OUT)
    foreach(TGT IN LISTS ARGN)
        if (TARGET ${TGT})
            list(APPEND ${OUT} ${TGT})
        endif()
    endforeach()
    set(${OUT} ${${OUT}} PARENT_SCOPE)
endfunction()

include(CMakePackageConfigHelpers)

set(BATMAT_INSTALLED_COMPONENTS)
macro(batmat_install_config PKG COMP)
    # Install the target CMake definitions
    install(EXPORT batmat${PKG}Targets
        FILE batmat${PKG}Targets.cmake
        DESTINATION "${BATMAT_INSTALL_CMAKEDIR}"
            COMPONENT ${COMP}
        NAMESPACE batmat::)
    # Add all targets to the build tree export set
    export(EXPORT batmat${PKG}Targets
        FILE "${PROJECT_BINARY_DIR}/batmat${PKG}Targets.cmake"
        NAMESPACE batmat::)
    # Generate the config file that includes the exports
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PKG}Config.cmake.in"
        "${PROJECT_BINARY_DIR}/batmat${PKG}Config.cmake"
        INSTALL_DESTINATION "${BATMAT_INSTALL_CMAKEDIR}"
        NO_SET_AND_CHECK_MACRO)
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/batmat${PKG}ConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMinorVersion)
    # Install the batmatConfig.cmake and batmatConfigVersion.cmake
    install(FILES
        "${PROJECT_BINARY_DIR}/batmat${PKG}Config.cmake"
        "${PROJECT_BINARY_DIR}/batmat${PKG}ConfigVersion.cmake"
        DESTINATION "${BATMAT_INSTALL_CMAKEDIR}"
            COMPONENT ${COMP})
    list(APPEND BATMAT_OPTIONAL_COMPONENTS ${PKG})
endmacro()

macro(batmat_install_cmake FILES COMP)
    # Install a CMake script
    install(FILES ${FILES}
        DESTINATION "${BATMAT_INSTALL_CMAKEDIR}"
            COMPONENT ${COMP})
endmacro()

set(BATMAT_INSTALLED_TARGETS_MSG "\nSummary of batmat components and targets to install:\n\n")

# Install the batmat core libraries
set(BATMAT_CORE_HIDDEN_TARGETS warnings common_options ${BATMAT_CODEGEN_TARGETS})
set(BATMAT_CORE_TARGETS config batmat)
if (BATMAT_CORE_TARGETS)
    install(TARGETS ${BATMAT_CORE_HIDDEN_TARGETS} ${BATMAT_CORE_TARGETS}
        EXPORT batmatCoreTargets
        RUNTIME DESTINATION "${BATMAT_INSTALL_BINDIR}"
            COMPONENT lib
        LIBRARY DESTINATION "${BATMAT_INSTALL_LIBDIR}"
            COMPONENT lib
            NAMELINK_COMPONENT dev
        ARCHIVE DESTINATION "${BATMAT_INSTALL_LIBDIR}"
            COMPONENT dev
        FILE_SET headers DESTINATION "${BATMAT_INSTALL_INCLUDEDIR}"
            COMPONENT dev)
    batmat_install_config(Core dev)
    list(JOIN BATMAT_CORE_TARGETS ", " TGTS)
    string(APPEND BATMAT_INSTALLED_TARGETS_MSG " * Core:  ${TGTS}\n")
    list(APPEND BATMAT_INSTALL_TARGETS ${BATMAT_CORE_TARGETS})
endif()

# Install the extra targets
set(BATMAT_EXTRA_TARGETS)
if (BATMAT_EXTRA_TARGETS)
    install(TARGETS warnings common_options ${BATMAT_EXTRA_TARGETS}
        EXPORT batmatExtraTargets
        RUNTIME DESTINATION "${BATMAT_INSTALL_BINDIR}"
            COMPONENT lib
        LIBRARY DESTINATION "${BATMAT_INSTALL_LIBDIR}"
            COMPONENT lib
            NAMELINK_COMPONENT dev
        ARCHIVE DESTINATION "${BATMAT_INSTALL_LIBDIR}"
            COMPONENT dev
        FILE_SET headers DESTINATION "${BATMAT_INSTALL_INCLUDEDIR}"
            COMPONENT dev)
    batmat_install_config(Extra dev)
    list(JOIN BATMAT_EXTRA_TARGETS ", " TGTS)
    string(APPEND BATMAT_INSTALLED_TARGETS_MSG " * Extra:  ${TGTS}\n")
    list(APPEND BATMAT_INSTALL_TARGETS ${BATMAT_EXTRA_TARGETS})
endif()

# Install the debug files
foreach(target IN LISTS BATMAT_CORE_TARGETS BATMAT_EXTRA_TARGETS)
    get_target_property(target_type ${target} TYPE)
    if (${target_type} STREQUAL "SHARED_LIBRARY")
        batmat_install_debug_syms(${target} debug
                                  ${BATMAT_INSTALL_LIBDIR}
                                  ${BATMAT_INSTALL_BINDIR})
    elseif (${target_type} STREQUAL "EXECUTABLE")
        batmat_install_debug_syms(${target} debug
                                  ${BATMAT_INSTALL_BINDIR}
                                  ${BATMAT_INSTALL_BINDIR})
    endif()
endforeach()

# Make stand-alone
if (BATMAT_STANDALONE)
    foreach(target IN LISTS BATMAT_CORE_TARGETS)
        set_target_properties(${TGT} PROPERTIES
            INSTALL_RPATH "$ORIGIN;$ORIGIN/${BATMAT_INSTALL_LIBRELBINDIR}")
    endforeach()
endif()

# Generate the main config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${PROJECT_BINARY_DIR}/batmatConfig.cmake"
    INSTALL_DESTINATION "${BATMAT_INSTALL_CMAKEDIR}"
    NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/batmatConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion)
# Install the main batmatConfig.cmake and batmatConfigVersion.cmake files
install(FILES
    "${PROJECT_BINARY_DIR}/batmatConfig.cmake"
    "${PROJECT_BINARY_DIR}/batmatConfigVersion.cmake"
    DESTINATION "${BATMAT_INSTALL_CMAKEDIR}"
        COMPONENT dev)

# Print the components and targets we're going to install
message(${BATMAT_INSTALLED_TARGETS_MSG})
