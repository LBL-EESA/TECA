# teca_interface_library
#
#   Adds an interface libarary that can be used to propagate
#   build dependencies.
#
#   ILIB_NAME   -- name of the interface library to create
#   DEFINITIONS -- compile defintitions
#   INCLUDES    -- header include paths
#   LIBRARIES   -- libraries to link to
function(teca_interface_library ILIB_NAME)
    set(opt_args SYSTEM)
    set(val_args)
    set(array_args DEFINITIONS INCLUDES LIBRARIES)
    cmake_parse_arguments(ILIB "${opt_args}" "${val_args}" "${array_args}" ${ARGN})
    add_library(${ILIB_NAME} INTERFACE)
    if (ILIB_DEFINITIONS)
        target_compile_definitions(${ILIB_NAME} INTERFACE ${ILIB_DEFINITIONS})
    endif()
    if (ILIB_INCLUDES)
        if (ILIB_SYSTEM)
            target_include_directories(${ILIB_NAME} SYSTEM INTERFACE ${ILIB_INCLUDES})
        else()
            target_include_directories(${ILIB_NAME} INTERFACE ${ILIB_INCLUDES})
        endif()
    endif()
    if (ILIB_LIBRARIES)
        target_link_libraries(${ILIB_NAME} INTERFACE ${ILIB_LIBRARIES})
    endif()
    install(TARGETS ${ILIB_NAME} EXPORT ${ILIB_NAME})
    install(EXPORT ${ILIB_NAME} DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endfunction()
