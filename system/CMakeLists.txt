project(teca_system)

include(CMake/teca_system_platform_test.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_system_srcs
    teca_system_interface.cxx
    )

set(teca_system_link teca_core)

if (UNIX)
    include(CheckIncludeFileCXX)
    # check for simple stack trace
    # usually it's in libc but on FreeBSD
    # it's in libexecinfo
    find_library(EXECINFO_LIB "execinfo")
    mark_as_advanced(EXECINFO_LIB)
    if (NOT EXECINFO_LIB)
        set(EXECINFO_LIB "")
    endif()
    CHECK_INCLUDE_FILE_CXX("execinfo.h" TECA_CXX_HAS_EXECINFOH)
    if (TECA_CXX_HAS_EXECINFOH)
        # we have the backtrace header check if it
        # can be used  with this compiler
        set(TECA_PLATFORM_CXX_TEST_LINK_LIBRARIES ${EXECINFO_LIB})
        TECA_PLATFORM_CXX_TEST(TECA_CXX_HAS_BACKTRACE
            "Checking whether backtrace works with this C++ compiler"
            DIRECT)
        set(TECA_PLATFORM_CXX_TEST_LINK_LIBRARIES)
        if (TECA_CXX_HAS_BACKTRACE)
            # backtrace is supported by this system and compiler.
            # now check for the more advanced capabilities.
            SET_PROPERTY(SOURCE teca_system_interface.cxx APPEND PROPERTY
                COMPILE_DEFINITIONS TECA_HAS_BACKTRACE=1)
            # check for symbol lookup using dladdr
            CHECK_INCLUDE_FILE_CXX("dlfcn.h" TECA_CXX_HAS_DLFCNH)
            if (TECA_CXX_HAS_DLFCNH)
                # we have symbol lookup libraries and headers
                # check if they can be used with this compiler
                set(TECA_PLATFORM_CXX_TEST_LINK_LIBRARIES ${CMAKE_DL_LIBS})
                    TECA_PLATFORM_CXX_TEST(TECA_CXX_HAS_DLADDR
                    "Checking whether dladdr works with this C++ compiler"
                    DIRECT)
                set(TECA_PLATFORM_CXX_TEST_LINK_LIBRARIES)
                if (TECA_CXX_HAS_DLADDR)
                    # symbol lookup is supported by this system
                    # and compiler.
                    SET_PROPERTY(SOURCE teca_system_interface.cxx APPEND
                        PROPERTY COMPILE_DEFINITIONS TECA_HAS_SYMBOL_LOOKUP=1)
                endif()
          endif()
          # c++ demangling support
          # check for cxxabi headers
          CHECK_INCLUDE_FILE_CXX("cxxabi.h" TECA_CXX_HAS_CXXABIH)
          if (TECA_CXX_HAS_CXXABIH)
              # check if cxxabi can be used with this
              # system and compiler.
              TECA_PLATFORM_CXX_TEST(TECA_CXX_HAS_CXXABI
                  "Checking whether cxxabi works with this C++ compiler"
                  DIRECT)
              if (TECA_CXX_HAS_CXXABI)
                  # c++ demangle using cxxabi is supported with
                  # this system and compiler
                  SET_PROPERTY(SOURCE teca_system_interface.cxx APPEND
                    PROPERTY COMPILE_DEFINITIONS TECA_HAS_CPP_DEMANGLE=1)
              endif()
          endif()
          # basic backtrace works better with release build
          # don't bother with advanced features for release
          SET_PROPERTY(SOURCE teca_system_interface.cxx APPEND
            PROPERTY COMPILE_DEFINITIONS_DEBUG TECA_HAS_DEBUG_BUILD=1)
          SET_PROPERTY(SOURCE teca_system_interface.cxx APPEND
            PROPERTY COMPILE_DEFINITIONS_RELWITHDEBINFO TECA_HAS_DEBUG_BUILD=1)
        endif()
    endif()
endif()

if (WIN32)
    list(APPEND teca_system_link ws2_32)
    if (TECA_SYS_HAS_PSAPI)
        list(APPEND teca_system_link Psapi)
    endif()
elseif (UNIX)
    if (EXECINFO_LIB AND TECA_CXX_HAS_BACKTRACE)
        # backtrace on FreeBSD is not in libc
        list(APPEND teca_system_link ${EXECINFO_LIB})
    endif()
    if (TECA_CXX_HAS_DLADDR)
        # for symbol lookup using dladdr
        list(APPEND teca_system_link ${CMAKE_DL_LIBS})
    endif()
endif()

add_library(teca_system ${teca_system_srcs})
target_link_libraries(teca_system ${teca_system_link})

target_include_directories(teca_system
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${INCLUDE_PREFIX}
    FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_system
    EXPORT teca_system
    INCLUDES DESTINATION ${INCLUDE_PREFIX}
    ARCHIVE DESTINATION ${LIB_PREFIX}
    LIBRARY DESTINATION ${LIB_PREFIX}
    )

install(EXPORT teca_system
    DESTINATION ${LIB_PREFIX}
    FILE teca_system.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
