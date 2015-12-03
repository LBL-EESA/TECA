cmake_minimum_required(VERSION 2.8.12)

project(teca)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# add the requisite flags to enable c++11 features
# on various platforms
if (MSVC)
    # TODO require vs2013
    # enable exceptions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -Wall -Wextra")
    if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
endif()

# platform specific compile time feature detection
message(STATUS "Check for c++ regex support")
try_run(CXX11_REGEX_WORKS CXX11_REGEX_COMPILES
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMake/teca_platform_tests.cpp"
    CMAKE_FLAGS ${CMAKE_CXX_FLAGS}
    COMPILE_DEFINITIONS "-DCXX11_REGEX_TEST")
if (NOT (CXX11_REGEX_COMPILES AND CXX11_REGEX_WORKS))
    message(STATUS "Check for c++ regex support -- not found. Use a c++11 compiler to enable.")
else()
    message(STATUS "Check for c++ regex support -- enabled")
    set(TECA_HAS_REGEX ON CACHE BOOL "Regex features")
endif()

# locate dependencies
find_package(MPI)
if (MPI_C_FOUND)
    message(STATUS "MPI features -- enabled")
    set(TECA_HAS_MPI ON CACHE BOOL "MPI features")
else()
    message(STATUS "MPI features -- not found.")
endif()

find_package(NetCDF)
if (NETCDF_FOUND)
    message(STATUS "NetCDF features -- enabled")
    set(TECA_HAS_NETCDF ON CACHE BOOL "NetCDF features")
else()
    message(STATUS "NetCDF features -- not found. set NETCDF_DIR to enable.")
endif()

find_package(VTK QUIET)
if (VTK_FOUND)
    message(STATUS "VTK features -- enabled")
    set(TECA_HAS_VTK ON CACHE BOOL "VTK features")
else()
    message(STATUS "VTK features -- not found. set VTK_DIR to enable.")
endif()

find_package(Boost QUIET COMPONENTS program_options)
if (Boost_FOUND)
    message(STATUS "Boost features -- enabled")
    set(TECA_HAS_BOOST ON CACHE BOOL "Boost features")
else()
    message(STATUS "Boost fetaures -- not found. set BOOST_ROOT to enable.")
endif()

set(TECA_PYTHON_BINDING ON CACHE BOOL "Python binding")
if (TECA_PYTHON_BINDING)
    message(STATUS "Python binding -- enabled")
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    find_package(Numpy REQUIRED)
    find_program(swig_cmd swig)
    if (NOT swig_cmd)
        message(WARNING "swig was not found, you won't be able to modify .i files")
    endif()
    function(wrap_swig input output)
        if (swig_cmd)
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${output}
                COMMAND ${swig_cmd} -c++ -python -w341
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../core
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../data
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../io
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
                    -o ${CMAKE_CURRENT_SOURCE_DIR}/${output}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${input}
                DEPENDS ${input}
                COMMENT "wrapping ${input}...")
        endif()
    endfunction()
    function(teca_python_module mname)
        set(extra_includes ${ARGV1})
        include_directories(SYSTEM
            ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIR})
        wrap_swig(teca_py_${mname}.i teca_py_${mname}.cxx ${extra_includes})
        PYTHON_ADD_MODULE(_teca_py_${mname}
            ${CMAKE_CURRENT_SOURCE_DIR}/teca_py_${mname}.cxx)
        target_link_libraries(_teca_py_${mname} ${PYTHON_LIBRARIES} teca_${mname})
        add_custom_command(TARGET _teca_py_${mname} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/teca_py_${mname}.py
                ${CMAKE_CURRENT_BINARY_DIR}/../lib)
        install(TARGETS _teca_py_${mname} LIBRARY DESTINATION lib)
        install(FILES teca_py_${mname}.py DESTINATION lib)
    endfunction()
else()
    message(STATUS "Python binding -- disabled")
endif()

# generaste global configuration
configure_file(teca_config.h.in teca_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# build library components and applications
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(io)
add_subdirectory(alg)
add_subdirectory(apps)

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
