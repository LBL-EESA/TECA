cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0028 NEW)
cmake_policy(SET CMP0053 NEW)
cmake_policy(SET CMP0063 NEW)

project(teca C CXX Fortran)

set(TECA_CUDA_ARCHITECTURES 75 CACHE STRING
    "Target CUDA compute capability")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES ${TECA_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
    #TODO -- using hidden visibility on MacOS with clang breaks our Python bindings
    message(STATUS "Symbol visibility is set to default")
else ()
    message(STATUS "Symbol visibility is set to hidden")
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_C_VISIBILITY_PRESET hidden)
endif()

# this prevents a relink when a shared library's implementation changes
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)



# set build/install sub dirs for various components
if (NOT LIB_PREFIX)
  set(LIB_PREFIX lib)
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIB_PREFIX})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIB_PREFIX})
if (NOT BIN_PREFIX)
  set(BIN_PREFIX bin)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${BIN_PREFIX})
if (NOT INCLUDE_PREFIX)
  set(INCLUDE_PREFIX include)
endif()

# bring our cmake codes into scope
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# if we forgot to set the build type default to release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "One of: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Configuring a ${CMAKE_BUILD_TYPE} build")

# add the requisite flags to enable c++11 features
# on various platforms.
if (MSVC)
    # TODO require vs2015
    # enable exceptions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc"
            CACHE STRING "" FORCE)
else()
    if (NOT CMAKE_CXX_FLAGS)
        set(tmp "-fPIC -std=c++17 -Wall -Wextra -Wno-unknown-pragmas")
        if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
            set(tmp "${tmp} -stdlib=libc++")
        endif()
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -O3 -march=native -mtune=native -fno-trapping-math -fno-math-errno")
            if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
                set(tmp "${tmp} -fno-signaling-nans")
            endif()
        endif()
        set(CMAKE_CXX_FLAGS "${tmp}"
            CACHE STRING "TECA build defaults"
            FORCE)
    endif()
    if (NOT CMAKE_Fortran_FLAGS)
        set(tmp "-Wall -Wextra -Wno-conversion -Wno-compare-reals -fno-math-errno")
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            set(tmp "${tmp} -DNDEBUG -march=native -mtune=native -fno-trapping-math -fno-signaling-nans -fno-math-errno")
        endif()
        set(CMAKE_Fortran_FLAGS "${tmp}"
            CACHE STRING "TECA build defaults"
            FORCE)
    endif()
    if (NOT CMAKE_CUDA_FLAGS)
        set(tmp "--default-stream per-thread --expt-relaxed-constexpr")
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -DNDEBUG -Xcompiler -Wall,-Wextra,-O3,-march=native,-mtune=native,-fno-trapping-math,-fno-math-errno")
            if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
                set(tmp "${tmp},-fno-signaling-nans")
            endif()
        elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(tmp "${tmp} -g -G -Xcompiler -Wall,-Wextra,-O0,-g")
        endif()
        set(CMAKE_CUDA_FLAGS "${tmp}"
            CACHE STRING "TECA build defaults"
            FORCE)
        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CXX_FLAGS_RELEASE}")
        set(CMAKE_CXX_FLAGS_RELEASE "${tmp}"
            CACHE STRING "TECA build defaults"
            FORCE)
        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CUDA_FLAGS_RELEASE}")
        set(CMAKE_CUDA_FLAGS_RELEASE "${tmp}"
            CACHE STRING "TECA build defaults"
            FORCE)
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

include(teca_interface_library)

# these control behavior of the build. most of the time you want to error
# out if these are not found. for those times when you don't set the corresponding
# REQUIRE variable to FALSE
set(REQUIRE_CUDA FALSE CACHE BOOL "Forces build failure when CUDA is missing")
set(REQUIRE_MPI TRUE CACHE BOOL "Forces build failure when MPI is missing")
set(REQUIRE_NETCDF TRUE CACHE BOOL "Forces build failure when NetCDF is missing")
set(REQUIRE_NETCDF_MPI TRUE CACHE BOOL "Forces build failure when NetCDF_MPI is missing")
set(REQUIRE_UDUNITS TRUE CACHE BOOL "Forces build failure when udunits2 is missing")
set(REQUIRE_BOOST TRUE CACHE BOOL "Forces build failure when boost is missing")
set(REQUIRE_OPENSSL ${REQUIRE_NETCDF} CACHE BOOL "Forces build failure when OpenSSL is missing")
set(REQUIRE_PYTHON TRUE CACHE BOOL "Forces build failure when Python is missing")
set(REQUIRE_LIBXLSXWRITER FALSE CACHE BOOL "Forces build failure when libxlsxwriter is missing")
set(REQUIRE_SHAPELIB FALSE CACHE BOOL "Forces build failure when shapelib is missing")
set(REQUIRE_PARAVIEW FALSE CACHE BOOL "Forces build failure when ParaView is missing")
set(REQUIRE_VTK FALSE CACHE BOOL "Forces build failure when VTK is missing")
set(REQUIRE_TECA_DATA ${BUILD_TESTING} CACHE BOOL "Forces build failure when TECA_data is missing")

# locate dependencies
#configure for CUDA
set(tmp OFF)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER AND ((DEFINED TECA_HAS_CUDA AND TECA_HAS_CUDA)
    OR (NOT DEFINED TECA_HAS_CUDA)))
    enable_language(CUDA)
    message(STATUS "CUDA features -- enabled (${TECA_CUDA_ARCHITECTURES})")
    set(tmp ON)
elseif (REQUIRE_CUDA)
    message(FATAL_ERROR "CUDA features -- required but not found.")
else()
    message(STATUS "CUDA features -- not found.")
endif()
set(TECA_HAS_CUDA ${tmp} CACHE BOOL "CUDA features")
if (TECA_HAS_CUDA)
    set(HAMR_ENABLE_OBJECTS OFF CACHE BOOL "")
    set(HAMR_ENABLE_CUDA ON CACHE BOOL "")
    set(HAMR_CUDA_ARCHITECTURES ${TECA_CUDA_ARCHITECTURES} CACHE STRING "")
else()
    set(HAMR_ENABLE_CUDA OFF CACHE BOOL "")
endif()

# configure for MPI
if (ENABLE_CRAY_MPICH)
    set(ENV{PKG_CONFIG_PATH} "$ENV{CRAY_MPICH_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(PkgConfig QUIET)
    pkg_check_modules(CRAY_MPICH REQUIRED QUIET mpich)
    set(MPI_C_INCLUDE_PATH ${CRAY_MPICH_INCLUDE_DIRS} CACHE STRING "MPI include directories")
    set(MPI_C_LIBRARIES ${CRAY_MPICH_LDFLAGS} CACHE STRING "MPI link dependencies")
    set(MPIEXEC srun CACHE STRING "Platform MPI run equivalent")
    set(MPI_C_FOUND CACHE BOOL ON "status of MPI config")
else()
    find_package(MPI)
endif()
set(tmp OFF)
if (MPI_C_FOUND  AND ((DEFINED TECA_HAS_MPI AND TECA_HAS_MPI)
    OR (NOT DEFINED TECA_HAS_MPI)))
    message(STATUS "MPI features -- enabled")
    set(tmp ON)
    teca_interface_library(MPI SYSTEM
        DEFINITIONS -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1
        INCLUDES ${MPI_C_INCLUDE_PATH} ${MPI_C_INCLUDE_DIRS}
        LIBRARIES ${MPI_C_LIBRARIES})
elseif (REQUIRE_MPI)
    message(FATAL_ERROR "MPI features -- required but not found.")
else()
    message(STATUS "MPI features -- not found.")
endif()
set(TECA_HAS_MPI ${tmp} CACHE BOOL "MPI features")

# configure for NetCDF
set(tmp OFF)
find_package(NetCDF)
if (NETCDF_FOUND AND ((DEFINED TECA_HAS_NETCDF AND TECA_HAS_NETCDF)
    OR (NOT DEFINED TECA_HAS_NETCDF)))
    message(STATUS "NetCDF features -- enabled")
    set(tmp ON)
    teca_interface_library(NetCDF SYSTEM
        INCLUDES ${NETCDF_INCLUDE_DIRS}
        LIBRARIES ${NETCDF_LIBRARIES})
     if (NOT NETCDF_IS_PARALLEL)
        message(STATUS "Check NetcCDF for MPI support -- not found")
        if (REQUIRE_NETCDF_MPI)
            message(FATAL_ERROR "NetCDF MPI support -- required but not found.")
        endif()
    else()
        message(STATUS "Check NetCDF for MPI support -- enabled")
    endif()
elseif (REQUIRE_NETCDF)
    message(FATAL_ERROR "NetCDF features -- required but not found. set NETCDF_DIR to enable.")
else()
    message(STATUS "NetCDF features -- not found. set NETCDF_DIR to enable.")
    message(WARNING "NetCDF is required for CF-2 I/O")
endif()
set(TECA_HAS_NETCDF ${tmp} CACHE BOOL "NetCDF features")
set(TECA_HAS_NETCDF_MPI ${NETCDF_IS_PARALLEL} CACHE BOOL "NetCDF MPI support")

# configure for libxlsxwriter
set(tmp OFF)
find_package(LibXLSXWriter)
if (LIBXLSXWRITER_FOUND AND ((DEFINED TECA_HAS_LIBXLSXWRITER AND TECA_HAS_LIBXLSXWRITER)
    OR (NOT DEFINED TECA_HAS_LIBXLSXWRITER)))
    message(STATUS "libxlsxwriter features -- enabled")
    set(tmp ON)
elseif (REQUIRE_LIBXLSXWRITER)
    message(STATUS "libxlsxwriter features -- required but not found. set LIBXLSXWRITER_DIR to enable.")
else()
    message(STATUS "libxlsxwriter features -- not found. set LIBXLSXWRITER_DIR to enable.")
endif()
set(TECA_HAS_LIBXLSXWRITER ${tmp} CACHE BOOL "libxlsxwriter features")

# configure for UDUnits
set(tmp OFF)
find_package(UDUnits)
if (UDUNITS_FOUND AND ((DEFINED TECA_HAS_UDUNITS AND TECA_HAS_UDUNITS)
    OR (NOT DEFINED TECA_HAS_UDUNITS)))
    message(STATUS "UDUnits features -- enabled")
    set(tmp ON)
elseif (REQUIRE_UDUNITS)
    message(FATAL_ERROR "UDUnits features -- required but not found. set UDUNITS_DIR to enable.")
else()
    message(STATUS "UDUnits features -- not found. set UDUNITS_DIR to enable.")
endif()
set(TECA_HAS_UDUNITS ${tmp} CACHE BOOL "UDUnits features")

# configure for ParaView
set(tmp OFF)
find_package(ParaView QUIET)
if (ParaView_FOUND AND ((DEFINED TECA_HAS_PARAVIEW AND TECA_HAS_PARAVIEW) OR (NOT DEFINED TECA_HAS_PARAVIEW)))
    message(STATUS "ParaView features -- enabled")
    set(tmp ON)
elseif (REQUIRE_PARAVIEW)
    message(FATAL_ERROR "ParaView features -- required but not found. set ParaView_DIR to enable.")
else()
    message(STATUS "ParaView features -- not found. set ParaView_DIR to enable.")
endif()
set(TECA_HAS_PARAVIEW ${tmp} CACHE BOOL "ParaView features")

# configure for VTK
set(tmp OFF)
if (NOT TECA_HAS_PARAVIEW)
    find_package(VTK QUIET)
    if (VTK_FOUND AND ((DEFINED TECA_HAS_VTK AND TECA_HAS_VTK) OR (NOT DEFINED TECA_HAS_VTK)))
        message(STATUS "VTK features -- enabled")
        set(tmp ON)
    elseif (REQUIRE_VTK)
        message(FATAL_ERROR "VTK features -- required but not found. set VTK_DIR to enable.")
    else()
        message(STATUS "VTK features -- not found. set VTK_DIR to enable.")
    endif()
endif()
set(TECA_HAS_VTK ${tmp} CACHE BOOL "VTK features")

#configure for Boost
set(tmp OFF)
find_package(Boost QUIET COMPONENTS program_options)
if (Boost_FOUND AND ((DEFINED TECA_HAS_BOOST AND TECA_HAS_BOOST) OR (NOT DEFINED TECA_HAS_BOOST)))
    message(STATUS "Boost features -- enabled")
    set(tmp ON)
elseif (REQUIRE_BOOST)
    message(FATAL_ERROR "Boost fetaures -- required but not found. set BOOST_ROOT to enable.")
else()
    message(STATUS "Boost fetaures -- not found. set BOOST_ROOT to enable.")
    message(WARNING "Boost is required for command line applications")
endif()
set(TECA_HAS_BOOST ${tmp} CACHE BOOL "Boost features")

#configure for OpenSSL
set(tmp OFF)
find_package(OpenSSL QUIET)
if (OpenSSL_FOUND AND ((DEFINED TECA_HAS_OPENSSL AND TECA_HAS_OPENSSL) OR (NOT DEFINED TECA_HAS_OPENSSL)))
    message(STATUS "OpenSSL features -- enabled")
    set(tmp ON)
elseif (REQUIRE_OPENSSL)
    message(FATAL_ERROR "OpenSSL fetaures -- required but not found. set OPENSSL_ROOT_DIR to enable.")
else()
    message(STATUS "OpenSSL fetaures -- not found. set OPENSSL_ROOT_DIR to enable.")
endif()
set(TECA_HAS_OPENSSL ${tmp} CACHE BOOL "OpenSSL features")

# configure for Python
set(TECA_PYTHON_VERSION 3 CACHE STRING "The major version number of Python TECA should use.")
set_property(CACHE TECA_PYTHON_VERSION PROPERTY STRINGS 2 3)
set(tmp OFF)
find_package(PythonInterp ${TECA_PYTHON_VERSION})
if(PYTHONINTERP_FOUND)
    find_program(PYTHON_CONFIG_EXECUTABLE python${PYTHON_VERSION_MAJOR}-config)
    if (NOT PYTHON_CONFIG_EXECUTABLE)
        message(SEND_ERROR "python${PYTHON_VERSION_MAJOR}-config executable is required.")
    endif()
    if (TECA_PYTHON_VERSION EQUAL 3)
        set(PYTHON3_ABI_FLAGS)
        execute_process(COMMAND ${PYTHON_CONFIG_EXECUTABLE} --abiflags
            OUTPUT_VARIABLE PYTHON3_ABI_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(py_lib_ver ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}${PYTHON3_ABI_FLAGS})
    else()
        set(py_lib_ver ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    endif()
    execute_process(COMMAND ${PYTHON_CONFIG_EXECUTABLE} --prefix
        OUTPUT_VARIABLE python_prefix OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(PYTHON_INCLUDE_DIR ${python_prefix}/include/python${py_lib_ver})
    if (EXISTS ${python_prefix}/lib/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(PYTHON_LIBRARY ${python_prefix}/lib/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
    elseif (EXISTS ${python_prefix}/lib64/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(PYTHON_LIBRARY ${python_prefix}/lib64/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
    elseif (EXISTS ${python_prefix}/lib/x86_64-linux-gnu/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(PYTHON_LIBRARY ${python_prefix}/lib/x86_64-linux-gnu/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
        message(SEND_ERROR "Failed to locate Python library for ${python_prefix}")
    endif()
endif()
find_package(PythonLibs ${TECA_PYTHON_VERSION})
find_package(NumPy)
find_package(CuPy)
find_package(PyTorch)
find_package(Matplotlib)
find_package(TcpyPI)
if (TECA_HAS_MPI)
    find_package(MPI4Py)
endif()
find_program(swig_cmd NAMES swig swig3.0 swig4.0)
if (swig_cmd)
    message(STATUS "Found SWIG: TRUE")
else()
    message(FATAL_ERROR "Found SWIG: FALSE")
endif()
if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND NumPy_FOUND AND swig_cmd
    AND ((TECA_HAS_MPI AND MPI4Py_FOUND) OR (NOT TECA_HAS_MPI))
    AND ((DEFINED TECA_HAS_PYTHON AND TECA_HAS_PYTHON)
    OR (NOT DEFINED TECA_HAS_PYTHON)))
    message(STATUS "Python ${TECA_PYTHON_VERSION} features -- enabled")
    set(tmp ON)
    teca_interface_library(PYTHON SYSTEM
        INCLUDES ${PYTHON_INCLUDE_PATH} ${MPI4Py_INCLUDE_DIR} ${NumPy_INCLUDE_DIR}
        LIBRARIES ${PYTHON_LIBRARIES})
elseif (REQUIRE_PYTHON)
    message(FATAL_ERROR "Python ${TECA_PYTHON_VERSION} features -- required but not found")
else()
    message(STATUS "Python ${TECA_PYTHON_VERSION} features -- not found. Requires Python, SWIG, numpy, and mpi4py if compiling with MPI")
endif()
set(TECA_HAS_PYTHON ${tmp} CACHE BOOL "TECA Python bindings")
set(TECA_HAS_NUMPY ${NumPy_FOUND} CACHE BOOL "Python NumPy module")
set(TECA_HAS_CUPY ${CuPy_FOUND} CACHE BOOL "Python CuPy module")
set(TECA_HAS_PYTORCH ${PyTorch_FOUND} CACHE BOOL "Python torch module")
set(TECA_HAS_MATPLOTLIB ${Matplotlib_FOUND} CACHE BOOL "Python matplotlib module")
set(TECA_HAS_TCPYPI ${TcpyPI_FOUND} CACHE BOOL "Python tcpyPI module")

# configure for shapelib
set(tmp OFF)
find_package(shapelib QUIET)
if (shapelib_FOUND AND ((DEFINED TECA_HAS_SHAPELIB AND TECA_HAS_SHAPELIB)
    OR (NOT DEFINED TECA_HAS_SHAPELIB)))
    message(STATUS "shapelib features -- enabled")
    set(tmp ON)
elseif (REQUIRE_SHAPELIB)
    message(FATAL_ERROR "shapelib features -- required but not found. set SHAPELIB_DIR to enable.")
else()
    message(STATUS "shapelib features -- not found. set SHAPELIB_DIR to enable.")
endif()
set(TECA_HAS_SHAPELIB ${tmp} CACHE BOOL "shapelib features")


# silence the annoying cmake warnings about REQUIRE_<X> when
# the required package X is actually found.
set(ignore ${REQUIRE_MPI} ${REQUIRE_NETCDF} ${REQUIRE_LIBXLSXWRITE}
    ${REQUIRE_UDUNITS} ${REQUIRE_PARAVIEW} ${REQUIRE_VTK} ${REQUIRE_BOOST}
    ${REQUIRE_OPENSSL} ${REQUIRE_PYTHON})

# configure library type
set(CMAKE_MACOSX_RPATH 1)
# ensure binaries can access linked libraries w/o LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
if (TECA_HAS_PYTHON)
    # SWIG RTTI system requires libs when multiple modules
    # have cross dependencies
    set(BUILD_SHARED_LIBS ON FORCE)
endif()

# enable/disable profiling interface
set(TECA_ENABLE_PROFILER OFF CACHE STRING "Enable internal profiler")
if (TECA_ENABLE_PROFILER)
    message(STATUS "Profiling features -- enabled")
else()
    message(STATUS "Profiling features -- disabled")
endif()

# get the build version descriptor
if (NOT DEFINED TECA_VERSION)
    set(tmp)
    find_package(Git)
    if (GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE}
            --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git describe --tags
            OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    if (NOT tmp)
        set(tmp "0.0.0")
    endif()
else()
    set(tmp ${TECA_VERSION})
endif()
set(TECA_VERSION_DESCR ${tmp} CACHE STRING "TECA version descriptor" FORCE)
message(STATUS "TECA version ${TECA_VERSION_DESCR}")

# do we have access to a checkout of TECA_data?
set(tmp OFF)
set(TECA_DATA_ROOT "/path/to/TECA_data"
    CACHE PATH "Path to TECA_data repository")
if (EXISTS "${TECA_DATA_ROOT}")
    set(tmp ON)
    message(STATUS "TECA_data -- available")
elseif (REQUIRE_TECA_DATA)
    message(FATAL_ERROR "TECA_data -- required but not found")
else()
    message(STATUS "TECA_data -- not available")
endif()
set(TECA_HAS_DATA ${tmp} CACHE BOOL "TECA_data is present")

# generate global configuration
include(CMake/teca_test.cmake)
include(CMake/teca_app.cmake)
include(CMake/teca_python.cmake)
configure_file(teca_config.h.in teca_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# build library components and applications
add_subdirectory(HAMR)
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(alg)
add_subdirectory(io)
add_subdirectory(python)
add_subdirectory(system)
add_subdirectory(apps)
add_subdirectory(paraview)

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    # these are dsiabled by default because they are slow and threaded version
    # provides sufficient coverage.
    set(TECA_SERIAL_TESTS OFF CACHE BOOL "Enables serial tests of threaded codes.")

    include(CTest)
    include(ProcessorCount)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake"
        @ONLY)

    # figure out how many cores we can use for parallel tests
    set(TECA_TEST_CORES 0 CACHE STRING
        "Max number of cores for use in parallel tests")

    # by default assume 2 hyperthreads per core, if this is not
    # the case override here
    set(HYPERTHREADS_PER_CORE 2 CACHE STRING
        "The number of hyperthreads per core.")

    # use CMake to get the number of logical cores. includes hyperthreads
    # in the count.
    if (TECA_TEST_CORES LESS 1)
        ProcessorCount(LOGICAL_CORES)
        if (LOGICAL_CORES EQUAL 0)
            message(FATAL_ERROR "Failed to detect the number of cores. "
                "Set TECA_TEST_CORES")
        endif()
    else()
        math(EXPR LOGICAL_CORES "${TECA_TEST_CORES}*${HYPERTHREADS_PER_CORE}")
    endif()

    # adjust count for hyperthreads.
    math(EXPR PHYSICAL_CORES "${LOGICAL_CORES}/${HYPERTHREADS_PER_CORE}")
    if (PHYSICAL_CORES LESS 1)
        message(FATAL_ERROR "Invalid CPU configuration. "
            "LOGICAL_CORES=${LOGICAL_CORES} HYPERTHREADS_PER_CORE="
            "${HYPERTHREADS_PER_CORE}")
    endif()

    # set the number of cores to use for pure MPI or purely threaded tests
    set(TEST_CORES ${PHYSICAL_CORES})
    message(STATUS "regression testing -- enabled (${TEST_CORES} cores).")

    # set the number of cores to use for MPI + threads tests. if there are too
    # few physical cores then disable hybrid parallel tests
    math(EXPR HALF_TEST_CORES "${TEST_CORES}/2")
    if (HALF_TEST_CORES LESS 2)
        message(STATUS "Hybrid parallel tests -- disabled.")
        set(TEST_MPI_THREADS OFF)
    else()
        message(STATUS "Hybrid parallel tests -- enabled.")
        set(TEST_MPI_THREADS ON)
    endif()

    # set the number of cores for oversubscription/streaming tests
    math(EXPR TWICE_TEST_CORES "${TEST_CORES}*2")

    add_subdirectory(test)
else()
    message(STATUS "regression testing -- disbaled")
endif()
