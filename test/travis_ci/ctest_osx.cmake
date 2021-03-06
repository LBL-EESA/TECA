set(CTEST_SITE "Travis_CI")
set(CTEST_BUILD_NAME "Apple_Mac_OS_$ENV{TRAVIS_BRANCH}_$ENV{SHA}")
set(CTEST_DASHBOARD_ROOT "$ENV{DASHROOT}")
set(CTEST_BUILD_CONFIGURATION $ENV{BUILD_TYPE})
set(CTEST_TEST_ARGS PARALLEL_LEVEL 1)
set(CTEST_BUILD_FLAGS -j4)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_PROJECT_NAME "TECA")
set(CTEST_NOTES_FILES)
set(ALWAYS_UPDATE ON)
set(INCREMENTAL_BUILD OFF)
set(DASHBOARD_TRACK "Continuous")
set(DATA_ROOT "${CTEST_DASHBOARD_ROOT}/TECA_data")
set(CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}")
set(CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/build")
set(CTEST_TEST_TIMEOUT 800)
set(INITIAL_CACHE
"CMAKE_BUILD_TYPE=$ENV{BUILD_TYPE}
BUILD_TESTING=ON
TECA_ENABLE_PROFILER=ON
TECA_PYTHON_VERSION=$ENV{TECA_PYTHON_VERSION}
TECA_DATA_ROOT=$ENV{DASHROOT}/TECA_data
TECA_TEST_CORES=2
HYPERTHREADS_PER_CORE=1
REQUIRE_OPENSSL=TRUE
OPENSSL_ROOT_DIR=/usr/local/opt/openssl@1.1
REQUIRE_BOOST=TRUE
REQUIRE_NETCDF=TRUE
REQUIRE_NETCDF_MPI=FALSE
REQUIRE_UDUNITS=TRUE
REQUIRE_MPI=TRUE
REQUIRE_PYTHON=TRUE
REQUIRE_TECA_DATA=TRUE")
file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" ${INITIAL_CACHE})
ctest_start("${DASHBOARD_TRACK}")
ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")
ctest_configure(RETURN_VALUE terr)
ctest_submit(PARTS Update Configure RETRY_DELAY 15 RETRY_COUNT 0)
if (NOT terr EQUAL 0)
    message(FATAL_ERROR "ERROR: ctest configure failed!")
endif()
ctest_build(RETURN_VALUE terr)
ctest_submit(PARTS Build RETRY_DELAY 15 RETRY_COUNT 0)
if (NOT terr EQUAL 0)
    message(FATAL_ERROR "ERROR: ctest build failed!")
endif()
ctest_test(RETURN_VALUE terr)
ctest_submit(PARTS Test RETRY_DELAY 15 RETRY_COUNT 0)
if (NOT terr EQUAL 0)
    message(FATAL_ERROR "ERROR: ctest test failed!")
endif()
message(STATUS "testing complete")
