project(teca_apps)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_app_link)

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    list(APPEND teca_app_link ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_app_link ${Boost_LIBRARIES})
endif()

teca_add_app(teca_tc_detect LIBS ${teca_app_link}
    FEATURES (TECA_HAS_BOOST AND TECA_HAS_NETCDF AND TECA_HAS_UDUNITS))

teca_add_app(teca_tc_wind_radii LIBS ${teca_app_link}
    FEATURES (TECA_HAS_BOOST AND TECA_HAS_NETCDF AND TECA_HAS_UDUNITS))

teca_add_app(teca_tc_trajectory LIBS ${teca_app_link}
    FEATURES (TECA_HAS_BOOST AND TECA_HAS_UDUNITS))

teca_add_app(teca_metadata_probe LIBS ${teca_app_link}
    FEATURES (TECA_HAS_BOOST AND TECA_HAS_NETCDF AND TECA_HAS_UDUNITS))

teca_py_install_apps(
    teca_convert_table.py
    teca_dataset_metadata.py
    teca_event_filter.py
    teca_tc_stats.py
    teca_tc_wind_radii_stats.py
    teca_tc_trajectory_scalars.py
    )
