project(teca_apps)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_system,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_app_link)

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_app_link ${Boost_LIBRARIES})
endif()

teca_add_app(teca_bayesian_ar_detect LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS})

teca_add_app(teca_integrated_vapor_transport LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS})

teca_add_app(teca_integrated_water_vapor LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS})

teca_add_app(teca_tc_detect LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS})

teca_add_app(teca_tc_wind_radii LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS})

teca_add_app(teca_tc_trajectory LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_UDUNITS})

teca_add_app(teca_metadata_probe LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS})

teca_add_app(teca_cf_restripe LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS})

teca_add_app(teca_table_diff LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_NETCDF})

teca_add_app(teca_cartesian_mesh_diff LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_NETCDF})

teca_add_python_app(teca_numpy_array_diff)

teca_add_app(teca_cpp_temporal_reduction LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_NETCDF})

teca_add_app(teca_regional_moisture_flux LIBS ${teca_app_link}
    FEATURES ${TECA_HAS_BOOST} ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
             ${TECA_HAS_SHAPELIB})

teca_add_python_app(teca_convert_table)
teca_add_python_app(teca_dataset_metadata FEATURES ${TECA_HAS_NETCDF})

teca_add_python_app(teca_deeplab_ar_detect
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_PYTORCH})

teca_add_python_app(teca_temporal_reduction FEATURES ${TECA_HAS_NETCDF})
teca_add_python_app(teca_event_filter)
teca_add_python_app(teca_profile_explorer)
teca_add_python_app(teca_tc_stats)
teca_add_python_app(teca_tc_wind_radii_stats)
teca_add_python_app(teca_tc_trajectory_scalars)

teca_add_python_app(teca_potential_intensity
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_TCPYPI})

teca_add_python_app(teca_lapse_rate)
teca_add_python_app(teca_spectral_filter FEATURES ${TECA_HAS_NETCDF})

# Installing helper functions for blocking event apps manually to binary
# directory since listing them as SOURCES in teca_add_python_app does remove
# the .py extension.
# TODO/FIXME: Should the helper functions be installed to a different
# directory than the apps? If yes, which one?
if(TECA_HAS_PYTHON)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/teca_blocking_event_helper_functions.py
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../${BIN_PREFIX})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/teca_blocking_event_operations.py
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../${BIN_PREFIX})
endif()
teca_add_python_app(teca_blocking_event_detect_blobs FEATURES ${TECA_HAS_NETCDF})
teca_add_python_app(teca_blocking_event_create_overlap_table FEATURES ${TECA_HAS_NETCDF})
teca_add_python_app(teca_blocking_event_tracker)
teca_add_python_app(teca_blocking_event_relabel FEATURES ${TECA_HAS_NETCDF})
