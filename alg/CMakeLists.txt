project(teca_alg)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_alg_cxx_srcs
    gfdl_tc_candidates.cxx
    teca_apply_binary_mask.cxx
    teca_binary_segmentation.cxx
    teca_bayesian_ar_detect.cxx
    teca_cartesian_mesh_subset.cxx
    teca_cartesian_mesh_regrid.cxx
    teca_connected_components.cxx
    teca_2d_component_area.cxx
    teca_component_area_filter.cxx
    teca_component_statistics.cxx
    teca_dataset_capture.cxx
    teca_dataset_source.cxx
    teca_derived_quantity.cxx
    teca_descriptive_statistics.cxx
    teca_evaluate_expression.cxx
    teca_geography.cxx
    teca_l2_norm.cxx
    teca_latitude_damper.cxx
    teca_laplacian.cxx
    teca_mask.cxx
    teca_parser.cxx
    teca_programmable_algorithm.cxx
    teca_programmable_reduce.cxx
    teca_table_calendar.cxx
    teca_table_reduce.cxx
    teca_table_region_mask.cxx
    teca_table_remove_rows.cxx
    teca_table_sort.cxx
    teca_table_to_stream.cxx
    teca_tc_candidates.cxx
    teca_tc_classify.cxx
    teca_tc_wind_radii.cxx
    teca_tc_trajectory.cxx
    teca_temporal_average.cxx
    teca_variant_array_operand.cxx
    teca_vorticity.cxx
    teca_dataset_diff.cxx
    )

teca_py_install(${LIB_PREFIX}
    teca_tc_stats.py
    teca_tc_activity.py
    teca_tc_wind_radii_stats.py
    teca_tc_trajectory_scalars.py
    )

set(teca_alg_f90_srcs)

set(teca_alg_f90_generics
    gfdl_spline
    gfdl_tc_candidates
    )

set(f_type real)
set(c_types float double)
foreach(generic_src ${teca_alg_f90_generics})
    foreach(c_type_var ${c_types})
        set(iso_c_type_var "${f_type}(c_${c_type_var})")
        string(SUBSTRING ${c_type_var} 0 1 var_name)
        foreach(c_type_coord ${c_types})
            string(SUBSTRING ${c_type_coord} 0 1 coord_name)
            set(decorator "c${coord_name}_v${var_name}")
            set(iso_c_type_coord "${f_type}(c_${c_type_coord})")
            configure_file(${generic_src}.f90.in ${generic_src}_${decorator}.f90 @ONLY)
            list(APPEND teca_alg_f90_srcs ${generic_src}_${decorator}.f90)
        endforeach()
    endforeach()
endforeach()

include(FortranCInterface)
FortranCInterface_VERIFY()

set(teca_alg_link pthread)

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_alg_link ${Boost_LIBRARIES})
endif()

add_library(teca_alg ${teca_alg_cxx_srcs} ${teca_alg_f90_srcs})
target_link_libraries(teca_alg teca_io teca_data teca_core ${teca_alg_link})

target_include_directories(teca_alg
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${INCLUDE_PREFIX}
    FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_alg
    EXPORT teca_alg
    INCLUDES DESTINATION ${INCLUDE_PREFIX}
    ARCHIVE DESTINATION ${LIB_PREFIX}
    LIBRARY DESTINATION ${LIB_PREFIX}
    )

install(EXPORT teca_alg
    DESTINATION ${LIB_PREFIX}
    FILE teca_alg.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
