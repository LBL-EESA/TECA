
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

teca_add_test(test_deeplabv3p_ar_detect_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_deeplabv3p_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    --pytorch_deeplab_model ${TECA_DATA_ROOT}/cascade_deeplab_IVT.pt
    --input_regex "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D.*\.nc$"
    --filename_time_template "ARTMIP_MERRA_2D_%Y%m%d_%H.nc"
    --t_axis_variable ''
    --t_units "days since 1979-01-01 00:00:00"
    --output_file "${TECA_DATA_ROOT}/test_deeplabv3p_ar_detect_app_output.nc"
    0 "${TECA_DATA_ROOT}/test_deeplabv3p_ar_detect_app_ref.bin"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    --input_regex "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D.*\.nc$"
    --cf_reader::filename_time_template "ARTMIP_MERRA_2D_%Y%m%d_%H.nc"
    --cf_reader::t_axis_variable ''
    --cf_reader::t_units "days since 1979-01-01 00:00:00"
    --output_file "${TECA_DATA_ROOT}/test_bayesian_ar_detect_app_output.nc"
    0 "${TECA_DATA_ROOT}/test_bayesian_ar_detect_app_ref.bin"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_detect_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    --input_regex "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    --candidate_file "${TECA_DATA_ROOT}/test_tc_candidates_app_output.bin"
    --track_file "${TECA_DATA_ROOT}/test_tc_track_app_output.bin"
    --lowest_lat -20 --highest_lat 20
    "${TECA_DATA_ROOT}/test_tc_candidates_app_ref.bin"
    "${TECA_DATA_ROOT}/test_tc_track_app_ref.bin" "storm_id"
    REQ_TECA_DATA)

teca_add_test(test_tc_trajectory_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    --candidate_file "${TECA_DATA_ROOT}/cam5_1_amip_run2_candidates_2005_09.bin"
    --track_file "${TECA_DATA_ROOT}/test_tc_trajectory_app_output.bin"
    --max_daily_distance 1600
    --min_wind_speed 17
    --min_wind_duration 2
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin"
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    --track_file "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    --wind_files "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$"
    --track_file_out "${TECA_DATA_ROOT}/test_tc_wind_radii_app_output.bin"
    --track_mask "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    --number_of_bins "32" --profile_type "avg" --n_threads "1"
    --first_track "0" --last_track "-1"
    "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_stats_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_stats_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin"
    "${TECA_DATA_ROOT}/test_tc_stats_app_output"
    "${TECA_DATA_ROOT}/test_tc_stats_app_output_class_table_ref.bin"
    REQ_TECA_DATA)

teca_add_test(test_tc_trajectory_scalars_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_scalars_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800_median_in_cat_wr.bin"
    "${TECA_DATA_ROOT}/test_tc_trajectory_scalars_app_output"
    --first_track 0
    --last_track -1
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_stats_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_stats_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800_median_in_cat_wr.bin"
    "${TECA_DATA_ROOT}/test_tc_wind_radii_stats_app_output"
    REQ_TECA_DATA)

teca_add_test(test_event_filter_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_event_filter_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_event_filter_app_output.bin"
    --region_x_coords 180 180 270 270 180
    --region_y_coords 10 10 10 -10 -10
    --region_sizes 5 --time_column time
    --start_time 4196.23 --end_time 4196.39
    "${TECA_DATA_ROOT}/test_event_filter_app_ref.bin"
    REQ_TECA_DATA)
