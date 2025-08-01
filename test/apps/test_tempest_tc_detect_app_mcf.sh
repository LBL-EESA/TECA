#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_tempest_tc_detect_app_mcf.sh [app prefix] [data root] " \
         "[mpi exec] [test cores] $#"
    exit -1
fi

app_prefix=${1}
data_root=${2}

if [[ $# -eq 4 ]]
then
    mpi_exec=${3}
    test_cores=${4}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_tempest_tc_detect                  \
    --input_file "${app_prefix}/../test/ERA5_TC_test.mcf"         \
    --closed_contour_cmd "MSL,200.0,5.5,0;thickness,-6.0,6.5,1.0" \
    --sea_level_pressure MSL                                      \
    --geopotential Z                                              \
    --surface_wind_u VAR_10U                                      \
    --surface_wind_v VAR_10V                                      \
    --geopotential_at_surface ZS                                  \
    --x_axis_variable longitude                                   \
    --y_axis_variable latitude                                    \
    --last_step 1                                                 \
    --candidate_file test_te_candidates_app_output.csv

# run the diff
${app_prefix}/teca_table_diff                              \
    "${data_root}/test_te_candidates_app_ref.csv"          \
    test_te_candidates_app_output.csv

# clean up
rm test_te_candidates_app_output.csv
