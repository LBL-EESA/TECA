#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_detect_nodes_app_mcf.sh [app prefix] [data root] "     \
         "[mpi exec] [test cores]"
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
${launcher} ${app_prefix}/teca_detect_nodes                              \
    --input_file "${app_prefix}/../test/ERA5_TC_test.mcf"                \
    --sea_level_pressure MSL --geopotential Z                            \
    --surface_wind_u VAR_10U --surface_wind_v VAR_10V                    \
    --geopotential_at_surface ZS                                         \
    --x_axis_variable longitude --y_axis_variable latitude               \
    --max_lat 90 --min_lat -90 --max_lon 359.5 --min_lon 0 --last_step 1 \
    --candidate_file test_detect_nodes_app_output.csv

# run the diff
#${app_prefix}/teca_table_diff                                            \
#    "${data_root}/test_detect_nodes_app_ref.csv"                         \
#    test_detect_nodes_app_output.csv "storm_id"
#
# clean up
#rm test_detect_nodes_app_output.csv
