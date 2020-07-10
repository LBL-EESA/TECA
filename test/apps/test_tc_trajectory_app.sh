#!/bin/bash

if [[ $# -ne 2 ]]
then
    echo "usage: test_tc_trajectory_app.sh [app prefix] [data root]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

set -x

# run the app
${launcher} ${app_prefix}/teca_tc_trajectory                                \
    --candidate_file "${data_root}/cam5_1_amip_run2_candidates_2005_09.bin" \
    --track_file test_tc_trajectory_app_output.bin                          \
    --max_daily_distance 1600 --min_wind_speed 17 --min_wind_duration 2     

# run the diff
${app_prefix}/teca_table_diff test_tc_trajectory_app_output.bin             \
    "${data_root}/cam5_1_amip_run2_tracks_2005_09.bin"

# clean up
rm test_tc_trajectory_app_output.bin
