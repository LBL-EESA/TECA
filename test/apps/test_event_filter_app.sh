#!/bin/bash

if [[ $# -ne 2 ]]
then
    echo "usage: test_event_filter_app.sh [app prefix] " \
         "[data root]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

set -x

# run the app
${app_prefix}/teca_event_filter                          \
    "${data_root}/test_tc_candidates_20.bin"             \
    test_event_filter_app_output.bin                     \
    --region_x_coords 180 180 270 270 180                \
    --region_y_coords 10 10 10 -10 -10                   \
    --region_sizes 5 --time_column time                  \
    --start_time 4196.23 --end_time 4196.39

# run the diff
${app_prefix}/teca_table_diff                            \
    test_event_filter_app_output.bin                     \
    "${data_root}/test_event_filter_app_ref.bin"

# clean up
rm test_event_filter_app_output.bin
