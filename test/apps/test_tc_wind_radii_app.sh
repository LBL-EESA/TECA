#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_tc_wind_radii_app.sh [app prefix] [data root] "   \
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

mask1="((track_id==4)&&(surface_wind*3.6d>=177.0d))"
mask2="((track_id==191)&&(surface_wind*3.6d>=249.0d))"
mask3="((track_id==523)&&(3.6d*surface_wind>=209.0d))"

set -x

# run the app
${launcher} ${app_prefix}/teca_tc_wind_radii                            \
    --track_file "${data_root}/tracks_1990s_3hr_mdd_4800.bin"           \
    --wind_files "${data_root}/cam5_1_amip_run2_1990s/.*\.nc$"          \
    --track_file_out test_tc_wind_radii_app_output.bin                  \
    --track_mask "!(${mask1}||${mask2}||${mask3})"                      \
    --number_of_bins "32" --profile_type "avg" --n_threads "1"          \
    --first_track "0" --last_track "-1"

# run the diff
${app_prefix}/teca_table_diff                                           \
    test_tc_wind_radii_app_output.bin                                   \
    "${data_root}/test_tc_wind_radii.bin"

# clean up
rm test_tc_wind_radii_app_output.bin
