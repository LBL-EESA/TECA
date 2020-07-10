#!/bin/bash

if [[ $# -ne 2 ]]
then
    echo "usage: test_tc_wind_radii_stats_app.sh [app prefix] "    \
         "[data root]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

output_prefix="test_tc_wind_radii_stats_app_output"

set -x

# run the app
${app_prefix}/teca_tc_wind_radii_stats                             \
    "${data_root}/tracks_1990s_3hr_mdd_4800_median_in_cat_wr.bin"  \
    ${output_prefix}

# check if number of outputs are correct
output_list=($(ls ${output_prefix}*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 2 ]]
then
    echo "error: unexpected number of tc_wind_radii_stats outputs"
    exit 1
fi

# check if outputs aren't empty
for i in ${output_list[@]}
do
    if [[ ! -s $i ]]
    then
        echo "error: file '$i' is empty"
        exit 1
    fi
done

# clean up
rm ${output_prefix}*
