#!/bin/bash

if [[ $# -ne 3 ]]
then
    echo "usage: test_tc_wind_radii_stats_app.sh" \
         "[app_prefix] [tracks_file] [output_prefix]"
    exit 1
fi

# First command
app_name="teca_tc_wind_radii_stats"
app_prefix=${1}
tracks_file=${2}
output_prefix=${3}

app_exec="${app_prefix}/${app_name}"

${app_exec} ${tracks_file} ${output_prefix}

output_list=($(ls ${output_prefix}*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 2 ]]
then
    echo "error: unexpected number of tc_wind_radii_stats outputs"
    exit 1
fi

for i in ${output_list[@]}
do
    if [[ ! -s $i ]]
    then
        echo "error: file '$i' is empty"
        exit 1
    fi
done
