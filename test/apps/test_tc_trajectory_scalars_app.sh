#!/bin/bash

if [[ $# -ne 7 ]]
then
    echo "usage: test_tc_trajectory_scalars_app.sh" \
         "[app_prefix] [tracks_file] [output_prefix]" \
         "[first_track_option] [first_track]" \
         "[last_track_option] [last_track]"
    exit 1
fi

# First command
app_name="teca_tc_trajectory_scalars"
app_prefix=${1}
tracks_file=${2}
output_prefix="${3}"
first_track_option=${4}
first_track="${5}"
last_track_option=${6}
last_track="${7}"

app_exec="${app_prefix}/${app_name}"

${app_exec} ${tracks_file} ${output_prefix} \
    ${first_track_option} ${first_track} \
    ${last_track_option} ${last_track}

output_list=($(ls ${output_prefix}_*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 6 ]]
then
    echo "error: unexpected number of tc_trajectory_scalars outputs"
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
