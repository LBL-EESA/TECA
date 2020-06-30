#!/bin/bash

if [[ $# -ne 12 ]]
then
    echo "usage: test_tc_trajectory_app.sh" \
         "[app_prefix] [candidates_file_option] [candidates_file] " \
         "[output_file_option] [output_file]" \
         "[max_daily_distance_option] [max_daily_distance]" \
         "[min_wind_speed_option] [min_wind_speed]" \
         "[min_wind_duration_option] [min_wind_duration]" \
         "[output_ref]"
    exit 1
fi

# First command
app_name="teca_tc_trajectory"
app_prefix=${1}
candidates_file_option=${2}
candidates_file=${3}
output_file_option=${4}
output_file="${5}"
max_daily_distance_option=${6}
max_daily_distance="${7}"
min_wind_speed_option=${8}
min_wind_speed="${9}"
min_wind_duration_option=${10}
min_wind_duration="${11}"

# Second command
output_ref="${12}"

app_exec="${app_prefix}/${app_name}"
verifier_exec="${app_prefix}/teca_table_diff"

${app_exec} ${candidates_file_option} ${candidates_file} \
    ${output_file_option} ${output_file} \
    ${max_daily_distance_option} ${max_daily_distance} \
    ${min_wind_speed_option} ${min_wind_speed} \
    ${min_wind_duration_option} ${min_wind_duration}

${verifier_exec} ${output_file} ${output_ref}
