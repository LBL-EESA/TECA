#!/bin/bash

if [[ $# -ne 24 ]]
then
    echo "usage: test_event_filter_app.sh" \
         "[app_prefix] [input_file] [output_file] " \
         "[region_x_coords_option] [+x_crd_vals]" \
         "[region_y_coords_option] [+y_crd_vals]" \
         "[region_sizes_option] [region_sizes]" \
         "[time_column_option] [time_column]" \
         "[start_time_option] [start_time]" \
         "[end_time_option] [end_time]" \
         "[output_ref]"
    exit 1
fi

# First command
app_name="teca_event_filter"
app_prefix=${1}
input_file=${2}
output_file=${3}
region_x_coords_option=${4}
x_crd_vals="${5} ${6} ${7} ${8} ${9}"
region_y_coords_option=${10}
y_crd_vals="${11} ${12} ${13} ${14} ${15}"
region_sizes_option=${16}
region_sizes=${17}
time_column_option=${18}
time_column=${19}
start_time_option=${20}
start_time=${21}
end_time_option=${22}
end_time=${23}

# Second command
output_ref="${24}"

app_exec="${app_prefix}/${app_name}"
verifier_exec="${app_prefix}/teca_table_diff"

${app_exec} ${input_file} ${output_file} \
    ${region_x_coords_option} ${x_crd_vals} \
    ${region_y_coords_option} ${y_crd_vals} \
    ${region_sizes_option} ${region_sizes} \
    ${time_column_option} ${time_column} \
    ${start_time_option} ${start_time} \
    ${end_time_option} ${end_time}

${verifier_exec} ${output_file} ${output_ref}
