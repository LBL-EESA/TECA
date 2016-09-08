#!/usr/bin/env python
from teca import *
import sys

class teca_dataset_metadata:
    """
    extracts time information from the input and stores
    it in an output table
    """
    @staticmethod
    def New():
        return teca_dataset_metadata()

    def __init__(self):
        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_execute_callback(self))

    def __str__(self):
        return ''

    def set_input_connection(self, obj):
        """
        set the input
        """
        self.impl.set_input_connection(obj)

    def get_output_port(self):
        """
        get the output
        """
        return self.impl.get_output_port()

    def update(self):
        """
        execute the pipeline from this algorithm up.
        """
        self.impl.update()

    @staticmethod
    def get_execute_callback(state):
        """
        return a teca_algorithm::execute function. a closure
        is used to gain state.
        """
        def execute(port, data_in, req):
            """
            create a table containing temporal metadata
            """
            mesh = as_teca_cartesian_mesh(data_in[0])

            table = teca_table.New()
            table.copy_metadata(mesh)
            table.declare_columns(['time_step', 'time'], ['ul', 'd'])
            table << mesh.get_time_step() << mesh.get_time()

            return table
        return execute



argc = len(sys.argv)
if argc < 3:
    sys.stderr.write('sends dataset metadata to a table\n\n')
    sys.stderr.write('teca_dataset_metadata [dataset regex] ' \
        '[output file] [first step] [last step] [n threads]\n\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]
first_step = 0 if argc < 4 else int(sys.argv[3])
last_step = -1 if argc < 5 else int(sys.argv[4])
n_threads = 1 if argc < 6 else int(sys.argv[5])

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)

mdf = teca_dataset_metadata.New()
mdf.set_input_connection(cfr.get_output_port())

mr = teca_table_reduce.New()
mr.set_input_connection(mdf.get_output_port())
mr.set_first_step(first_step)
mr.set_last_step(last_step)
mr.set_thread_pool_size(n_threads)

sort = teca_table_sort.New()
sort.set_input_connection(mr.get_output_port())
sort.set_index_column('time')

cal = teca_table_calendar.New()
cal.set_input_connection(sort.get_output_port())

tw = teca_table_writer.New()
tw.set_input_connection(cal.get_output_port())
tw.set_file_name(out_file)

tw.update()
