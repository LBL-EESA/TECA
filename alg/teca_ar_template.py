from teca import *
import numpy as np
from mpi4py import *
import time


class teca_sample_track(object):
    """ This class illustrates how to sample mesh based data along a cylcone
    trajectory. The example will make a plot of the sampled data. The data is
    returned in numpy arrays and could be processed as needed instead of making
    a plot. The class is parallel and on NERSC it must be run on compute nodes.
    The class parallelizes over tracks. Each rank is given a unique set of
    tracks to process. Thus this example shows how to work with complete
    tracks.

    In the report_callback we inform the down stream about the number of tracks
    available. The sets things up for the map-reduce over tracks.

    In the request_callback we look up the requested track, get the lat lon
    coordinates of the track and generate a request to the NetCDF CF2 reader
    for a widow centered on each point of the track.

    In the execute_callback we are served the data and make a plot of each
    array.
    """
    @staticmethod
    def New():
        return teca_sample_track()

    def __init__(self):
        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_execute_callback(self.get_execute_callback())

    def set_input_connection(self, port, obj):
        self.impl.set_input_connection(obj)

    def get_output_port(self):
        return self.impl.get_output_port()

    def update(self):
        self.impl.update()

    def get_report_callback(self):
        def report_callback(port, md_in):
            # pass through by default
            return md_in
        return report_callback

    def get_request_callback(self):
        def request_callback(port, md_in, req_in):
            # pass the request through
            reqs_out = [req_in]
            return reqs_out
        return request_callback

    def get_execute_callback(self):
        def execute_callback(port, data_in, req_in):
            # return an empty table
            return teca_table.New()
        return execute_callback
