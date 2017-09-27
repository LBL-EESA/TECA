import sys
import teca_py
import numpy as np

class teca_ar_shields_kiehl(object):
    """ """
    @staticmethod
    def New():
        return teca_ar_shields_kiehl()

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
